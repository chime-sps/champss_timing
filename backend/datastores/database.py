import sqlite3
import time
import json
import shutil
import os

from ..utils.utils import utils
from ..utils.logger import logger

class database:
    """
    Database structure:
    ---
    Table: info
    Columns: version
    ---
    Table: timing_info
    Columns: timestamp, files, obs_mjds, unfreeze_params, residuals, chi2, chi2_reduced, fitted_params, notes
    timestamp unique
    ---
    Table: toas
    Columns: timestamp, filename, freq, toa, toa_err, telescope, raw_tim, notes
    timestamp unique
    filename unique
    ---
    Table: archive_info
    Columns: timestamp, filename, psr_amps, psr_snr, notes
    timestamp unique
    filename unique
    ---
    Table: dealias_history
    Columns: timestamp, n_stacked, alias_factor, snr_stacked, notes
    timestamp unique
    ---
    Table: config
    Columns: key, value
    key unique
    """
    def __init__(self, psr_db, readonly=False, logger=logger()):
        self.version = "1.1"
        self.readonly = readonly
        self.psr_db = None
        self.logger = logger

        if not os.path.exists(os.path.dirname(psr_db)):
            raise Exception(f"Database folder {os.path.dirname(psr_db)} does not exist. Please provide a valid path for psr_db.")
        
        if self.readonly:
            # check if database exists
            if not os.path.exists(psr_db):
                raise Exception(f"Database {psr_db} does not exist. Please provide a valid database file.")

            # copy the database to a temporary file
            self.psr_db = os.path.abspath(f"{psr_db}.readonly{utils.get_rand_string()}.tmp")
            shutil.copyfile(psr_db, self.psr_db)
            self.logger.debug(f"Readonly temporary database created at {self.psr_db}")

            # open the temporary database in readonly mode
            self.conn = sqlite3.connect("file://" + self.psr_db + "?mode=ro", uri=True, check_same_thread=False)
        else:
            self.psr_db = psr_db
            self.conn = sqlite3.connect(self.psr_db)
        self.cur = self.conn.cursor()

    def initialize(self, self_check=True):
        if self.readonly:
            return
        
        if not os.path.exists(self.psr_db):
            self.logger.info(f"Creating database {self.psr_db}")

        # create tables
        self.cur.execute("CREATE TABLE IF NOT EXISTS info (version TEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS timing_info (timestamp INT, files LONGTEXT, obs_mjds LONGTEXT, unfreeze_params LONGTEXT, residuals LONGTEXT, chi2 REAL, chi2_reduced REAL, fitted_params TEXT, notes LONGTEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS toas (timestamp INT, filename TEXT, freq REAL, toa REAL, toa_err REAL, telescope TEXT, raw_tim LONGTEXT, notes LONGTEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS archive_info (timestamp INT, filename TEXT, psr_amps LONGTEXT, psr_snr REAL, notes LONGTEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS dealias_history (timestamp INT, n_stacked INT, alias_factor REAL, snr_stacked REAL, notes LONGTEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS config (key TEXT, value TEXT)")
        
        # create indices
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp ON toas (timestamp)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_filename ON toas (filename)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp_timing ON timing_info (timestamp)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_filename_archive ON archive_info (filename)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp_archive ON archive_info (timestamp)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp_dealias ON dealias_history (timestamp)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_key_config ON config (key)")

        # insert version
        # self.cur.execute("INSERT INTO info (version) VALUES (?)", (self.version,))
        self.cur.execute("SELECT version FROM info")
        version = self.cur.fetchone()
        if version is None:
            self.cur.execute("INSERT INTO info (version) VALUES (?)", (self.version,))
        
        # check integrity
        if self_check:
            self.self_check()

        self.conn.commit()

    def self_check(self):
        # Check if filenames in toas and archive_info are consistent

        ## Get filenames from toas and archive_info
        self.cur.execute("SELECT filename FROM toas")
        toas_filenames = self.cur.fetchall()
        toas_filenames = [filename[0] for filename in toas_filenames]

        ## Get filenames from archive_info
        self.cur.execute("SELECT filename FROM archive_info")
        archive_filenames = self.cur.fetchall()
        archive_filenames = [filename[0] for filename in archive_filenames]

        ## Check if filenames are consistent
        for filename in toas_filenames:
            if filename not in archive_filenames:
                self.logger.warning(f"WARNING: Filename {filename} in table[toas] but not in table[archive_info]. Removing entry filename={filename} from table[toas]")
                self.cur.execute("DELETE FROM toas WHERE filename = ?", (filename,))
                self.conn.commit()

        for filename in archive_filenames:
            if filename not in toas_filenames:
                self.logger.warning(f"WARNING: Filename {filename} in table[archive_info] but not in table[toas]. Removing entry filename={filename} from table[archive_info]")
                self.cur.execute("DELETE FROM archive_info WHERE filename = ?", (filename,))
                self.conn.commit()

        # Check if files in timing_info are consistent with filenames in toas

        ## Get filenames from toas and archive_info again (it might have been modified)
        self.cur.execute("SELECT filename FROM toas")
        toas_filenames = self.cur.fetchall()
        toas_filenames = [filename[0] for filename in toas_filenames]

        ## Get files from timing_info
        self.cur.execute("SELECT files FROM timing_info")
        files = self.cur.fetchall()
        files = [json.loads(file[0]) for file in files]

        ## Check if filenames are consistent
        for file in files:
            for filename in file:
                if filename not in toas_filenames:
                    self.logger.error(f"VERY IMPORTANT WARNING: Filename \"{filename}\" in table[timing_info] but not in table[toas]. This might cause issues with plotting and due to errors in the processing. Please resolve this issue manually.")
    
    def truncate_timing_info(self, show_warning=True):
        if show_warning:
            self.logger.warning("WARNING: Emptying timing_info table")
            input("Press Enter to continue...")
        self.cur.execute("DELETE FROM timing_info")
        self.conn.commit()

    def remove_timing_info(self, mjd_later_than, show_warning=True):
        if show_warning:
            self.logger.warning(f"WARNING: Removing timing_info entries with obs_mjd later than {mjd_later_than}")
            input("Press Enter to continue...")

        timing_info = self.get_all_timing_info()
        for this_info in timing_info:
            if max(this_info["obs_mjds"]) > mjd_later_than:
                self.logger.warning(f"Removing timing_info entry with timestamp={this_info['timestamp']}, mjd={max(this_info['obs_mjds'])}")
                self.cur.execute("DELETE FROM timing_info WHERE timestamp = ?", (this_info["timestamp"],))
        
        self.conn.commit()

    def insert_toa(self, filename, freq, toa, toa_err, telescope, raw_tim, notes, timestamp="auto", commit=True):
        notes = json.dumps(notes)
        
        if timestamp == "auto":
            timestamp = time.time()

        self.cur.execute("INSERT INTO toas (timestamp, filename, freq, toa, toa_err, telescope, raw_tim, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (timestamp, filename, freq, toa, toa_err, telescope, raw_tim, notes))
        
        if commit:
            self.conn.commit()

    def get_all_toas(self):
        self.cur.execute("SELECT * FROM toas ORDER BY timestamp")
        toas_raw = self.cur.fetchall()

        toas = []
        for toa in toas_raw:
            toas.append(self.format_toa(toa))

        return toas
    
    def get_last_toa(self):
        self.cur.execute("SELECT * FROM toas ORDER BY timestamp DESC LIMIT 1")
        return self.format_toa(self.cur.fetchone())
    
    def get_toa_by_filename(self, filename):
        self.cur.execute("SELECT * FROM toas WHERE filename = ?", (filename,))
        return self.format_toa(self.cur.fetchone())
    
    def check_toa_exists(self, filename):
        self.cur.execute("SELECT EXISTS(SELECT 1 FROM toas WHERE filename = ?)", (filename,))
        return self.cur.fetchone()[0]
    
    def get_toa_by_mjd(self, mjd_start, mjd_end):
        self.cur.execute(f"SELECT * FROM toas ORDER BY timestamp WHERE toa > {mjd_start} AND toa < {mjd_end}")
        toas_raw = self.cur.fetchall()

        toas = []
        for toa in toas_raw:
            toas.append(self.format_toa(toa))

        return toas
    
    def format_toa(self, toa):     
        if toa is None:  
            toa = [0, "", 0, 0, 0, "", "", "{}"]
        
        formatted_toa = {
            "timestamp": toa[0],
            "filename": toa[1],
            "freq": toa[2],
            "toa": toa[3],
            "toa_err": toa[4],
            "telescope": toa[5],
            "raw_tim": toa[6],
            "notes": json.loads(toa[7])
        }

        if "label" not in formatted_toa["notes"]:
            formatted_toa["notes"]["label"] = "NO_LABEL"

        if "rcvr" not in formatted_toa["notes"]:
            formatted_toa["notes"]["rcvr"] = "unknown"

        return formatted_toa

    def insert_timing_info(self, files, obs_mjds, unfreeze_params, residuals, chi2, chi2_reduced, fitted_params, notes, timestamp="auto", commit=True):
        files = json.dumps(files)
        obs_mjds = json.dumps(obs_mjds)
        residuals = json.dumps(residuals)
        unfreeze_params = json.dumps(unfreeze_params)
        fitted_params = json.dumps(fitted_params)
        notes = json.dumps(notes)
        
        if timestamp == "auto":
            timestamp = time.time()

        self.cur.execute("INSERT INTO timing_info (timestamp, files, obs_mjds, unfreeze_params, residuals, chi2, chi2_reduced, fitted_params, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (timestamp, files, obs_mjds, unfreeze_params, residuals, chi2, chi2_reduced, fitted_params, notes))
        
        if commit:
            self.conn.commit()
     
    def get_all_timing_info(self, mjd_sort=False):
        self.cur.execute("SELECT * FROM timing_info ORDER BY timestamp")
        timing_info_raw = self.cur.fetchall()

        timing_info = []
        for info in timing_info_raw:
            timing_info.append(self.format_timing_info(info))

        if mjd_sort:
            return self.sort_timing_info(timing_info)

        return timing_info
    
    def get_last_timing_info(self):
        self.cur.execute("SELECT * FROM timing_info ORDER BY timestamp DESC LIMIT 1")
        return self.format_timing_info(self.cur.fetchone())

    def format_timing_info(self, timing_info):
        if timing_info is None:
            timing_info = [0, "[]", "[]", "[]", "{}", 0, 0, "[]", "{}"]
        
        formatted_info = {
            "timestamp": timing_info[0],
            "files": json.loads(timing_info[1]),
            "obs_mjds": json.loads(timing_info[2]),
            "unfreeze_params": json.loads(timing_info[3]),
            "residuals": json.loads(timing_info[4]),
            "chi2": timing_info[5],
            "chi2_reduced": timing_info[6],
            "fitted_params": json.loads(timing_info[7]),
            "notes": json.loads(timing_info[8])
        }

        if "bad_toa_mjds" not in formatted_info["notes"]:
            formatted_info["notes"]["bad_toa_mjds"] = []
        
        if "bad_toa_residuals" not in formatted_info["notes"]:
            formatted_info["notes"]["bad_toa_residuals"] = {"val": [], "err": []}
        
        if "fitted_parfile" not in formatted_info["notes"]:
            formatted_info["notes"]["fitted_parfile"] = "NO_PARFILE_PROVIDED"

        if "fitted_summary" not in formatted_info["notes"]:
            formatted_info["notes"]["fitted_summary"] = "NO_SUMMARY_PROVIDED"

        if "remark" not in formatted_info["notes"]:
            formatted_info["notes"]["remark"] = []

        return formatted_info

    def sort_timing_info(self, timing_info):
        timing_info_mjd_idx = []
        for i, info in enumerate(timing_info):
            timing_info_mjd_idx.append((i, len(info["obs_mjds"])))

        timing_info_mjd_idx = sorted(timing_info_mjd_idx, key=lambda x: x[1])

        sorted_timing_info = []
        for idx in timing_info_mjd_idx:
            sorted_timing_info.append(timing_info[idx[0]])

        return sorted_timing_info

    
    def insert_archive_info(self, filename, psr_amps, psr_snr, notes, timestamp="auto", commit=True):
        psr_amps = json.dumps(psr_amps)
        notes = json.dumps(notes)
        
        if timestamp == "auto":
            timestamp = time.time()

        self.cur.execute("INSERT INTO archive_info (timestamp, filename, psr_amps, psr_snr, notes) VALUES (?, ?, ?, ?, ?)", (timestamp, filename, psr_amps, psr_snr, notes))
        
        if commit:
            self.conn.commit()

    def update_archive_info(self, filename=None, psr_amps=None, psr_snr=None, notes=None, commit=True):
        if filename is None:
            raise Exception("Filename must be provided")
            
        if psr_amps is None and psr_snr is None and notes is None:
            raise Exception("At least one of psr_amps, psr_snr, or notes must be provided")
        
        sql = "UPDATE archive_info SET "
        sql_values = []
        if psr_amps is not None:
            sql += "psr_amps = ?, "
            sql_values.append(json.dumps(psr_amps))
        if psr_snr is not None:
            sql += "psr_snr = ?, "
            sql_values.append(psr_snr)
        if notes is not None:
            sql += "notes = ?, "
            sql_values.append(json.dumps(notes))

        sql += "timestamp = ?, "
        sql_values.append(time.time())

        sql = sql[:-2] + " WHERE filename = ?"
        sql_values.append(filename)

        self.cur.execute(sql, sql_values)

        if commit:
            self.conn.commit()

    def update_archive_amps_info_many(self, filenames, amps, snrs, commit=True):
        args = []
        for i, filename in enumerate(filenames):
            args.append((json.dumps(amps[i]), snrs[i], time.time(), filename))

        self.cur.executemany("UPDATE archive_info SET psr_amps = ?, psr_snr = ?, timestamp = ? WHERE filename = ?", args)

        if commit:
            self.conn.commit()

    def get_all_archive_info(self):
        self.cur.execute("SELECT * FROM archive_info ORDER BY timestamp")
        archive_info_raw = self.cur.fetchall()

        archive_info = []
        for info in archive_info_raw:
            archive_info.append(self.format_archive_info(info))

        return archive_info
    
    def get_last_archive_info(self):
        self.cur.execute("SELECT * FROM archive_info ORDER BY timestamp DESC LIMIT 1")
        return self.format_archive_info(self.cur.fetchone())
    
    def get_archive_info_by_filename(self, filename):
        self.cur.execute("SELECT * FROM archive_info WHERE filename = ?", (filename,))
        return self.format_archive_info(self.cur.fetchone())
    
    def format_archive_info(self, archive_info):
        if archive_info is None:
            archive_info = [0, "", "[]", 0, "{}"]
        
        formatted_info = {
            "timestamp": archive_info[0],
            "filename": archive_info[1],
            "psr_amps": json.loads(archive_info[2]),
            "psr_snr": archive_info[3],
            "notes": json.loads(archive_info[4])
        }

        if self.get_version() < 1.1:
            formatted_info["notes"]["md5"] = "" # md5 information should be present in the notes
        
        if "rcvr" not in formatted_info["notes"]:
            formatted_info["notes"]["rcvr"] = "unknown"

        return formatted_info

    def insert_dealias_history(self, n_stacked, alias_factor, snr_stacked, notes, timestamp="auto", commit=True):
        notes = json.dumps(notes)
        
        if timestamp == "auto":
            timestamp = time.time()

        self.cur.execute("INSERT INTO dealias_history (timestamp, n_stacked, alias_factor, snr_stacked, notes) VALUES (?, ?, ?, ?, ?)", (timestamp, n_stacked, alias_factor, snr_stacked, notes))
        
        if commit:
            self.conn.commit()

    def get_all_dealias_history(self):
        self.cur.execute("SELECT * FROM dealias_history ORDER BY timestamp")
        dealias_history_raw = self.cur.fetchall()

        dealias_history = []
        for info in dealias_history_raw:
            dealias_history.append(self.format_dealias_history(info))

        return dealias_history

    def get_last_dealias_history(self):
        self.cur.execute("SELECT * FROM dealias_history ORDER BY timestamp DESC LIMIT 1")
        return self.format_dealias_history(self.cur.fetchone())

    def format_dealias_history(self, dealias_history):
        if dealias_history is None:
            dealias_history = [0, 0, 0, 0, "{}"]
        
        formatted_info = {
            "timestamp": dealias_history[0],
            "n_stacked": dealias_history[1],
            "alias_factor": dealias_history[2],
            "snr_stacked": dealias_history[3],
            "notes": json.loads(dealias_history[4])
        }

        if "remark" not in formatted_info["notes"]:
            formatted_info["notes"]["remark"] = ""

        return formatted_info
    
    def truncate_dealias_history(self, show_warning=True):
        if show_warning:
            self.logger.warning("WARNING: Emptying dealias_history table")
            input("Press Enter to continue...")
        self.cur.execute("DELETE FROM dealias_history")
        self.conn.commit()
    
    def remove_dealias_history(self, mjd_later_than=None, timestamp_later_than=None, show_warning=True):
        if show_warning:
            self.logger.warning(f"WARNING: Removing dealias_history entries with obs_mjd later than {mjd_later_than}")
            input("Press Enter to continue...")

        if timestamp_later_than is None:
            if mjd_later_than is None:
                raise ValueError("mjd_later_than must be provided if timestamp_later_than is provided")
            timestamp_later_than = utils.mjd_to_timestamp(mjd_later_than)

        dealias_history = self.get_all_dealias_history()
        for this_info in dealias_history:
            if this_info["timestamp"] > timestamp_later_than:
                self.logger.warning(f"Removing dealias_history entry with timestamp={this_info['timestamp']}, mjd={utils.timestamp_to_mjd(this_info['timestamp'])}")
                self.cur.execute("DELETE FROM dealias_history WHERE timestamp = ?", (this_info["timestamp"],))
        
        self.conn.commit()

    def get_all_info(self):
        timing_info = self.get_all_timing_info()

        for i, info in enumerate(timing_info):
            this_files = {}
            for file in info["files"]:
                this_files[file] = {}
                this_files[file]["toa"] = self.get_toa_by_filename(file)
                this_files[file]["archive_info"] = self.get_archive_info_by_filename(file)
            timing_info[i]["files"] = this_files

        return timing_info

    def insert_config(self, config, commit=True):
        for key, value in self.format_config(config, reverse=True):
            self.cur.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
        
        if commit:
            self.conn.commit()

    def get_config(self, key):
        self.cur.execute("SELECT value FROM config WHERE key = ?", (key,))
        return self.cur.fetchone()[0]

    def get_all_config(self):
        self.cur.execute("SELECT * FROM config")
        config_raw = self.cur.fetchall()

        config = {}
        for c in config_raw:
            config[c[0]] = c[1]

        return self.format_config(config.items())

    def format_config(self, config, reverse=False):
        if reverse:
            listed_config = []

            for key, value in config.items():
                for subkey, subvalue in value.items():
                    listed_config.append([key + ":" + subkey, json.dumps(subvalue)])

            return listed_config
        else:
            formatted_config = {}

            for key, value in config:
                parts = key.split(":")

                if len(parts) != 2:
                    raise ValueError("Invalid key: {}".format(key))

                if parts[0] not in formatted_config:
                    formatted_config[parts[0]] = {}
                formatted_config[parts[0]][parts[1]] = json.loads(value)

            return formatted_config

    def truncate_config(self, show_warning=True):
        if show_warning:
            self.logger.warning("WARNING: Emptying config table")
            input("Press Enter to continue...")
        self.cur.execute("DELETE FROM config")
        self.conn.commit()
    
    def get_version(self):
        self.cur.execute("SELECT version FROM info")
        return float(self.cur.fetchone()[0])

    def get_mjd_by_filename(self, filename):
        """
        Get the MJD of the TOA by filename
        If the TOA does not exist, return 0
        """

        return self.get_toa_by_filename(filename)["toa"]

    def is_blank_db(self):
        self.cur.execute("SELECT COUNT(*) FROM timing_info")
        count = self.cur.fetchone()[0]

        return (count == 0)

    def create_parfile(self):
        return self.get_all_timing_info()[-1]["notes"]["fitted_parfile"]

    def create_timfile(self, ar_list=None):
        timfile = ""

        if ar_list is None:
            ar_list = []
            for archive in self.get_all_archive_info():
                ar_list.append({"path": archive["filename"]})

        for ar_info in ar_list:
            this_toa = self.get_toa_by_filename(utils.get_archive_id(ar_info["path"]))

            if(not self.db_check_valid_toa(ar_info["path"])):
                self.logger.warning(f"INVALID_TOA remark was found for {ar_info['path']}. Skipped while creating timfile...", layer=1)
                continue

            if(this_toa["timestamp"] == 0 or this_toa["raw_tim"].strip() == ""):
                raise Exception(f"TOA from archive [{ar_info['path']}] does not exist in database. ")
            
            timfile += this_toa["raw_tim"] + f" -rcvr {this_toa['notes']['rcvr']} " + "\n"

        return timfile
    
    def db_check_valid_toa(self, archive):
        toa = self.get_toa_by_filename(utils.get_archive_id(archive))

        if "remark" in toa["notes"]:
            if toa["notes"]["remark"] == "INVALID_TOA":
                return False
            
        return True
    
    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

        if self.readonly:
            # remove temporary database
            self.logger.info(f"Removing temporary database {self.psr_db}")
            os.remove(self.psr_db)

    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
    
# test_db = "/Users/wenky/Downloads/timing.db"
# with database(test_db) as db:
#     db.insert_timing_info(["file1", "file2", "file3"], [1, 2, 3, 4], "JUMP", [1, 2, 3, 4], 1.0, 1.0, {}, {})
#     print(db.get_last_timing_info())
#     print(db.get_all_timing_info())
# with database(test_db) as db:
#     db.insert_config({"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}})
#     print(db.get_all_config())
# with database(test_db) as db:
#     filename = f"test.tim{utils.get_rand_string()}"
#     db.insert_toa(filename, 1234, 1234.1, 1234, "chime", "test", {})
#     print(db.get_last_toa())
#     print(db.get_all_toas())
#     print(db.get_toa_by_filename(filename))
#     print(db.check_toa_exists(filename))

#     db.insert_archive_info(filename, [1, 2, 3, 4], 1.0, {})
#     print(db.get_last_archive_info())
#     print(db.get_all_archive_info())
#     print(db.get_archive_info_by_filename(filename))