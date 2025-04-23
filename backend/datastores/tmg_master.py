import sqlite3
import time
import json
import shutil
import os

from ..utils.utils import utils
from ..utils.logger import logger

try:
    from ..io.archive import ArchiveReader
except ImportError:
    ArchiveReader = None

try:
    from ..io.filterbank import FilterbankReader
except ImportError:
    FilterbankReader = None

class tmg_master:
    """
    Database structure:
    ---
    Table: info
    Columns: version
    ---
    Table: raw_data
    Columns: id, psr_id, ar_id, location, mjd, md5sum, size, format, backend, status, metadata, notes
    psr_id + ar_id unique
    id index (autoincrement)
    psr_id index
    mjd index
    format index
    backend index
    status index
    ---
    Table: timing
    Columns: psr_id, timing_dir, last_updated, last_status, notes
    psr_id unique
    """
    def __init__(self, db_path, readonly=False, self_check=False, fast_mode=True, mem_gb=1, logger=logger()):
        self.version = "1.0"
        self.readonly = readonly
        self.db_path = None
        self.self_check = self_check
        self.logger = logger
        self.allowed_formats = ["archive", "filterbank"]
        self.allowed_backends = ["champss", "chimepsr_fm", "chimepsr_fil"]
        self.allowed_status = ["good", "corrupted"]
        self.fast_mode = fast_mode
        self.fast_mode_mem_gb = mem_gb

        if FilterbankReader == None or ArchiveReader == None:
            self.readonly = True
            self.logger.warning("PSRCHIVE or sigpyproc are not available. Readonly mode enabled.")

        if self.fast_mode:
            self.logger.info(f"Fast mode enabled with {self.fast_mode_mem_gb} GB memory.")

        if not os.path.exists(os.path.dirname(db_path)):
            raise Exception(f"Database folder {os.path.dirname(db_path)} does not exist. Please provide a valid path for TMGMaster DB.")
        
        if self.readonly:
            # check if database exists
            if not os.path.exists(db_path):
                raise Exception(f"Database {db_path} does not exist. Please provide a valid database file.")

            # copy the database to a temporary file
            self.db_path = os.path.abspath(f"{db_path}.readonly{utils.get_rand_string()}.tmp")
            shutil.copyfile(db_path, self.db_path)
            self.logger.debug(f"Readonly temporary database created at {self.db_path}")

            # open the temporary database in readonly mode
            self.conn = sqlite3.connect("file://" + self.db_path + "?mode=ro", uri=True, check_same_thread=False)
        else:
            self.db_path = db_path
            self.conn = sqlite3.connect(self.db_path)

        self.cur = self.conn.cursor()
    
    def initialize(self, self_check=True):
        if self.readonly:
            return
        
        if not os.path.exists(self.db_path):
            self.logger.info(f"Creating database {self.db_path}")
            
        # setup database
        if self.fast_mode:
            self.cur.execute("PRAGMA synchronous = OFF;") # disable synchronous mode
            self.cur.execute("PRAGMA journal_mode = MEMORY;") # use memory journal
            self.cur.execute("PRAGMA temp_store = MEMORY;") # use memory for temporary storage
            self.cur.execute("PRAGMA cache_size = 10000;") # set cache size to 10000 pages
            self.cur.execute(f"PRAGMA mmap_size = {self.fast_mode_mem_gb * 1000000000};") # set mmap size

        # create tables
        self.cur.execute("CREATE TABLE IF NOT EXISTS info (version TEXT)")
        # self.cur.execute("CREATE TABLE IF NOT EXISTS raw_data (psr_id TEXT, ar_id TEXT, location TEXT, mjd REAL, md5sum TEXT, size INTEGER, format TEXT, backend TEXT, metadata TEXT, notes TEXT, PRIMARY KEY (ar_id))")
        self.cur.execute("CREATE TABLE IF NOT EXISTS raw_data (id INTEGER PRIMARY KEY AUTOINCREMENT, psr_id TEXT, ar_id TEXT, location TEXT, mjd REAL, md5sum TEXT, size INTEGER, format TEXT, backend TEXT, status TEXT, metadata TEXT, notes TEXT)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS timing (psr_id TEXT, timing_dir TEXT, last_updated REAL, last_status TEXT, notes TEXT, PRIMARY KEY (psr_id))")

        # create indices
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_psr_id ON raw_data (psr_id)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_mjd ON raw_data (mjd)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_format ON raw_data (format)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_backend ON raw_data (backend)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_status ON raw_data (status)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_timing_psr_id ON timing (psr_id)")

        # create unique constraints
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_data_psr_id_ar_id ON raw_data (psr_id, ar_id)")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timing_psr_id ON timing (psr_id)")

        #  insert version
        self.cur.execute("SELECT version FROM info")
        version = self.cur.fetchone()
        if version is None:
            self.cur.execute("INSERT INTO info (version) VALUES (?)", (self.version,))
        elif version[0] != self.version:
            self.logger.warning(f"Database version mismatch. Expected: {self.version}, Found: {version[0]}")

        # self check
        if self_check:
            self.self_check()

        self.conn.commit()
    
    def self_check(self):
        # Check if all files in raw_data exist
        for data in self.get_raw_data():
            if not os.path.exists(data["location"]):
                self.logger.warning(f"File {data['location']} does not exist in the database.")

        # Check if all timing directories exist
        for timing in self.get_timing():
            if not os.path.exists(timing["timing_dir"]):
                self.logger.warning(f"Timing directory {timing['timing_dir']} does not exist in the database.")

    def close(self):
        if self.readonly:
            self.conn.close()
            os.remove(self.db_path)
            self.logger.debug(f"Readonly temporary database removed from {self.db_path}")
        else:
            self.conn.commit()
            self.conn.close()

    def get_version(self):
        self.cur.execute("SELECT version FROM info")
        return self.cur.fetchone()[0]
    
    def insert_raw_data(self, psr_id, ar_id, location, mjd, md5sum, size, format, backend, status, metadata, notes, skip_if_exists=False):
        if self.readonly:
            raise Exception("Cannot insert data into readonly database.")

        if format not in self.allowed_formats:
            raise Exception(f"Unrecognized format {format}. Allowed formats: {self.allowed_formats}")

        if backend not in self.allowed_backends:
            raise Exception(f"Unrecognized backend {backend}. Allowed backends: {self.allowed_backends}")

        if status not in self.allowed_status:
            raise Exception(f"Unrecognized status {status}. Allowed statuses: {self.allowed_status}")
        
        if skip_if_exists:
            self.cur.execute("SELECT COUNT(*) FROM raw_data WHERE psr_id = ? AND ar_id = ?", (psr_id, ar_id,))
            if self.cur.fetchone()[0] > 0:
                return
        
        self.cur.execute("INSERT INTO raw_data (psr_id, ar_id, location, mjd, md5sum, size, format, backend, status, metadata, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (psr_id, ar_id, location, mjd, md5sum, size, format, backend, status, json.dumps(metadata), json.dumps(notes)))
        self.conn.commit()

    def insert_raw_data_from_file(self, psr_id, location, backend, format="auto", skip_if_exists=False, placeholder_if_corrupted=False):
        if self.readonly:
            raise Exception("Cannot insert data into readonly database.")

        # Check if file is corrupted
        if os.path.getsize(location) == 0:
            if not placeholder_if_corrupted:
                raise Exception(f"File {location} is empty. This may be due to corruption.")
            else:
                self.logger("File is corrupted. Inserting placeholder.", location)

                # guess the format from the extension
                if format == "auto":
                    if location.endswith(".ar"):
                        format = "archive"
                    elif location.endswith(".fil"):
                        format = "filterbank"
                    else:
                        raise Exception(f"Cannot guess format from extension. Please provide format explicitly.")

                # insert placeholder
                return self.insert_raw_data(
                    psr_id = psr_id, 
                    ar_id = utils.get_archive_id(location), 
                    location = location,
                    mjd = 0,
                    md5sum = utils.get_md5sum(location),
                    size = os.path.getsize(location),
                    format = format, 
                    backend = backend,
                    status = "corrupted",
                    metadata = {},
                    notes = {}, 
                    skip_if_exists=skip_if_exists
                )

        if format == "auto":
            format = utils.get_raw_data_format(location)

        if format == "archive":
            au = ArchiveReader(location)
            return self.insert_raw_data(
                psr_id = psr_id, 
                ar_id = utils.get_archive_id(location), 
                location = location,
                mjd = au.get_mjd(),
                md5sum = utils.get_md5sum(location),
                size = os.path.getsize(location),
                format = "archive", 
                backend = backend,
                status = "good",
                metadata = au.get_metadata(),
                notes = {}, 
                skip_if_exists=skip_if_exists
            )
        elif format == "filterbank":
            fu = FilterbankReader(location)
            return self.insert_raw_data(
                psr_id = psr_id, 
                ar_id = utils.get_archive_id(location),
                location = location, 
                mjd = fu.get_mjd(),
                md5sum = utils.get_md5sum(location),
                size = os.path.getsize(location),
                format = "filterbank",
                backend = backend,
                status = "good",
                metadata = fu.get_metadata(),
                notes = {}, 
                skip_if_exists=skip_if_exists
            )
        else:
            raise Exception(f"Unrecognized format")


    def update_raw_data(self, psr_id, ar_id, location=None, mjd=None, md5sum=None, size=None, format=None, backend=None, status=None, metadata=None, notes=None, create_if_not_exists=False, force_update=False):
        if self.readonly:
            raise Exception("Cannot update data in readonly database.")
        
        if create_if_not_exists:
            self.cur.execute("SELECT COUNT(*) FROM raw_data WHERE psr_id = ? AND ar_id = ?", (psr_id, ar_id,))
            if self.cur.fetchone()[0] == 0:
                if psr_id == None or ar_id == None or location == None or mjd == None or md5sum == None or size == None or format == None or backend == None or status == None or metadata == None or notes == None:
                    raise Exception("Cannot create raw data entry without all required fields.")
                self.insert_raw_data(psr_id, ar_id, location, mjd, md5sum, size, format, backend, status, metadata, notes)
                return
        
        update_query = "UPDATE raw_data SET "
        update_values = []
        if location is not None:
            update_query += "location = ?, "
            update_values.append(location)
        if mjd is not None:
            if not force_update:
                raise Exception("It is not make sense to update MJD. If this is necessary, please set force_update=True to bypass this check.")
            update_query += "mjd = ?, "
            update_values.append(mjd)
        if md5sum is not None:
            update_query += "md5sum = ?, "
            update_values.append(md5sum)
        if size is not None:
            update_query += "size = ?, "
            update_values.append(size)
        if format is not None:
            if not force_update:
                raise Exception("It is not make sense to update FORMAT. If this is necessary, please set force_update=True to bypass this check.")
            if format not in self.allowed_formats:
                raise Exception(f"Unrecognized format {format}. Allowed formats: {self.allowed_formats}")
            update_query += "format = ?, "
            update_values.append(format)
        if backend is not None:
            if not force_update:
                raise Exception("It is not make sense to update BACKEND. If this is necessary, please set force_update=True to bypass this check.")
            if backend not in self.allowed_backends:
                raise Exception(f"Unrecognized backend {backend}. Allowed backends: {self.allowed_backends}")
            update_query += "backend = ?, "
            update_values.append(backend)
        if status is not None:
            if status not in self.allowed_status:
                raise Exception(f"Unrecognized status {status}. Allowed statuses: {self.allowed_status}")
            update_query += "status = ?, "
            update_values.append(status)
        if metadata is not None:
            if not force_update:
                raise Exception("It is not make sense to update METADATA. If this is necessary, please set force_update=True to bypass this check.")
            update_query += "metadata = ?, "
            update_values.append(json.dumps(metadata))
        if notes is not None:
            update_query += "notes = ?, "
            update_values.append(json.dumps(notes))
        
        if len(update_values) == 0:
            return
        
        update_query = update_query[:-2] + " WHERE psr_id = ? AND ar_id = ?"
        update_values.append(psr_id)
        update_values.append(ar_id)

        self.cur.execute(update_query, update_values)
        self.conn.commit()
    
    def format_raw_data(self, raw_data_res):
        if raw_data_res is None:
            raw_data_res = [-1, "", "", "", 0.0, "", 0, "", "", "", "{}", "{}"]
        
        formatted_raw_data = {
            # "id": raw_data_res[0],
            "psr_id": raw_data_res[1],
            "ar_id": raw_data_res[2],
            "location": raw_data_res[3],
            "mjd": raw_data_res[4],
            "md5sum": raw_data_res[5],
            "size": raw_data_res[6],
            "format": raw_data_res[7],
            "backend": raw_data_res[8],
            "status": raw_data_res[9],
            "metadata": json.loads(raw_data_res[10]),
            "notes": json.loads(raw_data_res[11])
        }

        return formatted_raw_data

    def format_raw_data_all(self, raw_data_res):
        formatted_raw_data = []
        for raw_data in raw_data_res:
            formatted_raw_data.append(self.format_raw_data(raw_data))
        
        return formatted_raw_data
    
    def get_raw_data(self, psr_id=None, ar_id=None, location=None, mjd=None, md5sum=None, size=None, format=None, backend=None, status=None, metadata=None, notes=None, remove_action=False):
        query = "SELECT * FROM raw_data WHERE "
        
        # Remove action
        if remove_action:
            query = "DELETE FROM raw_data WHERE "

        query_values = []
        if psr_id is not None:
            query += "psr_id = ? AND "
            query_values.append(psr_id)
        if ar_id is not None:
            query += "ar_id = ? AND "
            query_values.append(ar_id)
        if location is not None:
            query += "location = ? AND "
            query_values.append(location)
        if mjd is not None:
            query += "mjd = ? AND "
            query_values.append(mjd)
        if md5sum is not None:
            query += "md5sum = ? AND "
            query_values.append(md5sum)
        if size is not None:
            query += "size = ? AND "
            query_values.append(size)
        if format is not None:
            query += "format = ? AND "
            query_values.append(format)
        if backend is not None:
            query += "backend = ? AND "
            query_values.append(backend)
        if status is not None:
            query += "status = ? AND "
            query_values.append(status)
        if metadata is not None:
            query += "metadata = ? AND "
            query_values.append(json.dumps(metadata))
        if notes is not None:
            query += "notes = ? AND "
            query_values.append(notes)

        if len(query_values) == 0:
            query = query[:-7]
        else:
            query = query[:-5]

        # Execute
        self.cur.execute(query, query_values)

        if remove_action:
            self.conn.commit()
            return

        return self.format_raw_data_all(self.cur.fetchall())
    
    def count_raw_data(self, psr_id=None, ar_id=None, location=None, mjd=None, md5sum=None, size=None, format=None, backend=None, status=None, metadata=None, notes=None):
        query = "SELECT COUNT(*) FROM raw_data WHERE "
        query_values = []
        if psr_id is not None:
            query += "psr_id = ? AND "
            query_values.append(psr_id)
        if ar_id is not None:
            query += "ar_id = ? AND "
            query_values.append(ar_id)
        if location is not None:
            query += "location = ? AND "
            query_values.append(location)
        if mjd is not None:
            query += "mjd = ? AND "
            query_values.append(mjd)
        if md5sum is not None:
            query += "md5sum = ? AND "
            query_values.append(md5sum)
        if size is not None:
            query += "size = ? AND "
            query_values.append(size)
        if format is not None:
            query += "format = ? AND "
            query_values.append(format)
        if backend is not None:
            query += "backend = ? AND "
            query_values.append(backend)
        if status is not None:
            query += "status = ? AND "
            query_values.append(status)
        if metadata is not None:
            query += "metadata = ? AND "
            query_values.append(json.dumps(metadata))
        if notes is not None:
            query += "notes = ? AND "
            query_values.append(notes)

        if len(query_values) == 0:
            query = query[:-7]
        else:
            query = query[:-5]

        self.cur.execute(query, query_values)
        return self.cur.fetchone()[0]
    
    def get_psr_ids(self, table="raw_data"):
        if table not in ["raw_data", "timing"]:
            raise Exception("Unknown table", table)

        self.cur.execute("SELECT DISTINCT psr_id FROM " + table)
        
        return [res[0] for res in self.cur.fetchall()]
    
    def get_ar_ids(self, psr_id):
        self.cur.execute("SELECT ar_id FROM raw_data WHERE psr_id = ?", (psr_id,))
        return [res[0] for res in self.cur.fetchall()]

    def get_ar_ids_idxed_by_psr_id(self):
        ars = {}
        for psr_id in self.get_psr_ids():
            ars[psr_id] = self.get_ar_ids(psr_id)

        return ars
    
    def get_raw_data_by_mjd_range(self, psr_id, mjd_range):
        self.cur.execute("SELECT * FROM raw_data WHERE psr_id = ? AND mjd >= ? AND mjd <= ?", (psr_id, mjd_range[0], mjd_range[1]))
        return self.format_raw_data_all(self.cur.fetchall())
    
    def count_raw_data_by_mjd_range(self, psr_id, mjd_range):
        self.cur.execute("SELECT COUNT(*) FROM raw_data WHERE psr_id = ? AND mjd >= ? AND mjd <= ?", (psr_id, mjd_range[0], mjd_range[1]))
        return self.cur.fetchone()[0]

    def get_timing_data_config(self, psr, mjd_range=None):
        if mjd_range is None:
            self.cur.execute("SELECT * FROM raw_data WHERE psr_id = ? AND status != 'corrupted'", (psr,))
        else:
            self.cur.execute("SELECT * FROM raw_data WHERE psr_id = ? AND status != 'corrupted' AND mjd >= ? AND mjd <= ?", (psr, mjd_range[0], mjd_range[1]))
        
        psr_config = {}
        counts = {"pulsar": 0, "psrfil": 0, "champss": 0, "total": 0}
        db_data = self.format_raw_data_all(self.cur.fetchall())

        def get_mjd_idx(d):
            return round(d["mjd"], 1)

        for d in db_data:
            psr_config[get_mjd_idx(d)] = []
        for d in db_data:
            if d["backend"] == "chimepsr_fm":
                psr_config[get_mjd_idx(d)].append({
                    "path": d["location"],
                    "label": "CHIME/Pulsar Fold Mode", 
                    "rcvr": "pulsar",
                    "mjd": d["mjd"], 
                    "arch_dm": 0 # no longer need this information. 
                })
                counts["pulsar"] += 1
                counts["total"] += 1
        for d in db_data:
            if d["backend"] == "chimepsr_fil":
                psr_config[get_mjd_idx(d)].append({
                    "path": d["location"],
                    "label": "CHIME/Pulsar Search Mode", 
                    "rcvr": "psrfil",
                    "mjd": d["mjd"], 
                    "arch_dm": 0 # no longer need this information. 
                })
                counts["psrfil"] += 1
                counts["total"] += 1
        for d in db_data:
            if d["backend"] == "champss":
                psr_config[get_mjd_idx(d)].append({
                    "path": d["location"],
                    "label": "CHAMPSS Fold Mode", 
                    "rcvr": "champss",
                    "mjd": d["mjd"], 
                    "arch_dm": 0 # no longer need this information. 
                })
                counts["champss"] += 1
                counts["total"] += 1

        # sort by mjd
        psr_config = dict(sorted(psr_config.items()))

        return psr_config, counts
    
    def insert_timing(self, psr_id, timing_dir, last_updated, last_status, notes, skip_if_exists=False):
        if self.readonly:
            raise Exception("Cannot insert data into readonly database.")
        
        if skip_if_exists:
            self.cur.execute("SELECT COUNT(*) FROM timing WHERE psr_id = ?", (psr_id,))
            if self.cur.fetchone()[0] > 0:
                return
        
        self.cur.execute("INSERT INTO timing (psr_id, timing_dir, last_updated, last_status, notes) VALUES (?, ?, ?, ?, ?)", (psr_id, timing_dir, last_updated, last_status, json.dumps(notes)))
        self.conn.commit()

    def update_timing(self, psr_id, timing_dir=None, last_updated=None, last_status=None, notes=None, create_if_not_exists=False):
        if self.readonly:
            raise Exception("Cannot update data in readonly database.")
        
        if create_if_not_exists:
            self.cur.execute("SELECT COUNT(*) FROM timing WHERE psr_id = ?", (psr_id,))
            if self.cur.fetchone()[0] == 0:
                if psr_id == None or timing_dir == None or last_updated == None or last_status == None or notes == None:
                    raise Exception("Cannot create timing entry without all required fields.")
                self.insert_timing(psr_id, timing_dir, last_updated, last_status, notes)
                return
        
        update_query = "UPDATE timing SET "
        update_values = []
        if timing_dir is not None:
            update_query += "timing_dir = ?, "
            update_values.append(timing_dir)
        if last_updated is not None:
            update_query += "last_updated = ?, "
            update_values.append(last_updated)
        if last_status is not None:
            update_query += "last_status = ?, "
            update_values.append(last_status)
        if notes is not None:
            update_query += "notes = ?, "
            update_values.append(json.dumps(notes))

        if len(update_values) == 0:
            return
        
        update_query = update_query[:-2] + " WHERE psr_id = ?"
        update_values.append(psr_id)

        self.cur.execute(update_query, update_values)
        self.conn.commit()

    def format_timing(self, timing_res):
        if timing_res is None:
            timing_res = ["", "", 0.0, "", "{}"]
        
        formatted_timing = {
            "psr_id": timing_res[0],
            "timing_dir": timing_res[1],
            "last_updated": timing_res[2],
            "last_status": timing_res[3],
            "notes": json.loads(timing_res[4])
        }

        return formatted_timing
    
    def format_timing_all(self, timing_res):
        formatted_timing = []
        for timing in timing_res:
            formatted_timing.append(self.format_timing(timing))

        return formatted_timing
    
    def get_timing(self, psr_id=None, timing_dir=None, last_updated=None, last_status=None, notes=None):
        query = "SELECT * FROM timing WHERE "
        query_values = []
        if psr_id is not None:
            query += "psr_id = ? AND "
            query_values.append(psr_id)
        if timing_dir is not None:
            query += "timing_dir = ? AND "
            query_values.append(timing_dir)
        if last_updated is not None:
            query += "last_updated = ? AND "
            query_values.append(last_updated)
        if last_status is not None:
            query += "last_status = ? AND "
            query_values.append(last_status)
        if notes is not None:
            query += "notes = ? AND "
            query_values.append(notes)

        if len(query_values) == 0:
            query = query[:-7]
        else:
            query = query[:-5]

        self.cur.execute(query, query_values)
        return self.format_timing_all(self.cur.fetchall())
    
    def get_timing_by_mjd_range(self, mjd_range):
        self.cur.execute("SELECT * FROM raw_data WHERE mjd >= ? AND mjd <= ?", mjd_range)
        return self.format_raw_data_all(self.cur.fetchall())
    
    def count_timing_by_mjd_range(self, mjd_range):
        self.cur.execute("SELECT COUNT(*) FROM raw_data WHERE mjd >= ? AND mjd <= ?", mjd_range)
        return self.cur.fetchone()[0]

    def __enter__(self):
        self.initialize(self_check=self.self_check)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# Tests
# db_path = "/Users/wenky/Downloads/TMGMasrer.db"
# if os.path.exists(db_path):
#     os.remove(db_path)
# with tmg_master(db_path) as db:
#     db.insert_raw_data("J1234+5678", "AR12345", "/path/to/file", 58765.1234, "md5sum", 123456, "archive", "champss", "good", {"key": "value"}, {"key": "value"})
#     print(db.get_raw_data())
#     print(db.get_raw_data(psr_id="J1234+5678"))
#     print(db.get_raw_data(ar_id="AR12345"))
#     print(db.get_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     print(db.count_raw_data())
#     print(db.count_raw_data(psr_id="J1234+5678"))
#     print(db.count_raw_data(ar_id="AR12345"))
#     print(db.count_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     db.update_raw_data("J1234+5678", "AR12345", location="/path/to/file2_modified", mjd=58765.1235, md5sum="md5sum2", size=123457, format="filterbank", backend="chimepsr_fm", status="corrupted", metadata={"key": "value2"}, notes={"key": "value2"}, force_update=True)
#     print(db.get_raw_data())
#     print(db.get_raw_data(psr_id="J1234+5678"))
#     print(db.get_raw_data(ar_id="AR12345"))
#     print(db.get_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     print(db.count_raw_data())
#     print(db.count_raw_data(psr_id="J1234+5678"))
#     print(db.count_raw_data(ar_id="AR12345"))
#     print(db.count_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     db.insert_raw_data("J1234+5678", "AR12346", "/path/to/file", 58765.1234, "md5sum", 123456, "archive", "champss", "good", {"key": "value"}, {"key": "value"})
#     print(db.get_raw_data())
#     print(db.get_raw_data(psr_id="J1234+5678"))
#     print(db.get_raw_data(ar_id="AR12345"))
#     print(db.get_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     print(db.count_raw_data())
#     print(db.count_raw_data(psr_id="J1234+5678"))
#     print(db.count_raw_data(ar_id="AR12345"))
#     print(db.count_raw_data(psr_id="J1234+5678", ar_id="AR12345"))
#     db.insert_timing("J1234+5678", "/path/to/timing", time.time(), "status", {"key": "value"})
#     print(db.get_timing())
#     print(db.get_timing(psr_id="J1234+5678"))
#     print(db.get_timing(timing_dir="/path/to/timing"))
#     print(db.get_timing(psr_id="J1234+5678", timing_dir="/path/to/timing"))
#     db.update_timing("J1234+5678", timing_dir="/path/to/timing2", last_updated=time.time(), last_status="status2", notes={"key": "value2"})
#     print(db.get_timing())
#     print(db.get_timing(psr_id="J1234+5678"))
#     print(db.get_timing(timing_dir="/path/to/timing"))
#     print(db.get_timing(psr_id="J1234+5678", timing_dir="/path/to/timing"))
#     db.insert_timing("J1234+5678A", "/path/to/timing", time.time(), "status", {"key": "value"})
#     print(db.get_timing())
#     print(db.get_timing(psr_id="J1234+5678A"))
#     print(db.get_timing(timing_dir="/path/to/timing"))
#     print(db.get_timing(psr_id="J1234+5678A", timing_dir="/path/to/timing"))
#     print(db.get_version())
#     print(db.get_psr_ids())
#     print(db.get_raw_data_by_mjd_range("J1234+5678", (58765.1234, 58765.1235)))
#     print(db.count_raw_data_by_mjd_range("J1234+5678", (58765.1234, 58765.1235)))
#     print(db.get_timing_by_mjd_range((58765.1234, 58765.1235)))
#     print(db.count_timing_by_mjd_range((58765.1234, 58765.1235)))
#     print(db.get_ar_ids_idxed_by_psr_id())