import os
import shutil
import hashlib
import tqdm
import json
import numpy as np
from multiprocessing import Pool

from .database import database
from ..utils.exec import exec
from ..utils.utils import utils
from ..utils.data_quality import MatchedFilterSNR
from ..io.archive import ArchiveReader
from ..processing.archive_shutils import archive_shutils

# Putting function outside of the class since db_hdl cannot be pickled and passed to Pool
def _archive_cache__db_update_psr_amps_many__get_amp_and_snr(filename, prof_templ):
    # Get profile amps
    amps = ArchiveReader(filename).get_amps()

    # Calculate matched filter SNR
    if prof_templ is None:
        snr = 0
    else:
        snr = MatchedFilterSNR(amps, prof_templ).compute()

    return amps, snr

def _archive_cache__shutils_update_model(archive, parfile, jump):
        return archive_shutils(archive).install_parfile(parfile, jump)

class archive_cache:
    def __init__(self, psr_dir, db_hdl=None, db_path=None):
        self.db_hdl = db_hdl
        self.db_path = db_path
        self.psr_dir = psr_dir
        self.cache_dir = f"{psr_dir}/__champss_archive_cache__"
        self.utils = utils

    def initialize(self):
        # check database connection
        if self.db_hdl is None and self.db_path is None:
            raise Exception("Either db_hdl or db_path must be provided.")
        
        # create database connection
        if self.db_path is not None:
            self.db_hdl = database(self.db_path)
            self.db_hdl.initialize()
            
        # create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # check archive cache integrity
        archive_info = self.db_hdl.get_all_archive_info()
        for ar in archive_info:
            if not os.path.exists(f"{self.cache_dir}/{ar['filename']}"):
                self.utils.print_warning(f"Archive {ar['filename']} not found in cache. Please resolve this issue manually. Maybe the cache was deleted and needs to be created manually.")
        
    def add_archive(self, filename, rcvr="unknown"):
        if not os.path.exists(filename):
            raise Exception(f"Archive {filename} does not exist.")

        # copy archive to cache
        print(f"  [Archive] {filename} -> archive cache")
        shutil.copyfile(filename, f"{self.cache_dir}/{self.utils.get_archive_id(filename)}")

        # insert archive info to database
        print(f"  [Archive] {filename} -> database")
        self.db_insert_archive_info(filename, rcvr)

    def get_md5(self, filename):
        md5 = hashlib.md5()

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        return md5.hexdigest()

    def archive_exists(self, filename):
        return os.path.exists(f"{self.cache_dir}/{self.utils.get_archive_id(filename)}")
    
    def archives_exists(self, filenames):
        for f in filenames:
            if not self.archive_exists(f):
                return False
        return True
    
    def get_archive(self, filename, dest):
        if not self.archive_exists(filename):
            raise Exception(f"Archive {filename} not found in cache.")
        
        shutil.copyfile(f"{self.cache_dir}/{self.utils.get_archive_id(filename)}", dest)

    def update_model(self, jumps, parfile="auto", n_pools="auto", tempdir="auto", cleanup=True):
        # Initialze variables
        if parfile == "auto":
            parfile = f"{self.psr_dir}/pulsar.par"

        if tempdir == "auto":
            tempdir = f"{self.cache_dir}/temp"
        else:
            tempdir = f"{tempdir}/" + self.utils.get_rand_string()

        if not os.path.exists(tempdir):
            os.makedirs(tempdir, exist_ok=True)

        # get all timed files
        timed_files = self.db_hdl.get_last_timing_info()["files"] # only get latest timing info files

        # get all archives
        archives = []
        archive_shutils_objects = []
        for ar in tqdm.tqdm(self.db_hdl.get_all_archive_info(), desc="Preparing archives"):
            if ar["filename"] not in timed_files:
                self.utils.print_warning(f"Archive {ar['filename']} not in timing_info. Skipping.")
                continue

            this_path = f"{self.cache_dir}/{ar['filename']}"
            this_temp_path = f"{tempdir}/{ar['filename']}"
            if os.path.exists(f"{this_path}"):
                # copy archive to temp directory
                shutil.copyfile(this_path, this_temp_path)

                # get jump value
                if ar["notes"]["rcvr"] in jumps:
                    jump = jumps[ar["notes"]["rcvr"]][0]
                else:
                    raise Exception(f"Receiver {ar['notes']['rcvr']} not found in jumps. Please add it to jumps in configurations.")

                # append to archives
                archives.append({
                    "filename": this_path,
                    "temp_filename": this_temp_path,
                    "rcvr": ar["notes"]["rcvr"],
                    "jump": jump
                })
            else:
                self.utils.print_warning(f"Archive {ar['filename']} not found in cache. Skipping.")

        # update model and apply jump
        with Pool(processes=n_pools) as pool:
            print(f"  [update_model] Using {n_pools} processes... ")
            tqdm.tqdm(
                pool.starmap(
                    _archive_cache__shutils_update_model, 
                    [
                        (this_ar["temp_filename"], parfile, this_ar["jump"]) for this_ar in archives
                    ]
                ), 
                total=len(archives), 
                desc="Updating model in archives"
            )

        # update psr_amps in database
        print(f"  [update_model] updating archive information in database... ")
        self.db_update_psr_amps_many(
            [this_ar["temp_filename"] for this_ar in archives],
            n_pools=n_pools, 
            commit=True
        )
        utils.print_success(f"  [update_model] archive information in database updated for {len(archives)} observations. ")

        # cleanup
        if cleanup:
            shutil.rmtree(tempdir)

        return True

    def db_insert_archive_info(self, filename, rcvr):
        archive_hdl = ArchiveReader(filename)

        self.db_hdl.insert_archive_info(
            filename = self.utils.get_archive_id(filename), 
            psr_amps = archive_hdl.get_amps(), 
            psr_snr = archive_hdl.get_snr(), 
            notes = {
                "md5": self.get_md5(filename), 
                "rcvr": rcvr
            }
        )
    
    def db_update_psr_amps(self, filename, commit=True):
        archive_hdl = ArchiveReader(filename)

        last_archive_info = self.db_hdl.get_archive_info_by_filename(self.utils.get_archive_id(filename))
        notes = last_archive_info["notes"]
        notes["md5"] = self.get_md5(filename)

        self.db_hdl.update_archive_info(
            filename = self.utils.get_archive_id(filename),
            psr_amps = archive_hdl.get_amps(),
            psr_snr = archive_hdl.get_snr(), 
            notes = notes, 
            commit = commit
        )

    def db_update_psr_amps_many(self, filenames, n_pools=4, commit=True):
        # get profile template from database
        prof_templ = self.db_hdl.get_config("__template:amps")
        if prof_templ is not None: 
            prof_templ = json.loads(prof_templ)

        with Pool(processes=n_pools) as pool:
            # results = list(tqdm.tqdm(pool.imap(_archive_cache__db_update_psr_amps_many__get_amp_and_snr, filenames), total=len(filenames)))
            results = list(
                pool.starmap(
                    _archive_cache__db_update_psr_amps_many__get_amp_and_snr, 
                    tqdm.tqdm(
                        [(f, prof_templ) for f in filenames], 
                        total=len(filenames)
                    )
                )
            )
        
        ar_ids = []
        amps = []
        snrs = []
        for i, filename in enumerate(filenames):
            ar_ids.append(self.utils.get_archive_id(filename))
            amps.append(results[i][0])
            snrs.append(results[i][1])

        self.db_hdl.update_archive_amps_info_many(
            filenames = ar_ids,
            amps = amps,
            snrs = snrs,
            commit = commit
        )

    def db_commit(self):
        self.db_hdl.conn.commit()
        
    def cleanup(self):
        if self.db_path is not None: # close if database connection was created here
            self.db_hdl.close()