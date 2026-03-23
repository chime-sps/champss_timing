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
from ..utils.logger import logger
from ..utils.data_quality import MatchedFilterSNR
from ..io.archive import ArchiveReader
from ..processing.archive_shutils import archive_shutils
from ..tools.ephm_install import EphmInstall

# # Putting function outside of the class since db_hdl cannot be pickled and passed to Pool
# def _archive_cache__db_update_psr_amps_many__get_amp_and_snr(filename, prof_templ):
#     # Get profile amps
#     amps = ArchiveReader(filename).get_amps()

#     # Calculate matched filter SNR
#     if prof_templ is None:
#         snr = 0
#     else:
#         snr = MatchedFilterSNR(amps, prof_templ).compute()

#     return amps, snr

# def _archive_cache__shutils_update_model(archive, parfile, jump):
#         return archive_shutils(archive).install_parfile(parfile, jump)

def _archive_cache__pint_update_model(ar, parfile, jumps):
    # Initialize EphmInstall object
    ei = EphmInstall(
        amps=ar["notes"]["init_amps"], 
        freq=ar["notes"]["freq"], 
        epoch=ar["notes"]["init_epoch"], 
        site=ar["notes"]["site"]
    )

    # Install parfile
    ei.install_parfile(parfile=parfile)

    # Apply jump
    if ar["notes"]["rcvr"] in jumps:
        ei.jump_by_time_given_parfile(parfile, jumps[ar["notes"]["rcvr"]][0])

    return ar['filename'], ei.get_model_installed_amps().tolist()

class archive_cache:
    def __init__(self, psr_dir, db_hdl=None, db_path=None, logger=logger()):
        self.db_hdl = db_hdl
        self.db_path = db_path
        self.psr_dir = psr_dir
        self.cache_dir = f"{psr_dir}/__champss_archive_cache__"
        self.utils = utils
        self.logger = logger

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
                self.logger.warning(f"Archive {ar['filename']} not found in cache. Please resolve this issue manually. Maybe the cache was deleted and needs to be created manually.")
        
    def add_archive(self, filename, rcvr="unknown"):
        if not os.path.exists(filename):
            raise Exception(f"Archive {filename} does not exist.")
        self.logger.debug(f"Adding archive {filename} to cache...")

        # copy archive to cache
        self.logger.debug(f"{filename} -> archive cache", layer=1)
        shutil.copyfile(filename, f"{self.cache_dir}/{self.utils.get_archive_id(filename)}")

        # insert archive info to database
        self.logger.debug(f"{filename} -> database", layer=1)
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

    def update_model_internal(self, jumps, parfile="auto", n_pools="auto"):
        self.logger.debug(f"Updating model in all cached archives using internal PINT method... ", layer=1)
        
        # Initialze variables
        if parfile == "auto":
            parfile = f"{self.psr_dir}/pulsar.par"

        # Determine number of pools
        if n_pools == "auto":
            n_pools = os.cpu_count()

        # Query all timed files
        timed_files = self.db_hdl.get_last_timing_info()["files"] # only get latest timing info files

        # Query all archives
        cached_archives = self.db_hdl.get_all_archive_info()

        # Make sure all timed files are in cache
        cached_archive_ids = [ar["filename"] for ar in cached_archives]
        for f in timed_files:
            if f not in cached_archive_ids:
                raise Exception(f"Timed file {f} not found in cache. Please add it to cache first.")

        # Make sure all cached archives contain freq, site, init_epoch, init_amps info
        # (both information are introduced later as of Oct. 2025)
        for ar in cached_archives:
            if "freq" not in ar["notes"] or "site" not in ar["notes"] or "init_epoch" not in ar["notes"] or "init_amps" not in ar["notes"]:
                # add freq and epoch info
                archive_hdl = ArchiveReader(f"{self.cache_dir}/{ar['filename']}")
                ar["notes"]["freq"] = archive_hdl.get_freq()
                ar["notes"]["site"] = archive_hdl.get_telescope()
                ar["notes"]["init_epoch"] = archive_hdl.get_epoch()
                ar["notes"]["init_amps"] = archive_hdl.get_amps()
                self.db_hdl.update_archive_info(
                    filename = ar["filename"],
                    notes = ar["notes"],
                    commit = True
                )
                self.logger.success(f"[update_model_internal] Added freq and epoch info to archive {ar['filename']} in database.", layer=1)

            # # recalculate snr
            # # uncomment the following lines if you want to recalculate SNR for all archives for any reason
            # snr = MatchedFilterSNR(ar["notes"]["init_amps"], json.loads(self.db_hdl.get_config("__template:amps"))).compute()
            # self.logger.info(f"[update_model_internal] Recalculated SNR for archive {ar['filename']}: {snr}", layer=1)
            # self.db_hdl.update_archive_info(
            #     filename = ar["filename"],
            #     psr_snr = snr,
            #     commit = True
            # )

        # # Update model in all timed files
        # filenames = []
        # post_install_amps = []
        # for ar in tqdm.tqdm(cached_archives, desc="Installing new ephmeris"):
        #     # if ar["filename"] not in timed_files:
        #     #     continue
        #     # NOW WE ARE ABLE TO UPDATE ALL ARCHIVES SINCE INTERNAL METHOD RUNS MUCH FASTER!! :)

        #     # Initialize EphmInstall object
        #     ei = EphmInstall(
        #         amps=ar["notes"]["init_amps"], 
        #         freq=ar["notes"]["freq"], 
        #         epoch=ar["notes"]["init_epoch"], 
        #         site=ar["notes"]["site"]
        #     )

        #     # Install parfile
        #     ei.install_parfile(parfile=parfile)

        #     # Apply jump
        #     if ar["notes"]["rcvr"] in jumps:
        #         ei.jump_by_time_given_parfile(parfile, jumps[ar["notes"]["rcvr"]][0])

        #     # Append data
        #     filenames.append(ar['filename'])
        #     post_install_amps.append(ei.get_model_installed_amps().tolist())

        # Update model in all timed files using multiprocessing
        filenames = []
        post_install_amps = []
        with Pool(processes=n_pools) as pool:
            self.logger.info(f"Using {n_pools} processes... ", layer=1)
            for res in list(
                tqdm.tqdm(
                    pool.starmap(
                        _archive_cache__pint_update_model, 
                        [
                            (ar, parfile, jumps) for ar in cached_archives
                        ]
                    ), 
                    total=len(cached_archives),
                    desc="Installing new ephmeris"
                )
            ): 
                filenames.append(res[0])
                post_install_amps.append(res[1])

        # Commit changes to database
        self.logger.debug(f"Committing changes to database... ", layer=1)
        self.db_hdl.update_archive_amps_info_many(
            filenames = filenames,
            amps = post_install_amps, 
            commit = True
        )

        self.logger.success(f"Archive information in database updated for {len(filenames)} observations. ", layer=1)
        return True

    def update_model(self, jumps, parfile="auto", n_pools="auto", tempdir="auto", cleanup=True):
        '''
        Legacy method replaced by update_model_internal. 
        '''

        self.update_model_internal(jumps, parfile=parfile, n_pools=n_pools)

    def filter_archives_by_quality_checks(self, ar_list, checks=[]):
        if len(checks) == 0:
            return {"good": ar_list, "bad": []}

        res = {"good": [], "bad": []}
        for ar in ar_list:
            ar_info = self.db_hdl.get_archive_info_by_filename(self.utils.get_archive_id(ar["path"]))

            passed_checks = False
            for check in checks:
                if check["name"] == "snr":
                    if ar_info["psr_snr"] > check["threshold"]:
                        passed_checks = True
                        break
                elif check["name"] == "normaltest_p":
                    normaltest_p = ar_info["notes"].get("normaltest_p", None)
                    if normaltest_p is None: 
                        continue # skip this check if normaltest_p is not available in notes (this can happen for older archives before this feature is introduced)
                    if normaltest_p < check["threshold"]:
                        passed_checks = True
                        break
                else:
                    self.logger.warning(f"Quality check {check['name']} not recognized. Skipping this check.", layer=1)

            if passed_checks:
                res["good"].append(ar)
            else:
                res["bad"].append(ar)

        self.logger.info(f"Applied quality checks. {len(res['good'])}/{len(ar_list)} archives passed the checks. ", layer=1)
        return res

    def db_insert_archive_info(self, filename, rcvr):
        archive_hdl = ArchiveReader(filename)

        # Calculate matched filter SNR
        amps = archive_hdl.get_amps()
        prof_templ = self.db_hdl.get_config("__template:amps")
        if prof_templ is None:
            snr = 0
            normaltest_p = 0
        else:
            mf_snr = MatchedFilterSNR(amps, json.loads(prof_templ))
            snr = mf_snr.compute()
            normaltest_p = mf_snr.normaltest()[1]

        self.db_hdl.insert_archive_info(
            filename = self.utils.get_archive_id(filename), 
            psr_amps = archive_hdl.get_amps(), 
            # psr_snr = archive_hdl.get_snr(), 
            psr_snr = snr, 
            notes = {
                "md5": self.get_md5(filename), 
                "rcvr": rcvr, 
                "freq": archive_hdl.get_freq(), 
                "site": archive_hdl.get_telescope(), 
                "init_epoch": archive_hdl.get_epoch(), 
                "init_amps": archive_hdl.get_amps(), 
                "normaltest_p": normaltest_p
            }
        )

    def db_commit(self):
        self.db_hdl.conn.commit()
        
    def cleanup(self):
        if self.db_path is not None: # close if database connection was created here
            self.db_hdl.close()