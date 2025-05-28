import os
import shutil
import hashlib
import tqdm
import numpy as np
from multiprocessing import Pool

from .database import database
from ..utils.exec import exec
from ..utils.utils import utils
from ..io.archive import ArchiveReader

# Putting function outside of the class since db_hdl cannot be pickled and passed to Pool
def _archive_cache__db_update_psr_amps_many__get_amp_and_snr(filename):
    archive_hdl = ArchiveReader(filename)
    return archive_hdl.get_amps(), archive_hdl.get_snr()

def _archive_cache__update_model__get_md5(filename):
    md5 = hashlib.md5()

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest()

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
        # TODO: we might want replace this method with the one in processing.archive_shutils sometime in the future.
        if parfile == "auto":
            parfile = f"{self.psr_dir}/pulsar.par"

        if tempdir == "auto":
            tempdir = f"{self.cache_dir}/temp"

        if not os.path.exists(tempdir):
            os.makedirs(tempdir, exist_ok=True)

        # get all timed files
        timed_files = []
        timing_info = self.db_hdl.get_all_timing_info()
        for this_info in timing_info:
            timed_files += this_info["files"]
        timed_files = list(set(timed_files))

        # get all archives
        archives = []
        archives_tmp = []
        archives_rcvr = []
        for ar in self.db_hdl.get_all_archive_info():
            if ar["filename"] not in timed_files:
                self.utils.print_warning(f"Archive {ar['filename']} not in timing_info. Skipping.")
                continue

            this_path = f"{self.cache_dir}/{ar['filename']}"
            this_temp_path = f"{tempdir}/{ar['filename']}"
            if os.path.exists(f"{this_path}"):
                # copy archive to temp directory
                shutil.copyfile(this_path, this_temp_path)
                # append to archives
                archives.append(this_path)
                archives_tmp.append(this_temp_path)
                archives_rcvr.append(ar["notes"]["rcvr"])
            else:
                self.utils.print_warning(f"Archive {ar['filename']} not found in cache. Skipping.")

        # Remove TZRSITE to fix a problem with psrchive for CHIME observations
        open(f"{tempdir}/pulsar.par.tmp", "w").write(
            open(parfile).read().replace("TZRSITE", "# TZRSITE")
        )

        # update model for each archive
        self.exec_update_model(archives_tmp, f"{tempdir}/pulsar.par.tmp", n_pools=n_pools)
        utils.print_success(f"  [update_model] timing model updated for {len(archives_tmp)} observations. ")

        # get md5 of archives
        with Pool(processes=n_pools) as pool:
            archives_md5s = list(pool.imap(_archive_cache__update_model__get_md5, archives))
            archives_tmp_md5s = list(pool.imap(_archive_cache__update_model__get_md5, archives_tmp))

        # check whether the files were updated
        for i in range(len(archives_tmp)):
            # if self.get_md5(archives_tmp[i]) == self.get_md5(archives[i]):
            if archives_tmp_md5s[i] == archives_md5s[i]:
                this_toa_notes = self.db_hdl.get_toa_by_filename(self.utils.get_archive_id(archives_tmp[i]))["notes"]
                if "remark" in this_toa_notes:
                    if this_toa_notes["remark"] == "INVALID_TOA":
                        self.utils.print_warning(f"Failed to update model for {archives_tmp[i]} due to INVALID_TOA.")
                        continue
                raise Exception(f"Failed to update model for {archives_tmp[i]}")
            
        # apply jump for each archive
        for rcvr in jumps:
            if jumps[rcvr][0] == 0:
                continue
            
            jump_ars = []
            jump_ars_md5s = []
            for i, ar in enumerate(archives_tmp):
                if archives_rcvr[i] == rcvr:
                    jump_ars.append(ar)
                    jump_ars_md5s.append(self.get_md5(ar))
            
            if len(jump_ars) == 0:
                continue

            print(f"  [update_model] applying jump for {len(jump_ars)} archives (RCVR={rcvr}, JUMP={jumps[rcvr]})... ")
            self.exec_apply_jump(jump_ars, jumps[rcvr][0], f"{tempdir}/pulsar.par.tmp", n_pools=n_pools)

            # check whether the files were updated
            for i in range(len(jump_ars)):
                if self.get_md5(jump_ars[i]) == jump_ars_md5s[i]:
                    this_toa_notes = self.db_hdl.get_toa_by_filename(self.utils.get_archive_id(jump_ars[i]))["notes"]
                    if "remark" in this_toa_notes:
                        if this_toa_notes["remark"] == "INVALID_TOA":
                            self.utils.print_warning(f"Failed to apply jump for {jump_ars[i]} due to INVALID_TOA.")
                            continue
                    
                    # check if the archive is actually blank (so that the file before and after jump are the same)
                    this_archive_info = self.db_hdl.get_archive_info_by_filename(self.utils.get_archive_id(jump_ars[i]))
                    if this_archive_info["timestamp"] != 0:
                        if np.std(this_archive_info["psr_amps"]) == 0:
                            self.utils.print_warning(f"Failed to apply jump for {jump_ars[i]} due to blank archive (std=0).")
                            continue

                    raise Exception(f"Failed to apply jump for {jump_ars[i]} ({i + 1}/{len(jump_ars)})")

        # update psr_amps in database
        # for i, ar in enumerate(archives_tmp):
        #     print(f"  [update_model] updating archive information in database for {i + 1}/{len(archives_tmp)}... ")
        #     self.db_update_psr_amps(ar, commit=False)
        # print(f"  [update_model] committing changes to database... ")
        # self.db_commit()
        print(f"  [update_model] updating archive information in database... ")
        self.db_update_psr_amps_many(archives_tmp, n_pools=n_pools, commit=True)
        utils.print_success(f"  [update_model] archive information in database updated for {len(archives_tmp)} observations. ")

        # cleanup
        if cleanup:
            shutil.rmtree(tempdir)

        return True
    
    def exec_update_model(self, fs, parfile, n_pools=4):
        # pam -e .pam -E pulsar.par xxx.ar.clfd.FTp
        exec_hlr = exec(n_pools=n_pools)
        for f in fs:
            exec_hlr.append(f"pam -m -E {parfile} {f}") # -m: modify the original file
        exec_hlr.run()
        
        if(not exec_hlr.check()):
            raise Exception("Failed to update model")

        return True
    
    def exec_apply_jump(self, fs, jump, parfile, n_pools=4):
        # pam -m -r -0.1730288421 xxx.ar
        
        # read f0 from parfile
        f0 = None
        with open(parfile, "r") as f:
            for l in f:
                if l.strip().startswith("F0"):
                    f0 = float(l.strip().split()[1])
                    break
        
        if f0 is None:
            raise Exception("Failed to read F0 from parfile")
        
        # calculate phase offset
        if jump < 0:
            jump = (1/f0) + jump
        phase_offset = - ((jump / (1/f0)) % 1)

        # apply phase offset
        exec_hlr = exec(n_pools=n_pools)
        for f in fs:
            exec_hlr.append(f"pam -m -r {phase_offset} {f}")
        exec_hlr.run()

        if(not exec_hlr.check()):
            raise Exception("Failed to apply jump cpt")
        
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
        with Pool(processes=n_pools) as pool:
            results = list(tqdm.tqdm(pool.imap(_archive_cache__db_update_psr_amps_many__get_amp_and_snr, filenames), total=len(filenames)))
        
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