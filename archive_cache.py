import os
import shutil
import hashlib

from .exec import exec
from .utils import utils
from .database import database
from .archive_utils import archive_utils

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
        
    def add_archive(self, filename):
        if not os.path.exists(filename):
            raise Exception(f"Archive {filename} does not exist.")

        # copy archive to cache
        print(f"  [Archive] {filename} -> archive cache")
        shutil.copyfile(filename, f"{self.cache_dir}/{self.utils.get_archive_id(filename)}")

        # insert archive info to database
        print(f"  [Archive] {filename} -> database")
        self.db_insert_archive_info(filename)

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

    def update_model(self, parfile="auto", n_pools="auto", tempdir="auto", cleanup=True):
        if parfile == "auto":
            parfile = f"{self.psr_dir}/pulsar.par"

        if tempdir == "auto":
            tempdir = f"{self.cache_dir}/temp"

        if not os.path.exists(tempdir):
            os.makedirs(tempdir, exist_ok=True)

        # get all archives
        archives = []
        for ar in self.db_hdl.get_all_archive_info():
            this_path = f"{self.cache_dir}/{ar['filename']}"
            this_temp_path = f"{tempdir}/{ar['filename']}"
            if os.path.exists(f"{this_path}"):
                # copy archive to temp directory
                shutil.copyfile(this_path, this_temp_path)
                # append to archives
                archives.append(this_temp_path)
            else:
                self.utils.print_warning(f"Archive {ar['filename']} not found in cache. Skipping.")

        # Remove TZRSITE to fix a problem with psrchive for CHIME observations
        open(f"{tempdir}/pulsar.par.tmp", "w").write(
            open(parfile).read().replace("TZRSITE", "# TZRSITE")
        )

        # update model for each archive
        self.exec_update_model(archives, f"{tempdir}/pulsar.par.tmp", n_pools=n_pools)
        utils.print_success(f"  [update_model] timing model updated for {len(archives)} observations. ")

        # update psr_amps in database
        for i, ar in enumerate(archives):
            print(f"  [update_model] updating archive information in database for {i + 1}/{len(archives)}... ")
            self.db_update_psr_amps(ar, commit=False)
        print(f"  [update_model] committing changes to database... ")
        self.db_commit()
        utils.print_success(f"  [update_model] archive information in database updated for {len(archives)} observations. ")

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

    def db_insert_archive_info(self, filename):
        archive_hdl = archive_utils(filename)

        self.db_hdl.insert_archive_info(
            filename = self.utils.get_archive_id(filename), 
            psr_amps = archive_hdl.get_amps(), 
            psr_snr = archive_hdl.get_snr(), 
            notes = {
                "md5": self.get_md5(filename)
            }
        )
    
    def db_update_psr_amps(self, filename, commit=True):
        archive_hdl = archive_utils(filename)

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
    
    def db_commit(self):
        self.db_hdl.conn.commit()
        
    def cleanup(self):
        if self.db_path is not None: # close if database connection was created here
            self.db_hdl.close()