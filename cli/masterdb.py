import os
import glob
import traceback
import shutil
import time
from multiprocessing import Pool
from backend.datastores.tmg_master import tmg_master
from backend.utils.utils import utils
from backend.utils.logger import logger

class CLIMasterDBHandler:
    def __init__(self, db_path, backends, fast_mode_mem_gb, logger=logger()):
        self.db_path = db_path
        self.backends = backends
        self.fast_mode_mem_gb = fast_mode_mem_gb
        self.logger = logger

    # def ls_champss(self, psr="*"):
    #     return glob.glob(self.path_champss.replace("%PSR%", psr))
        
    # def ls_chimepsr_fm(self, psr="*"):
    #     return glob.glob(self.path_chimepsr_fm.replace("%PSR%", psr))

    # def ls_chimepsr_fil(self, psr="*"):
    #     return glob.glob(self.path_chimepsr_fil.replace("%PSR%", psr))

    def ls(self, path, psr="*"):
        """
        List files in the given path with the specified pulsar ID.
        """
        return glob.glob(path.replace("%PSR%", psr))
    
    def insert_data(self, placeholder_if_corrupted):
        with tmg_master(self.db_path, fast_mode=True, mem_gb=self.fast_mode_mem_gb) as tm_hdl:
            db_records = tm_hdl.get_ar_ids_idxed_by_psr_id()

            for bknd, info in self.backends.items():
                self.logger.info(f"Inserting raw data from {info['label']} (path: {info['data_path']})")
                files = self.ls(info['data_path'], "*")
                for i, file in enumerate(files):
                    psr_id = file.split("/")[-2] # TODO: there should be a better way to get the pulsar ID!!
                    ar_id = utils.get_archive_id(file)

                    if psr_id in db_records:
                        if ar_id in db_records[psr_id]:
                            continue

                    self.logger.debug(f"[{i+1}/{len(files)}] Inserting: {psr_id} -> {file}", end="\r", layer=1)

                    try:
                        tm_hdl.insert_raw_data_from_file(
                            psr_id = psr_id, 
                            location = file, 
                            backend = bknd, 
                            format = "auto", 
                            skip_if_exists = True, 
                            placeholder_if_corrupted = placeholder_if_corrupted
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to insert: {psr_id} -> {file} ({e})")
                        self.logger.error(traceback.format_exc())

            # # CHAMPSS
            # self.logger.info(f"Inserting raw data from CHAMPSS (path: {self.path_champss})")
            # # champss__files = glob.glob(champss_data__path.replace("%PSR%", "*"))
            # champss__files = self.ls_champss("*")
            # for i, path in enumerate(champss__files):
            #     psr_id = path.split("/")[-2]
            #     ar_id = utils.get_archive_id(path)

            #     if psr_id in db_records:
            #         if ar_id in db_records[psr_id]:
            #             continue

            #     self.logger.debug(f"[{i+1}/{len(champss__files)}] Inserting: {psr_id} -> {path}", end="\r")

            #     try:
            #         tm_hdl.insert_raw_data_from_file(
            #             psr_id = psr_id, 
            #             location = path, 
            #             backend = "champss", 
            #             format = "archive", 
            #             skip_if_exists = True, 
            #             placeholder_if_corrupted = placeholder_if_corrupted
            #         )
            #     except Exception as e:
            #         self.logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
            #         self.logger.error(traceback.format_exc())
            
            # # CHIME/Pulsar Fold-mode
            # self.logger.info(f"Inserting raw data from CHIMEPSR_FM (path: {self.path_chimepsr_fm})")
            # # chimepsr_fm__files = glob.glob(chimepsr_fm__data_path.replace("%PSR%", "*"))
            # chimepsr_fm__files = self.ls_chimepsr_fm("*")
            # for i, path in enumerate(chimepsr_fm__files):
            #     psr_id = path.split("/")[-2]
            #     ar_id = utils.get_archive_id(path)

            #     if psr_id in db_records:
            #         if ar_id in db_records[psr_id]:
            #             continue

            #     self.logger.debug(f"[{i+1}/{len(chimepsr_fm__files)}] Inserting: {psr_id} -> {path}", end="\r")

            #     try:
            #         tm_hdl.insert_raw_data_from_file(
            #             psr_id = psr_id, 
            #             location = path, 
            #             backend = "chimepsr_fm", 
            #             format = "archive", 
            #             skip_if_exists=True, 
            #             placeholder_if_corrupted = placeholder_if_corrupted
            #         )
            #     except Exception as e:
            #         self.logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
            #         self.logger.error(traceback.format_exc())

            # # CHIME/Pulsar Filterbank
            # self.logger.info(f"Inserting raw data from CHIMEPSR_FIL (path: {self.path_chimepsr_fil})")
            # # chimepsr_fil__files = glob.glob(chimepsr_fil__data_path.replace("%PSR%", "*"))
            # chimepsr_fil__files = self.ls_chimepsr_fil("*")
            # for i, path in enumerate(chimepsr_fil__files):
            #     psr_id = path.split("/")[-2]
            #     ar_id = utils.get_archive_id(path)

            #     if psr_id in db_records:
            #         if ar_id in db_records[psr_id]:
            #             continue
                        
            #     self.logger.debug(f"[{i+1}/{len(chimepsr_fil__files)}] Inserting (skip if exist): {psr_id} -> {path}", end="\r")
                
            #     try:
            #         tm_hdl.insert_raw_data_from_file(
            #             psr_id = psr_id, 
            #             location = path, 
            #             backend = "chimepsr_fil", 
            #             format = "filterbank", 
            #             skip_if_exists=True, 
            #             placeholder_if_corrupted = placeholder_if_corrupted
            #         )
            #     except Exception as e:
            #         self.logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
            #         self.logger.error(traceback.format_exc())
    
    def cleanup_raw_data(self):
        with tmg_master(self.db_path, fast_mode=True, mem_gb=self.fast_mode_mem_gb) as tm_hdl:
            # Get pulsars that are in use in the timing pipeline
            psrs_in_use = tm_hdl.get_psr_ids(table="timing")

            # Get all pulsars in record
            db_records = tm_hdl.get_ar_ids_idxed_by_psr_id()

            # Cross-matching to get unused pulsars
            psrs_unused = []
            tot_size = 0
            tot_n_files = 0
            for this_psr in db_records.keys():
                if this_psr not in psrs_in_use:
                    # Append info
                    psrs_unused.append({
                        "psr_id": this_psr, 
                        "records": [tm_hdl.get_raw_data(ar_id=ar_id, psr_id=this_psr)[0] for ar_id in db_records[this_psr]]
                    })

                    # Calculate total size
                    total_size = 0
                    for ar_info in psrs_unused[-1]["records"]:
                        total_size += ar_info["size"]

                    # Add to stats
                    tot_size += total_size
                    tot_n_files += len(db_records[this_psr])

                    self.logger.debug(f"Unused pulsar: {this_psr} [n_files= {len(db_records[this_psr])}, size= {total_size/1e9:.2f} GB]")

            # Prompt user to confirm
            self.logger.info(f"Will cleanup {len(psrs_unused)} pulsars.")
            self.logger.info(f"This action will remove {tot_n_files} files, freeing {tot_size/1e9:.2f} GB of space.")

            if input("Commit changes? Type \"commit\" to confirm.").lower() != "commit":
                self.logger.success("Aborted.")
                return

            # Remove files from masterdb
            self.logger.debug("Removing raw data from masterdb...")
            for this_psr in psrs_unused:
                self.logger.debug(f"Removing {this_psr['psr_id']} from masterdb.", layer=1)
                for ar_info in this_psr["records"]:
                    try:
                        tm_hdl.get_raw_data(ar_id=ar_info["ar_id"], psr_id=this_psr["psr_id"], remove_action=True)
                    except Exception as e:
                        self.logger.error(f"Failed to remove {ar_info['ar_id']} from masterdb: {e}")
                        self.logger.error(traceback.format_exc())
                        time.sleep(1)

            # Remove files from disk
            self.logger.debug("Removing raw data from disk...")
            for this_psr in psrs_unused:
                self.logger.debug(f"Removing {this_psr['psr_id']} from disk.", layer=1)
                for ar_info in this_psr["records"]:
                    if os.path.exists(ar_info["location"]):
                        try:
                            os.remove(ar_info["location"])
                        except Exception as e:
                            self.logger.error(f"Failed to remove {ar_info['location']}: {e}")
                            self.logger.error(traceback.format_exc())
                            time.sleep(1)
                    else:
                        self.logger.warning(f"File {ar_info['location']} does not exist. Skipping.")

            self.logger.success("Cleanup completed.")
            self.logger.success(f"Freed {tot_size/1e9:.2f} GB of space.")
            self.logger.success(f"Removed {tot_n_files} files, {len(psrs_unused)} pulsars.")

    def set_corrupted(self, psr_id, ar_id):
        with tmg_master(self.db_path, fast_mode=True, mem_gb=self.fast_mode_mem_gb) as tm_hdl:
            try:
                tm_hdl.update_raw_data(
                    psr_id=psr_id, 
                    ar_id=ar_id, 
                    status="corrupted"
                )
                self.logger.success(f"Set {psr_id} -> {ar_id} as corrupted.")
            except Exception as e:
                self.logger.error(f"Failed to set {psr_id} -> {ar_id} as corrupted: {e}")
                self.logger.error(traceback.format_exc())