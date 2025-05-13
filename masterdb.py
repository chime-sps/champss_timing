import argparse
import os
import glob
import traceback
from multiprocessing import Pool
from backend.datastores.tmg_master import tmg_master
from backend.utils.utils import utils
from backend.utils.logger import logger
from cli.config import CLIConfig

# Load modules
logger = logger()
cli_config = CLIConfig()

# Initialize paths
tmg_master_path = "./timing_sources/TMGMaster.sqlite3.db"
champss_data__path = cli_config.config["data_paths"]["champss"]
chimepsr_fm__data_path = cli_config.config["data_paths"]["chimepsr_fm"]
chimepsr_fil__data_path = cli_config.config["data_paths"]["chimepsr_fil"]

parser = argparse.ArgumentParser(description="TMGMaster database utilities. ")
parser.add_argument("--auto-insert-raw-data", action="store_true", help="Auto insert all data into database. ")
parser.add_argument("--placeholder-if-corrupted", action="store_true", default=False, help="Insert placeholder if a file is corrupted. ")
parser.add_argument("--mem", type=str, default="1G", help="Memory to use (e.g., 5G, 5M) for database fast mode. ")
args = parser.parse_args()
logger.info(f"TMGMaster path: {tmg_master_path}")

fast_mode_mem_gb = 1
if args.mem[-1] == "G":
    fast_mode_mem_gb = int(args.mem[:-1])
elif args.mem[-1] == "M":
    fast_mode_mem_gb = int(args.mem[:-1]) / 1024
else:
    raise ValueError(f"Invalid memory format: {args.mem}")

if args.auto_insert_raw_data:
    logger.debug("Auto insert all data into database. ")
    logger.info(f"Add placehold if a file is corrupted: {args.placeholder_if_corrupted}")
    
    with tmg_master(tmg_master_path, fast_mode=True, mem_gb=fast_mode_mem_gb) as tm_hdl:
        db_records = tm_hdl.get_ar_ids_idxed_by_psr_id()

        logger.info(f"Inserting raw data from CHAMPSS (path: {champss_data__path})")
        champss__files = glob.glob(champss_data__path.replace("%PSR%", "*"))
        for i, path in enumerate(champss__files):
            psr_id = path.split("/")[-2]
            ar_id = utils.get_archive_id(path)

            if psr_id in db_records:
                if ar_id in db_records[psr_id]:
                    continue

            logger.debug(f"[{i+1}/{len(champss__files)}] Inserting: {psr_id} -> {path}", end="\r")

            try:
                tm_hdl.insert_raw_data_from_file(
                    psr_id = psr_id, 
                    location = path, 
                    backend = "champss", 
                    format = "archive", 
                    skip_if_exists = True, 
                    placeholder_if_corrupted = args.placeholder_if_corrupted
                )
            except Exception as e:
                logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
                logger.error(traceback.format_exc())
        
        logger.info(f"Inserting raw data from CHIMEPSR_FM (path: {chimepsr_fm__data_path})")
        chimepsr_fm__files = glob.glob(chimepsr_fm__data_path.replace("%PSR%", "*"))
        for i, path in enumerate(chimepsr_fm__files):
            psr_id = path.split("/")[-2]
            ar_id = utils.get_archive_id(path)

            if psr_id in db_records:
                if ar_id in db_records[psr_id]:
                    continue

            logger.debug(f"[{i+1}/{len(chimepsr_fm__files)}] Inserting: {psr_id} -> {path}", end="\r")

            try:
                tm_hdl.insert_raw_data_from_file(
                    psr_id = psr_id, 
                    location = path, 
                    backend = "chimepsr_fm", 
                    format = "archive", 
                    skip_if_exists=True, 
                    placeholder_if_corrupted = args.placeholder_if_corrupted
                )
            except Exception as e:
                logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
                logger.error(traceback.format_exc())

        logger.info(f"Inserting raw data from CHIMEPSR_FIL (path: {chimepsr_fil__data_path})")
        chimepsr_fil__files = glob.glob(chimepsr_fil__data_path.replace("%PSR%", "*"))
        for i, path in enumerate(chimepsr_fil__files):
            psr_id = path.split("/")[-2]
            ar_id = utils.get_archive_id(path)

            if psr_id in db_records:
                if ar_id in db_records[psr_id]:
                    continue
                    
            logger.debug(f"[{i+1}/{len(chimepsr_fil__files)}] Inserting (skip if exist): {psr_id} -> {path}", end="\r")
            
            try:
                tm_hdl.insert_raw_data_from_file(
                    psr_id = psr_id, 
                    location = path, 
                    backend = "chimepsr_fil", 
                    format = "filterbank", 
                    skip_if_exists=True, 
                    placeholder_if_corrupted = args.placeholder_if_corrupted
                )
            except Exception as e:
                logger.error(f"Failed to insert: {psr_id} -> {path} ({e})")
                logger.error(traceback.format_exc())
else:
    logger.error("No operation specified. ")
    parser.print_help()