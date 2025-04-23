import argparse
import os
import glob
import traceback
from multiprocessing import Pool
from backend.datastores.tmg_master import tmg_master
from backend.utils.utils import utils
from backend.utils.logger import logger
from cli.config import CLIConfig
from cli.masterdb import CLIMasterDBHandler

# Load modules
logger = logger()
cli_config = CLIConfig()

# Initialize paths
tmg_master_path = "./timing_sources/TMGMaster.sqlite3.db"
champss_data__path = cli_config.config["data_paths"]["champss"]
chimepsr_fm__data_path = cli_config.config["data_paths"]["chimepsr_fm"]
chimepsr_fil__data_path = cli_config.config["data_paths"]["chimepsr_fil"]

# Initialize parser
parser = argparse.ArgumentParser(description="TMGMaster database utilities. ")
parser.add_argument("--auto-insert-raw-data", action="store_true", help="Auto insert all data into database. ")
parser.add_argument("--placeholder-if-corrupted", action="store_true", default=False, help="Insert placeholder if a file is corrupted. ")
parser.add_argument("--cleanup-raw-data", action="store_true", help="Cleanup unused raw data on the disk. ")
parser.add_argument("--mem", type=str, default="1G", help="Memory to use (e.g., 5G, 5M) for database fast mode. ")
args = parser.parse_args()
logger.info(f"TMGMaster path: {tmg_master_path}")

# Parse fast_mode_mem_gb
fast_mode_mem_gb = 1
if args.mem[-1] == "G":
    fast_mode_mem_gb = int(args.mem[:-1])
elif args.mem[-1] == "M":
    fast_mode_mem_gb = int(args.mem[:-1]) / 1024
else:
    raise ValueError(f"Invalid memory format: {args.mem}")

# Initialize CLI action handler
cli_masterdb_hdl = CLIMasterDBHandler(
    db_path=tmg_master_path,
    path_champss=champss_data__path, 
    path_chimepsr_fm=chimepsr_fm__data_path, 
    path_chimepsr_fil=chimepsr_fil__data_path, 
    fast_mode_mem_gb=fast_mode_mem_gb, 
    logger=logger.copy()
)

# Handle actions
if args.auto_insert_raw_data:
    logger.debug("Auto insert all data into database. ")
    logger.info(f"Add placehold if a file is corrupted: {args.placeholder_if_corrupted}")

    # Run action
    cli_masterdb_hdl.insert_data(placeholder_if_corrupted=args.placeholder_if_corrupted)

elif args.cleanup_raw_data:
    logger.debug("Cleanup unused raw data on the disk. ")
    
    # Run action
    cli_masterdb_hdl.cleanup_raw_data()
else:
    logger.error("No operation specified. ")
    parser.print_help()