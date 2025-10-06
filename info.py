import argparse
import json
from backend import champss_timing
from backend.datastores import database, tmg_master
from cli.config import CLIConfig
from cli.info import PSRInfo

def pretty_print_info(info_dict, indent=0):
    """Pretty print the info dictionary."""
    for key, value in info_dict.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            pretty_print_info(value, indent + 4)
        else:
            print(' ' * indent + f"{key}: {value}")

# Define
TIMING_SOURCES_PATH = "./timing_sources"
MASTER_DB_PATH = TIMING_SOURCES_PATH + "/TMGMaster.sqlite3.db"

# Load configuration
cli_config = CLIConfig(load_error=False)

# Parse arg
parser = argparse.ArgumentParser(description="Show information about timing sources.")
parser.add_argument("--psr", type=str, required=True, help="Pulsar name (run timing for all pulsars if not specified).")
args = parser.parse_args()

# Show masterdb info
print(f"Fetching data...")
with tmg_master.tmg_master(MASTER_DB_PATH, readonly=True) as tmgm:
    with database.database(f"./{TIMING_SOURCES_PATH}/{args.psr}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
        psr_info = PSRInfo(psr=args.psr, masterdb=tmgm, sourcedb=db_hdl).get_info()

# Show info in pretty format
print(f"Showing information for pulsar: {args.psr}")
pretty_print_info(psr_info)