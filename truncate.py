import time
import os
import shutil
import glob
import argparse
from backend.datastores.database import database
from backend.utils.utils import utils

# Initialize parameters
psrbasedir = "./timing_sources"
pulsars = [v.split("/")[-1] for v in glob.glob(psrbasedir + "/*")]

# Define arguments
parser = argparse.ArgumentParser(description="Truncate timing info for CHAMPSS Timing Pipeline. ")
parser.add_argument("--psr", type=str, help="Pulsar name.")
parser.add_argument("--info-later-than", type=int, help="Remove timing info later than this MJD.")
parser.add_argument("--delete-archive-cache", action="store_true", help="Delete archive cache.")
parser.add_argument("--delete-database", action="store_true", help="Delete database.")
parser.add_argument("--truncate-config", action="store_true", help="Truncate config.")
args = parser.parse_args()

# Get pulsars
if args.psr is None:
    # Show warning if no pulsar name is provided (so that all pulsars will be truncated)
    utils.print_warning(f"WARNING: No pulsar name provided. Will truncate all pulsars in {psrbasedir}")
    for i in range(5):
        print(f"Press Ctrl+C to cancel or wait {i} seconds to continue...", end="\r")
        time.sleep(1)
else:
    pulsars = [args.psr]

# Show information and ask for confirmation
print(f"Truncating timing info for pulsars: {pulsars}")
if args.info_later_than:
    print(f"Remove timing info later than MJD: {args.info_later_than}")
else:
    print(f"Truncate db & restore parfile: True by default")
print(f"Delete archive cache: {args.delete_archive_cache}")
print(f"Delete database: {args.delete_database}")
print(f"Truncate config: {args.truncate_config}")
response = input("Continue? (y/n): ")
if response != "y":
    exit()

# Loop through each pulsar and perform the operations
for pulsar in pulsars:
    db_path = f"./timing_sources/{pulsar}/champss_timing.sqlite3.db"
    parfile_bak_path = f"./timing_sources/{pulsar}/parfile_bak/initial_parfile.bak"
    parfile_path = f"./timing_sources/{pulsar}/pulsar.par"
    ar_cache_path = f"./timing_sources/{pulsar}/__champss_archive_cache__"
    
    # Truncate timing info
    if os.path.exists(db_path):
        print(f"Truncating timing info for {pulsar} at {db_path}")
        with database(db_path) as db:
            if args.info_later_than:
                db.remove_timing_info(mjd_later_than=args.info_later_than, show_warning=False)
                db.remove_dealias_history(mjd_later_than=args.info_later_than, show_warning=False)
            else:
                db.truncate_timing_info(show_warning=False)
                db.truncate_dealias_history(show_warning=False)

            # Truncate config if requested
            if args.truncate_config:
                db.truncate_config(show_warning=False)
        print("Done")
        
    # If in the case of info_later_than, we need to restore the parfile from the last timing info
    if args.info_later_than:
        with database(db_path) as db:
            last_timing_info = db.get_last_timing_info()
        parfile = last_timing_info["notes"]["fitted_parfile"]
        print(f"Restoring parfile for {pulsar} at {parfile_path} from MJD {max(last_timing_info['obs_mjds'])}")
        with open(parfile_path, "w") as f:
            f.write(parfile)
    else:
        # Or restore the initial parfile from the backup
        if os.path.exists(parfile_bak_path):
            print(f"Restoring parfile for {pulsar} at {parfile_path} from {parfile_bak_path}")
            shutil.copy(parfile_bak_path, parfile_path)
            print("Done")

    # Delete archive cache
    if args.delete_archive_cache and os.path.exists(ar_cache_path):
        print(f"Deleting archive cache for {pulsar} at {ar_cache_path}")
        shutil.rmtree(ar_cache_path)
        print("Done")

    # Delete database
    if args.delete_database and os.path.exists(db_path):
        print(f"Deleting database for {pulsar} at {db_path}")
        os.remove(db_path)
        print("Done")