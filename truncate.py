import time
import datetime
import os
import shutil
import glob
import argparse
from astropy.time import Time
from backend.datastores.database import database
from backend.utils.utils import utils

# Initialize parameters
psrbasedir = "./timing_sources"
pulsars = [v.split("/")[-1] for v in glob.glob(psrbasedir + "/*")]

# Define arguments
parser = argparse.ArgumentParser(description="Truncate timing info for CHAMPSS Timing Pipeline. ")
parser.add_argument("--psr", type=str, help="Pulsar name.")
parser.add_argument("-m", "--mjd", type=int, help="Remove timing info later than this MJD.")
parser.add_argument("-d", "--days", type=int, help="Remove timing info within the last N days before today.")
parser.add_argument("-l", type=int, help="Remove timing info within the last N days before the latest timing entry.")
parser.add_argument("-n", type=int, help="Remove the latest N timing info entries.")
parser.add_argument("--delete-archive-cache", action="store_true", help="Delete archive cache.")
parser.add_argument("--delete-database", action="store_true", help="Delete database.")
parser.add_argument("--truncate-config", action="store_true", help="Truncate config.")
parser.add_argument("--confirm", action="store_true", help="Confirm without asking.")
args = parser.parse_args()

# Sanity check for -m -l -n arguments
if sum([args.mjd is not None, args.l is not None, args.n is not None, args.days is not None]) > 1:
    print("Only one of -m, -l, -n, or -d can be specified at a time.")
    exit()
    
# Sanity check for positive values of -m, -l, -n, and -d arguments
for name in ("mjd", "days", "l", "n"):
    v = getattr(args, name)
    if v is not None and v <= 0:
        parser.error(f"argument {name} must be positive")

# Get pulsars
if args.psr is None:
    # Show warning if no pulsar name is provided (so that all pulsars will be truncated)
    utils.print_warning(f"WARNING: No pulsar name provided. Will truncate all pulsars in {psrbasedir}")
    for i in range(5):
        print(f"Press Ctrl+C to cancel or wait {i} seconds to continue...", end="\r")
        time.sleep(1)
    response = input("Type 'confirm' to continue: ")
    if response != "confirm":
        print("Truncation cancelled by user.")
        exit()
else:
    pulsars = [args.psr]

# Show information
print(f"Truncating timing info for pulsars: {pulsars}")
if args.mjd:
    print(f"Remove timing info later than MJD: {args.mjd}")
elif args.l:
    print(f"Remove timing info within the last N days before the latest timing entry: {args.l}")
elif args.n:
    print(f"Remove the latest N timing info entries: {args.n}")
elif args.days:
    print(f"Remove timing info within the last N days before today: {args.days}")
else:
    print(f"Truncate entire timing info: {not any([args.mjd, args.l, args.n])}")
print(f"Delete archive cache: {args.delete_archive_cache}")
print(f"Delete database: {args.delete_database}")
print(f"Truncate config: {args.truncate_config}")

# Ask for confirmation
if not args.confirm:
    response = input("Continue? (y/n): ")
    if response != "y":
        print("Truncation cancelled by user.")
        exit()
else:
    for i in range(2):
        print(f"Continuing in {2 - i} seconds...", end="\r")
        time.sleep(1)

# Loop through each pulsar and perform the operations
for pulsar in pulsars:
    db_path = f"./timing_sources/{pulsar}/champss_timing.sqlite3.db"
    parfile_bak_path = f"./timing_sources/{pulsar}/parfile_bak/initial_parfile.bak"
    parfile_path = f"./timing_sources/{pulsar}/pulsar.par"
    ar_cache_path = f"./timing_sources/{pulsar}/__champss_archive_cache__"
    
    # Truncate timing info
    if os.path.exists(db_path):
        print(f"Truncating timing info for {pulsar} at {db_path}")

        # Get timing info
        print("Reading database")
        with database(db_path, readonly=True) as db:
            all_timing_info = db.get_all_timing_info()

        # Get mjds for timing info and sort them in descending order
        mjds = [max(info["obs_mjds"]) for info in all_timing_info]
        mjds.sort(reverse=True)

        # Determine the MJD threshold for truncation based on the provided arguments
        this_truncate_mjd = None
        if args.mjd:
            if args.mjd <= min(mjds):
                print(f"Provided MJD {args.mjd} is earlier than the earliest timing info MJD {min(mjds)}")
                print(f" Skipping truncation for {pulsar}")
                continue
            
            this_truncate_mjd = args.mjd
        elif args.l:
            if max(mjds) - args.l <= min(mjds):
                print(f"Truncating by {args.l} days before the latest timing info MJD {max(mjds)} would result in a truncation MJD earlier than the earliest timing info MJD {min(mjds)}")
                print(f" Skipping truncation for {pulsar}")
                continue
            
            this_truncate_mjd = max(mjds) - args.l
        elif args.n:
            if len(mjds) <= args.n:
                print(f"Cannot truncate to the last {args.n} timing info entries because there are only {len(mjds)} entries available.")
                print(f" Skipping truncation for {pulsar}")
                continue

            this_truncate_mjd = mjds[args.n]
        elif args.days:
            mjd_today = Time(datetime.datetime.now()).mjd
            this_truncate_mjd = mjd_today - args.days

            if this_truncate_mjd <= min(mjds):
                print(f"Truncating by {args.days} days before today would result in a truncation MJD earlier than the earliest timing info MJD {min(mjds)}")
                print(f" Skipping truncation for {pulsar}")
                continue

        # Final safety check
        if this_truncate_mjd is not None:
            kept = [m for m in mjds if m <= this_truncate_mjd]
            if not kept:
                print(f"Truncation at MJD {this_truncate_mjd} would remove all timing info. Skipping {pulsar}")
                continue
            if len(kept) == len(mjds):
                print(f"No timing info later than MJD {this_truncate_mjd}. Skipping {pulsar}")
                continue

        print(f"Truncating timing info for {pulsar} with MJD threshold: {this_truncate_mjd}")

        # Reopen the database for writing
        with database(db_path) as db:
            # Truncate timing info and dealias history based on the calculated MJD threshold
            print("Truncating timing info and dealias history")
            if this_truncate_mjd is not None:
                db.remove_timing_info(mjd_later_than=this_truncate_mjd, show_warning=False)
                db.remove_dealias_history(mjd_later_than=this_truncate_mjd, show_warning=False)
            else:
                db.truncate_timing_info(show_warning=False)
                db.truncate_dealias_history(show_warning=False)

            # Truncate config if requested
            if args.truncate_config:
                print("Truncating config")
                db.truncate_config(show_warning=False)

            # Restore parfile
            if this_truncate_mjd is not None:
                last = db.get_last_timing_info()
                parfile_to_restore = last["notes"]["fitted_parfile"]
                print(f"Restoring parfile for {pulsar} at {parfile_path} from MJD {max(last['obs_mjds'])}")
                with open(parfile_path, "w") as f:
                    f.write(parfile_to_restore)
            else:
                print(f"Restoring initial parfile for {pulsar} at {parfile_path}")
                if os.path.exists(parfile_bak_path):
                    shutil.copy(parfile_bak_path, parfile_path)

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

        print(f"Finished truncation for {pulsar}")
    else:
        print(f"Skipping truncation for {pulsar}. Database does not exist.")