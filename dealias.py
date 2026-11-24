import argparse
import glob
import os
import time
import numpy as np
import traceback
import pandas as pd
import datetime

from cli.config import CLIConfig
from backend.datastores.database import database
from backend.datastores.tmg_master import tmg_master
from backend.utils.logger import logger
from backend.utils.utils import utils
from backend.tools.dealiasing import (
    Dealiasing, 
    estimate_minimal_obs_for_subint_snr_from_database, 
    get_files_for_dealiasing_pipeline
)

##################################################
# Initialize parameters                          #
##################################################

# Initialize modules and parameters
logger = logger()
cli_config = CLIConfig()
TIMING_SOURCES_PATH = "./timing_sources"
MASTER_DB_PATH = TIMING_SOURCES_PATH + "/TMGMaster.sqlite3.db"
TEMPDIR = "./__champss_timing__workspace/__alias_utils_workspaces"
JUMPS = cli_config.get_config()["toa_jumps"]
t_start = time.time()
mdb_hdl = tmg_master(MASTER_DB_PATH)

# Initialize parser
parser = argparse.ArgumentParser(description=f"Run the de-aliasing pipeline for a timing source. The pipeline will also automatically run after the main pipeline processing regularly once the signal-to-noise ratio of the stacked data reaches a certain threshold. \nThe pipeline will generate dealiasing diagnostics under {TIMING_SOURCES_PATH}/<psr_id>/dealias. If the pulsar's timing solution is likely aliased, the pipeline will attempt to create an unaliased timing solution and write it as pulsar.dealiased.par. \nTo accept the new unaliased timing solution, replace {TIMING_SOURCES_PATH}/<psr_id>/pulsar.par with {TIMING_SOURCES_PATH}/<psr_id>/dealiasing/pulsar.dealiased.par. \nTo trigger the dealiasing pipeline run automatically after the next main pipeline process (so you don't have to manually re-run the de-aliasing pipeline), remove the {TIMING_SOURCES_PATH}/<psr_id>/dealias directory. \nTo refit the new timing solution, run `python champss_timing pintk --psr <psr_id> --dealiased-par`.")
parser.add_argument("-p", "--psr", type=str, help="Pulsar name.", default=None, required=True)
parser.add_argument("-n", "--ncpus", type=int, help="Number of pools.", default=1)
parser.add_argument("-s", "--subint-snr", type=float, help="Desired SNR per subintegration (default: 15.0).", default=15.0, required=False)
parser.add_argument("-N", "--n-files", type=int, help="Maximum number of archives to use (set N=0 to use all available files; default=200).", default=200, required=False)
parser.add_argument("--parfile", type=str, help="Specify the path to timing model (default: ./timing_sources/<psr_id>/pulsar.par).", default=None, required=False)
parser.add_argument("--ignore-subints", type=str, help="Subint indices to ignore for alias factor calculation (e.g., 0,1,2).", default=None, required=False)
parser.add_argument("--n-subints", type=int, help="Binsize for alias factor calculation.", default=8, required=False)
parser.add_argument("--n-bins", type=int, help="Number of bins for alias factor calculation.", default=256, required=False)
parser.add_argument("--n-freqs", type=int, help="Number of frequency channels (frequencies are averaged during de-aliasing, but multi-frequency stacked data are produced as a by-product for other purposes).", default=256, required=False)
parser.add_argument("--mjd-range", type=str, help="MJD range to use for alias factor calculation (e.g., 59000:60000, inclusive).", default=None, required=False)
parser.add_argument("--backend", type=str, help="Use the data from the specified backend.", default=None, required=False)
parser.add_argument("--show-history", action="store_true", help="Show dealias history and skip processing.", required=False, default=False)
parser.add_argument("--no-commit", action="store_true", help="Do not commit changes to the psrdir and database.", required=False, default=False)
parser.add_argument("--no-beep", action="store_true", help="Do not beep when the process is finished.", required=False, default=False)
args = parser.parse_args()

# Get the path to the source database
db_path = f"{TIMING_SOURCES_PATH}/{args.psr}/champss_timing.sqlite3.db"


##################################################
# Sanity checks                                  #
##################################################

# Ignore subints
ignore_subints = []
if args.ignore_subints is not None:
    ignore_subints = list(map(int, args.ignore_subints.split(",")))

# MJD range
mjd_range = []
if args.mjd_range is not None:
    mjd_range = list(map(float, args.mjd_range.split(":")))
    if len(mjd_range) != 2:
        raise ValueError("Invalid MJD range.")


##################################################
# Show dealias history                           #
##################################################

# Print dealias histories
if args.show_history:
    # Fetech history
    with database(db_path, readonly=True) as db_hdl:
        dealias_histories = db_hdl.get_all_dealias_history()

    # Format dealias history
    dealias_histories_formatted = {"Last Updated": [], "AF": [], "N_stacked": [], "SNR_stacked": [], "Remark": []}
    for this_history in dealias_histories:
            # A bug in early version of the dealias_utils caused some numbers to be saved as bytes
        if type(this_history["alias_factor"]) == bytes:
            import struct
            this_history["alias_factor"] = struct.unpack('d', this_history["alias_factor"])[0]
        if type(this_history["snr_stacked"]) == bytes:
            import struct
            this_history["snr_stacked"] = struct.unpack('f', this_history["snr_stacked"])[0]

        dealias_histories_formatted["Last Updated"].append(datetime.datetime.fromtimestamp(this_history["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'))
        dealias_histories_formatted["AF"].append(this_history["alias_factor"])
        dealias_histories_formatted["N_stacked"].append(this_history["n_stacked"])
        dealias_histories_formatted["SNR_stacked"].append(this_history["snr_stacked"])
        dealias_histories_formatted["Remark"].append(this_history["notes"]["remark"])
        
    dealias_histories_txt = (
        "======================== DEALIAS HISTORIES ========================" + "\n" +
        pd.DataFrame(dealias_histories_formatted).to_string() + "\n" +
        "==================================================================="
    )
    logger.info(dealias_histories_txt)
    exit()


##################################################
# Print user input                               #
##################################################

# Print arguments
logger.info(f"Pulsar: {args.psr}")
logger.info(f"Number of pools: {args.ncpus}")
logger.info(f"Ignore subints: {args.ignore_subints}")
logger.info(f"Number of subints: {args.n_subints}")
logger.info(f"MJD range: {mjd_range}")


##################################################
# Gather useful information                      #
##################################################

# Process pulsars
logger.debug(f"Processing pulsar {args.psr}")

# Get psrdir
psrdir = f"./{TIMING_SOURCES_PATH}/{args.psr}"
logger.info(f"PsrDir: {psrdir}")
if not os.path.exists(psrdir):
    raise ValueError(f"Pulsar directory not found: {psrdir}")

# Get parfile
if args.parfile is not None:
    parfile = args.parfile
else:
    parfile = f"./{TIMING_SOURCES_PATH}/{args.psr}/pulsar.par"

#  determine range of mjds
if mjd_range == []:
    mjd_range = utils.read_start_end_from_parfile(parfile, raise_exception=False)
    logger.info(f"Auto-detected MJD range: {mjd_range}")

# Check if parfile exists
if not os.path.exists(parfile):
    raise ValueError(f"Parfile not found for pulsar {args.psr}")

# Compute the desired number of archives required
min_obs_by_rcvr = estimate_minimal_obs_for_subint_snr_from_database(db_path, args.n_subints, args.subint_snr)


##################################################
# Find what data to use                          #
##################################################

# Get the list all available files
ar_list_by_backend = {}
for f_info in mdb_hdl.get_raw_data_by_mjd_range(args.psr, mjd_range=mjd_range):
    # Skip bad files
    if f_info["status"] != "good":
        logger.info(f"Skipping archive {f_info['location']} due to bad status.")
        continue

    if f_info["backend"] not in ar_list_by_backend:
        ar_list_by_backend[f_info["backend"]] = []

    ar_list_by_backend[f_info["backend"]].append({
        "location": f_info["location"],
        "backend": f_info["backend"],
        "mjd": f_info["mjd"]
    })

# Determine backend data to use
this_backend = None
if args.backend is None: 
    this_backend, ar_list, min_n_obs = get_files_for_dealiasing_pipeline(min_obs_by_rcvr, ar_list_by_backend)

    if this_backend is None:
        raise ValueError("No suitable backend found with enough good archives.")

    logger.info(f"Using the most sensitive backend that has enough good archives: {this_backend}")
else:
    this_backend = args.backend
    ar_list = ar_list_by_backend.get(this_backend, [])
    min_n_obs = min_obs_by_rcvr.get(this_backend, 0)
    logger.info(f"Using user specified backend: {this_backend}")

# Check if there are archives to process
if len(ar_list) == 0:
    raise ValueError(f"No archives found for pulsar {args.psr} from backend {this_backend}")


##################################################
# Determine the number of files to use           #
##################################################

# Determine the number of files to use
n_files = min_n_obs * 2
if args.n_files is not None:
    if args.n_files == 0: # use all available archives
        args.n_files = len(ar_list)
    elif args.n_files < n_files: # cap on the user specified number of files
        n_files = args.n_files

# Sort the list of archives by mjd (newest to oldest)
ar_list = sorted(ar_list, key=lambda ar: ar["mjd"], reverse=True)

# Cap the list to the specified number of files (always the newest ones)
ar_list = ar_list[:n_files]

logger.info(f"Cap the list of archives to the newest {n_files} files.")


##################################################
# Show information                               #
##################################################

# Show archive information
logger.info(f"Number of archives: {len(ar_list)}")
for ar in ar_list:
    logger.info(f"{ar['location']} (backend: {ar['backend']})", layer=1)


##################################################
# Run pipeline                                   #
##################################################

# Find alias
with Dealiasing(
    psrdir=f"./{TIMING_SOURCES_PATH}/{args.psr}", 
    ar_list=ar_list, 
    parfile=parfile, 
    n_subints=args.n_subints, 
    n_bins=args.n_bins, 
    n_freqs=args.n_freqs, 
    jumps=JUMPS, 
    workspace=TEMPDIR, 
    ncpus=args.ncpus, 
    logger=logger.copy()
) as adg:
    
    # Optimize DM
    adg.optimize_dm()

    # Resolve aliasing
    adg.resolve_aliasing(
        ignore_subints=ignore_subints
    )

    # Get dealias results
    dealias_results = {
        "n_stacked": adg.payload.af_res_summary["n_stacked"],
        "alias_factor": adg.payload.af_res_summary["alias_factor"],
        "snr_stacked": adg.payload.af_res_summary["snr_stacked"],
        "remark": adg.payload.get_remark()
    }

    # Commit changes
    if not args.no_commit:
        adg.commit()


##################################################
# Finishing                                      #
##################################################

# print summary
dealias_results = pd.DataFrame([dealias_results])
dealias_results_txt = (
        "====================== DEALIAS RESULTS ======================" + "\n" +
        pd.DataFrame(dealias_results).to_string() + "\n" +
        "============================================================="
)
logger.info(dealias_results_txt)

# Send alert when finished
if not args.no_beep:
    logger.cli_alert()