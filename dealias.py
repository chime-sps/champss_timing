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
from backend.tools.alias_utils import alias_utils
from backend.utils.logger import logger
from backend.utils.utils import utils

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
parser = argparse.ArgumentParser(description="Find alias factor of a pulsar.")
parser.add_argument("-p", "--psr", type=str, help="Pulsar name.", default=None, required=True)
parser.add_argument("-n", "--ncpus", type=int, help="Number of pools.", default=1)
parser.add_argument("-o", "--pickle-output", type=str, help="Output directory of pickle for debug purpose.", default=None, required=False)
parser.add_argument("-N", "--n-files", type=int, help="Maximum number of archives to use.", default=None, required=False)
parser.add_argument("--parfile", type=str, help="Specify the path to timing model (default: ./timing_sources/<psr_id>/pulsar.par).", default=None, required=False)
parser.add_argument("--subints", type=str, help="Subint range to use for alias factor calculation (e.g., 20:128). Subint converted to data point index by [int(np.floor(subint_range[0] / bin_size)), int(np.ceil(subint_range[1] / bin_size))]", default=None, required=False)
parser.add_argument("--n-subints", type=int, help="Binsize for alias factor calculation.", default=8, required=False)
parser.add_argument("--n-bins", type=int, help="Number of bins for alias factor calculation.", default=64, required=False)
parser.add_argument("--smoothing", type=int, help="Smoothing factor for alias factor calculation.", default=0, required=False)
parser.add_argument("--mjd-range", type=str, help="MJD range to use for alias factor calculation (e.g., 59000:60000, inclusive).", default=None, required=False)
parser.add_argument("--backend", type=str, help="Use the data from the specified backend.", default=None, required=False)
parser.add_argument("--show-history", action="store_true", help="Show dealias history and skip processing.", required=False, default=False)
parser.add_argument("--no-commit", action="store_true", help="Do not commit changes to the psrdir and database.", required=False, default=False)
parser.add_argument("--no-beep", action="store_true", help="Do not beep when the process is finished.", required=False, default=False)
args = parser.parse_args()


##################################################
# Sanity checks                                  #
##################################################

# Subint range
subint_range = []
if args.subints is not None:
    subint_range = list(map(int, args.subints.split(":")))
    if len(subint_range) != 2:
        raise ValueError("Invalid subint range.")

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
    with database(f"{TIMING_SOURCES_PATH}/{args.psr}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
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
# More steps to repare processing                #
##################################################

# Print arguments
logger.info(f"Pulsar: {args.psr}")
logger.info(f"Number of pools: {args.ncpus}")
logger.info(f"Output directory: {args.pickle_output}")
logger.info(f"Maximum number of archives: {args.n_files}")
logger.info(f"Subint range: {subint_range}")
logger.info(f"Number of subints: {args.n_subints}")
logger.info(f"Smoothing: {args.smoothing}")
logger.info(f"MJD range: {mjd_range}")


##################################################
# Dealias                                        #
##################################################

# Process pulsars
dealias_results = []
logger.debug(f"Processing pulsar {args.psr}")

# Get psrdir
psrdir = f"./{TIMING_SOURCES_PATH}/{args.psr}"
logger.info(f"PsrDir: {psrdir}")
if not os.path.exists(psrdir):
    raise ValueError(f"Pulsar directory not found: {psrdir}")

# Check output directory
if args.pickle_output is not None:
    if not os.path.exists(args.pickle_output):
        raise ValueError(f"Output directory not found: {args.pickle_output}")

# Get parfile
if args.parfile is not None:
    parfile = args.parfile
else:
    parfile = f"./{TIMING_SOURCES_PATH}/{args.psr}/pulsar.par"

#  determine range of mjds
if mjd_range == [] or len(psrs) > 1:
    mjd_range = utils.read_start_end_from_parfile(parfile, raise_exception=False)
    logger.info(f"Auto-detected MJD range: {mjd_range}")

# Get the list of time to process
ar_list = []
backend_stats = {}
for f_info in mdb_hdl.get_raw_data_by_mjd_range(args.psr, mjd_range=mjd_range):
    if f_info["status"] != "good":
        logger.info(f"Skipping archive {f_info['location']} due to bad status.")
        continue

    # Append to backend stats
    if f_info["backend"] not in backend_stats:
        backend_stats[f_info["backend"]] = 0
    backend_stats[f_info["backend"]] += 1

    ar_list.append({
        "location": f_info["location"],
        "backend": f_info["backend"]
    })

# Determine backend to use
this_backend = None
if args.backend is None: 
    # Use the backend that has the most observations
    if len(backend_stats) == 0:
        raise ValueError(f"No good archives found for pulsar {args.psr} in the specified MJD range.")
    this_backend = max(backend_stats, key=backend_stats.get)
    logger.info(f"Auto-detected backend: {this_backend}")
else:
    this_backend = args.backend
    logger.info(f"Using user specified backend: {this_backend}")

# Select archives from the specified backend
ar_list = [ar for ar in ar_list if ar["backend"] == this_backend]
logger.info(f"Number of archives from backend {this_backend}: {len(ar_list)}", layer=1)

# Check if there are archives to process
if len(ar_list) == 0:
    raise ValueError(f"No archives found for pulsar {args.psr}")

# Check if parfile exists
if not os.path.exists(parfile):
    raise ValueError(f"Parfile not found for pulsar {args.psr}")

# Cut the list of archives if needed
if args.n_files is not None:
    # Shuffle the list of archives so that we get a random sample
    np.random.shuffle(ar_list)

    # Cut the list
    ar_list = ar_list[:args.n_files]

# Show archive information
logger.info(f"Number of archives: {len(ar_list)}")
for ar in ar_list:
    logger.info(f"{ar['location']} (backend: {ar['backend']})", layer=1)

# Find alias
with alias_utils(f"./{TIMING_SOURCES_PATH}/{args.psr}", ar_list, parfile, n_subints=args.n_subints, n_bins=args.n_bins, jumps=JUMPS, workspace=TEMPDIR, n_pools=args.ncpus, logger=logger.copy()) as au:
    # Get alias factor
    au.cf_get_alias_factor(subint_range=subint_range, smooth_sigma=args.smoothing)

    # Dealias
    au_summary = au.dealias()

    # Get dealias results
    dealias_results.append({
        "n_stacked": au_summary["n_stacked"],
        "alias_factor": au_summary["alias_factor"],
        "snr_stacked": au_summary["snr_stacked"],
        "remark": au_summary["notes"]["remark"],
    })

    # Commit changes
    if not args.no_commit:
        au.commit()

    if args.pickle_output is not None:
        # Save dealias information to user specified directory
        au.save_outfiles(args.pickle_output)
        logger.info(f"Dealias information saved to {args.pickle_output} (user specified)")


##################################################
# Finishing                                      #
##################################################

# print summary
dealias_results = pd.DataFrame(dealias_results)
dealias_results_txt = (
        "====================== DEALIAS RESULTS ======================" + "\n" +
        pd.DataFrame(dealias_results).to_string() + "\n" +
        "============================================================="
)
logger.info(dealias_results_txt)

# Send alert when finished
if not args.no_beep:
    logger.cli_alert()