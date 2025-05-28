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
TEMPDIR = "/tmp/__alias_utils_workspaces"
JUMPS = cli_config.get_config()["toa_jumps"]
t_start = time.time()
mdb_hdl = tmg_master(MASTER_DB_PATH)

# Initialize parser
parser = argparse.ArgumentParser(description="Find alias factor of a pulsar.")
parser.add_argument("-p", "--psr", type=str, help="Pulsar name.", default=None, required=False)
parser.add_argument("-n", "--ncpus", type=int, help="Number of pools.", default=1)
parser.add_argument("-o", "--pickle-output", type=str, help="Output directory of pickle for debug purpose.", default=None, required=False)
parser.add_argument("-N", "--n-files", type=int, help="Maximum number of archives to use.", default=None, required=False)
parser.add_argument("--subints", type=str, help="Subint range to use for alias factor calculation (e.g., 20:128). Subint converted to data point index by [int(np.floor(subint_range[0] / bin_size)), int(np.ceil(subint_range[1] / bin_size))]", default=None, required=False)
parser.add_argument("--n-subints", type=int, help="Binsize for alias factor calculation.", default=16, required=False)
parser.add_argument("--smoothing", type=int, help="Smoothing factor for alias factor calculation.", default=1, required=False)
parser.add_argument("--max-n-psrs", type=int, help="Maximum number of pulsars to process.", default=None, required=False)
parser.add_argument("--max-execution-time", type=int, help="Maximum execution time in seconds. If the execution time exceeds this value, the process will be terminated after the current pulsar is processed.", default=None, required=False)
parser.add_argument("--mjd-range", type=str, help="MJD range to use for alias factor calculation (e.g., 59000:60000, inclusive).", default=None, required=False)
parser.add_argument("--show-dealias-history", action="store_true", help="Show dealias history and skip processing.", required=False, default=False)
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
    if args.psr is None:
        raise ValueError("MJD range can only be used when a single pulsar is specified.")

# Check if needs to read pulsars from db
if args.psr is None:
    psrs = mdb_hdl.get_psr_ids(table="timing")
else:
    psrs = [args.psr]


##################################################
# Calculate weights                              #
##################################################

# Get weights of processing pulsars
logger.debug("Calculating weights for probabilistic selection...")
weights = np.ones(len(psrs))
dealias_histories = []
for i, psr in enumerate(psrs):
    with database(f"{TIMING_SOURCES_PATH}/{psr}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
        last_timing_info = db_hdl.get_last_timing_info()
        last_dealias_history = db_hdl.get_last_dealias_history()
        all_dealias_histories = db_hdl.get_all_dealias_history()

    last_dealias_history["psr_id"] = psr
    last_dealias_history["weight"] = weights[i]

    # A bug in early version of the dealias_utils caused some numbers to be saved as bytes
    if type(last_dealias_history["alias_factor"]) == bytes:
        import struct
        last_dealias_history["alias_factor"] = struct.unpack('d', last_dealias_history["alias_factor"])[0]
    if type(last_dealias_history["snr_stacked"]) == bytes:
        import struct
        last_dealias_history["snr_stacked"] = struct.unpack('f', last_dealias_history["snr_stacked"])[0]
        
    dealias_histories.append(last_dealias_history)

    if last_timing_info["timestamp"] == 0:
        weights[i] = -1
        continue
    
    if last_dealias_history["alias_factor"] == 0:
        weights[i] += (time.time() - last_dealias_history["timestamp"]) / 90
        if (time.time() - last_dealias_history["timestamp"]) < 30 * 24 * 3600:
            weights[i] = -1 # Skip pulsars with alias factor calculated within 30 days
            continue
    else:
        weights[i] += (time.time() - last_dealias_history["timestamp"]) / 30
        if (time.time() - last_dealias_history["timestamp"]) < 7 * 24 * 3600:
            weights[i] = -1 # Skip pulsars with alias factor calculated within 7 days
            continue
    
    if last_dealias_history["timestamp"] == 0:
        weights[i] += 999
    else:
        weights[i] /= len(all_dealias_histories)

    if len(last_timing_info["fitted_params"]) < 4:
        weights[i] /= 10 # ≈ pulsars with less than 4 fitted parameters

    if last_timing_info["chi2_reduced"] > 100:
        weights[i] /= last_timing_info["chi2_reduced"] # ç high chi2 pulsars
    
    if len(last_timing_info["residuals"]["val"]) < 30:
        weights[i] = -1 # Skip pulsars with less than 60 toas
    else:
        weights[i] *= len(last_timing_info["residuals"]["val"]) * 0.1 # Skip pulsars with less than 60 toas

    dealias_histories[-1]["weight"] = weights[i]

# Normalize weights
# weights = weights / np.abs(np.max(weights))
for i, weight in enumerate(weights):
    dealias_histories[i]["weight"] = weight


##################################################
# Show dealias history                           #
##################################################

# Print dealias histories
if args.show_dealias_history:
    dealias_histories_formatted = {"PSR": [], "Last Updated": [], "AF": [], "N_stacked": [], "SNR_stacked": [], "Remark": [], "Weight": []}
    for dealias_history in dealias_histories:
        dealias_histories_formatted["PSR"].append(dealias_history["psr_id"])
        dealias_histories_formatted["Last Updated"].append(datetime.datetime.fromtimestamp(dealias_history["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'))
        dealias_histories_formatted["AF"].append(dealias_history["alias_factor"])
        dealias_histories_formatted["N_stacked"].append(dealias_history["n_stacked"])
        dealias_histories_formatted["SNR_stacked"].append(dealias_history["snr_stacked"])
        dealias_histories_formatted["Remark"].append(dealias_history["notes"]["remark"])
        dealias_histories_formatted["Weight"].append(dealias_history["weight"])
    dealias_histories_formatted = {k: [v for _, v in sorted(zip(dealias_histories_formatted["PSR"], v))] for k, v in dealias_histories_formatted.items()}
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
logger.info(f"Pulsar(s): {psrs}")
logger.info(f"Number of pools: {args.ncpus}")
logger.info(f"Output directory: {args.pickle_output}")
logger.info(f"Maximum number of archives: {args.n_files}")
logger.info(f"Subint range: {subint_range}")
logger.info(f"Number of subints: {args.n_subints}")
logger.info(f"Smoothing: {args.smoothing}")
logger.info(f"Maximum number of pulsars to process: {args.max_n_psrs}")
logger.info(f"MJD range: {mjd_range}")

# Sort pulsars by weights
if len(psrs) > 1:
    indices = np.argsort(weights)[::-1]
    indices = indices[weights[indices] > 0] # Skip pulsars with negative weights
    psrs = [psrs[i] for i in indices]
    weights = [weights[i] for i in indices]
if args.max_n_psrs is not None:
    psrs = psrs[:args.max_n_psrs]
    weights = weights[:args.max_n_psrs]

# Print pulsars to process
if len(psrs) > 1:
    logger.info(f"Processing plan: ", end="")
    processing_plan_text = ""
    for i, psr in enumerate(psrs):
        processing_plan_text += f"{psr} [{weights[i]:.2f}]" + " -> "
        if i == len(psrs) - 1:
            processing_plan_text = processing_plan_text[:-4]
    logger.info(processing_plan_text)


##################################################
# Dealias                                        #
##################################################

# Process pulsars
dealias_results = []
for i, psr in enumerate(psrs): 
    logger.debug(f"Processing pulsar {psr} ({i+1}/{len(psrs)}) [{weights[i]:.2f}]")

    try:
        # Get psrdir
        psrdir = f"./{TIMING_SOURCES_PATH}/{psr}"
        logger.info(f"PsrDir: {psrdir}")
        if not os.path.exists(psrdir):
            raise ValueError(f"Pulsar directory not found: {psrdir}")

        # Check output directory
        if args.pickle_output is not None:
            if not os.path.exists(args.pickle_output):
                raise ValueError(f"Output directory not found: {args.pickle_output}")

        # Get parfile
        parfile = f"./{TIMING_SOURCES_PATH}/{psr}/pulsar.par"

        #  determine range of mjds
        if mjd_range == [] or len(psrs) > 1:
            mjd_range = utils.read_start_end_from_parfile(parfile, raise_exception=False)
            logger.info(f"Auto-detected MJD range: {mjd_range}")

        # Get the list of time to process
        ar_list = []
        for f_info in mdb_hdl.get_raw_data_by_mjd_range(psr, mjd_range=mjd_range):
            if f_info["status"] != "good":
                continue

            ar_list.append({
                "location": f_info["location"],
                "backend": f_info["backend"]
            })

        # Check if there are archives to process
        if len(ar_list) == 0:
            raise ValueError(f"No archives found for pulsar {psr}")

        # Check if parfile exists
        if not os.path.exists(parfile):
            raise ValueError(f"Parfile not found for pulsar {psr}")

        # Cut the list of archives if needed
        if args.n_files is not None:
            # Shuffle the list of archives so that we get a random sample
            np.random.shuffle(ar_list)

            # Cut the list
            ar_list = ar_list[:args.n_files]

        # Show archive count information
        logger.info(f"Number of archives: {len(ar_list)}")

        # Find alias
        with alias_utils(psrdir, ar_list, parfile, n_subints=args.n_subints, jumps=JUMPS, workspace=TEMPDIR, n_pools=args.ncpus, logger=logger.copy()) as au:
            # Get alias factor
            au.cf_get_alias_factor(subint_range=subint_range, smooth_sigma=args.smoothing)

            # Dealias
            au_summary = au.dealias()

            # Get dealias results
            dealias_results.append({
                "psr_id": au_summary["psr_id"],
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
                
    except Exception as e:
        logger.error(f"Error while processing pulsar {psr}: {str(e)}")
        logger.error(traceback.format_exc())
        dealias_results.append({
            "psr_id": psr,
            "n_stacked": "NaN",
            "alias_factor": "NaN",
            "snr_stacked": "NaN",
            "dealias_res": "Error"
        })

    if args.max_execution_time is not None:
        if time.time() - t_start > args.max_execution_time:
            logger.warning(f"Maximum execution time reached. Terminating process.")
            break


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