import os
import glob
import time
import datetime
import traceback
import pandas as pd
import tqdm
import argparse
from multiprocessing import Pool
from backend import champss_timing
from backend.utils import notification, logger, utils
from backend.datastores import database, tmg_master
from cli.config import CLIConfig

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # To avoid SSL error on Narval

# Load configuration
cli_config = CLIConfig(load_error=False)

# Parse arg
parser = argparse.ArgumentParser(description="CHAMPSS Timing Main Pipeline.")
parser.add_argument("--ncpus", type=int, help="Number of CPUs to use.")
parser.add_argument("--psr", type=str, help="Pulsar name (run timing for all pulsars if not specified).")
parser.add_argument("--slack-token", type=str, help="Slack token.")
args = parser.parse_args()

print(f"Number of CPUs: {args.ncpus}")
print(f"Pulsar: {args.psr}")
print(f"Slack token: {args.slack_token}")
print(f"Start timing... (press Ctrl+C to cancel)")
time.sleep(3)

# Define
TIMING_SOURCES_PATH = "./timing_sources"
MASTER_DB_PATH = TIMING_SOURCES_PATH + "/TMGMaster.sqlite3.db"
CONFIG_FILENAME = "champss_timing.config"
MODEL_FILENAME = "pulsar.par"
TEMPLATE_FILENAME = "paas.std"
BACKENDS = cli_config.config["backends"]
N_POOL = args.ncpus

# Slack tokens
if args.slack_token is not None:
    if args.slack_token in cli_config.config["slack_token"]:
        print(f"Using {args.slack_token} slack token.")
        SLACK_TOKEN = cli_config.config["slack_token"][args.slack_token]
        SLACK_TOKEN = {
            "CHANNEL_ID": cli_config.config["slack_token"][args.slack_token]["CHANNEL_ID"],
            "SLACK_BOT_TOKEN": cli_config.config["slack_token"][args.slack_token]["SLACK_BOT_TOKEN"],
            "SLACK_APP_TOKEN": cli_config.config["slack_token"][args.slack_token]["SLACK_APP_TOKEN"]
        }
    else:
        raise ValueError("Known slack token name.")
else:
    print("Disabling slack notification.")
    SLACK_TOKEN = False

# Format jumps and labels
JUMPS = {}
LABELS = {}
for bknd in BACKENDS:
    JUMPS[bknd] = BACKENDS[bknd]["jump"]
    LABELS[bknd] = BACKENDS[bknd]["label"]

# Initialize hamdlers
logger = logger.logger()
noti = notification.notification(SLACK_TOKEN)

# Fetch master db
with tmg_master.tmg_master(MASTER_DB_PATH, readonly=True) as tm_hdl:
    # Load pulsars
    pulsars = []
    if args.psr == None:
        dirs = glob.glob(f"{TIMING_SOURCES_PATH}/*")
        for d in dirs:
            if os.path.isdir(d):
                pulsars.append(d.split("/")[-1])
    else:
        pulsars = [args.psr]

    # Get pulsar_data
    pulsar_data = []
    for pulsar in pulsars:
        # Check if model exists
        model_path = f"{TIMING_SOURCES_PATH}/{pulsar}/{MODEL_FILENAME}"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            continue

        # Check if template exists
        template_path = f"{TIMING_SOURCES_PATH}/{pulsar}/{TEMPLATE_FILENAME}"
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            continue

        # Query master db
        filenames, counts = tm_hdl.get_timing_data_config(pulsar, backends=list(BACKENDS.keys()), labels=LABELS)

        # Create pulsar data
        this_pulsar_data = {
            "id": pulsar,
            "dir": f"{TIMING_SOURCES_PATH}/{pulsar}", 
            "log": f"{TIMING_SOURCES_PATH}/{pulsar}/script_log.txt",
            "config": f"{TIMING_SOURCES_PATH}/{pulsar}/{CONFIG_FILENAME}",
            "model": f"{TIMING_SOURCES_PATH}/{pulsar}/{MODEL_FILENAME}",
            "template": f"{TIMING_SOURCES_PATH}/{pulsar}/{TEMPLATE_FILENAME}",
            "data": filenames, 
            "counts": counts
        }

        # Check if there are files to process
        if this_pulsar_data["counts"]["total"] == 0:
            logger.error(f"No data to process for {pulsar}")
            continue

        pulsar_data.append(this_pulsar_data)

# Start timing
timing_results = []
for d in pulsar_data:
    logger.debug(f"Timing {d['id']}")
    logger.debug(f"> PsrDir: {d['dir']}")
    logger.debug(f"> Model: {d['model']}")
    logger.debug(f"> Template: {d['template']}")
    logger.debug(f"> Data: ")
    for c in d["counts"]:
        logger.debug(f"> {c}: {d['counts'][c]}", layer=1)
    timing_results.append({"psr": d['id'], "n_files": d['counts']["total"], "success": True, "n_timed": 0})

    try:
        # Start timing
        logger.debug(f"Starting pipeline for {d['id']}")
        with champss_timing.champss_timing(
                psr_dir=d["dir"],
                data_archives=d["data"],
                toa_jumps=JUMPS, 
                logger=logger.copy(),
                slack_token=SLACK_TOKEN,
                n_pools=N_POOL
        ) as ctim:
            timing_results[-1]["n_timed"] = ctim.run()["n_timed"]
        
    except Exception as e:
        logger.error(f"Error while timing {d['id']}: {e}")
        logger.error(traceback.format_exc())
        timing_results[-1]["success"] = False
    
    # Update master db
    try:
        with tmg_master.tmg_master(MASTER_DB_PATH) as tm_hdl:
            tm_hdl.update_timing(
                psr_id=d["id"], 
                timing_dir=os.path.abspath(d["dir"]),
                last_updated=time.time(), 
                last_status="ok" if timing_results[-1]["success"] else "error", 
                notes={}, 
                create_if_not_exists=True
            )
        logger.success("Status synced with master db.")
    except Exception as e:
        logger.error(f"Error while syncing status with master db: {e}")
        logger.error(traceback.format_exc())

# Generate timing results text
timing_results = pd.DataFrame(timing_results)
timing_results_txt = (
        "============== TIMING RESULTS ==============" + "\n" +
        pd.DataFrame(timing_results).to_string() + "\n" +
        "============================================"
)

# Save summary
with open(f"{TIMING_SOURCES_PATH}/timing_summary.txt", "w") as f:
    f.write(timing_results_txt)