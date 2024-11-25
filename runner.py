import os
import glob
import datetime
import traceback
import pandas as pd
from . import champss_timing, archive_utils, notification, logger

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # To avoid SSL error on Narva

# Define
TIMING_SOURCES_PATH = "./timing_sources"
TIMING_DATA_PATH_PULSAR = "/home/wenkexia/projects/ctb-vkaspi/champss/fold_mode/pulsar/"
TIMING_DATA_PATH_CHAMPSS = "/home/wenkexia/projects/ctb-vkaspi/champss/fold_mode/champss/"
DATA_FILENAME_PULSAR = "*%PSR_ID%*.ar"
DATA_FILENAME_CHAMPSS = "%PSR_ID%/*.ar"
MODEL_FILENAME = "pulsar.par"
TEMPLATE_FILENAME = "paas.std"
N_POOL = 1

# SLACK_TOKEN = {
#     "CHANNEL_ID": "C080AKT7GEM",
#     "SLACK_BOT_TOKEN": "xoxb-194910630096-8000595811924-QU9jYymnW4dLCg68Ckl9qOq4",
#     "SLACK_APP_TOKEN": "xapp-1-A080LPRPBPA-7991431433846-018a5948f84f0d548a71c59e51602127622dac3b16209a95c82dbaa3a3791bd8"
# }# chime channel

SLACK_TOKEN = {
    "CHANNEL_ID": "C07RU3FQ0CW",
    "SLACK_BOT_TOKEN": "xoxb-7884645340771-7881932432293-i09Q8IckBmGaxQDhed7HXyU7",
    "SLACK_APP_TOKEN": "xapp-1-A07RU3YCM8E-7897477213873-e4ae988142ed9d73aaa452bfb5b899e058aeb1b09b1348da383adc33ae3ae9fb"
}  # test channel

# Initialize hamdlers
logger = logger.logger()
noti = notification.notification(SLACK_TOKEN)

# Push start notification
logger.debug("Timing started. ")
noti.send_message(f"ℹ️ TIMING STARTED AT {datetime.datetime.now()}")

# Load pulsars
pulsars = []
for this_psr_dir in glob.glob(TIMING_SOURCES_PATH + "/*"):
    this_psr_config = {
        "id": this_psr_dir.split("/")[-1],
        "dir": this_psr_dir,
        "log": f"{this_psr_dir}/script_log.txt",
        "model": f"{this_psr_dir}/{MODEL_FILENAME}",
        "template": f"{this_psr_dir}/{TEMPLATE_FILENAME}",
        "data_pulsar": glob.glob(
            f"{TIMING_DATA_PATH_PULSAR}/{DATA_FILENAME_PULSAR.replace('%PSR_ID%', this_psr_dir.split('/')[-1])}"
        ),
        "data_champss": glob.glob(
            f"{TIMING_DATA_PATH_CHAMPSS}/{DATA_FILENAME_CHAMPSS.replace('%PSR_ID%', this_psr_dir.split('/')[-1])}"
        )
    }

    # Check if file exists
    if not os.path.exists(this_psr_config["model"]):
        logger.debug(f"> Model file for {this_psr_config['id']} does not exist. Skipping.")
        continue

    if not os.path.exists(this_psr_config["template"]):
        logger.debug(f"> Template file for {this_psr_config['id']} does not exist. Skipping.")
        continue

    if len(this_psr_config["data"]) + len(this_psr_config["data_champss"]) == 0:
        logger.debug(f"> No observation for {this_psr_config['id']} found. Skipping.")
        continue

    pulsars.append(this_psr_config)
    logger.success(f"> {this_psr_config['id']} loaded.")

logger.info(f"{len(pulsars)} pulsars loaded.")

# Time pulsars
timing_results = []
for pulsar in pulsars:
    # if(pulsar['id'] not in ["J2318+4912", "J2100+4718"]):
    # if(pulsar['id'] not in ["J2318+4912"]):
    # if(pulsar['id'] not in ["J2100+4718"]):B0559-05
    # if(pulsar['id'] not in ["B1508+55"]):
    #     continue

    logger.debug(f"Timing {pulsar['id']}")
    logger.debug(f"> PsrDir: {pulsar['dir']}")
    logger.debug(f"> Model: {pulsar['model']}")
    logger.debug(f"> Template: {pulsar['template']}")
    logger.debug(f"> Data: {len(pulsar['data_pulsar'])}", "+", f"{len(pulsar['data_champss'])}")
    timing_results.append({"psr": pulsar['id'], "n_files": len(pulsar['data_pulsar']), "success": True, "n_timed": 0})

    try:
        psr_dir = pulsar['dir']
        archives_aval_pulsar = pulsar['data_pulsar']
        archives_aval_champss = pulsar['data_champss']

        archives = {}
        for ar in archives_aval_pulsar:
            # Get MJD
            this_mjd = round(archive_utils.archive_utils(ar).get_mjd())

            # Check if MJD exists
            if this_mjd not in archives.keys():
                archives[this_mjd] = []

            # Append to archives
            archives[this_mjd].append({
                "path": ar,
                "jump": 0
            })

            logger.data(this_mjd, ar)

        for ar in archives_aval_champss:
            # Get MJD
            this_mjd = round(archive_utils.archive_utils(ar).get_mjd())

            # Check if MJD exists
            if this_mjd not in archives.keys():
                archives[this_mjd] = []

            # Append to archives
            archives[this_mjd].append({
                "path": ar,
                "jump": 0
            })

            logger.data(this_mjd, ar)

        with champss_timing.champss_timing(
                psr_dir=psr_dir,
                data_archives=archives,
                logger=logger.copy(),
                slack_token=SLACK_TOKEN,
                n_pools=N_POOL
        ) as ctim:
            timing_results[-1]["n_timed"] = ctim.run()["n_timed"]
    except Exception as e:
        logger.error(f"Error timing {pulsar['id']}: {e}")
        logger.error(traceback.format_exc())
        noti.send_urgent_message(f"A fetal error occurred while timing for {pulsar['id']}. Script exited.")
        noti.send_code(traceback.format_exc())
        timing_results[-1]["success"] = False
        # break

timing_results = pd.DataFrame(timing_results)
timing_results_txt = (
        "============== TIMING RESULTS ==============" + "\n" +
        pd.DataFrame(timing_results).to_string() + "\n" +
        "============================================"
)

# Push end notification
logger.success("Timing ended. ")
noti.send_message(f"✅ TIMING ENDED AT {datetime.datetime.now()}")

# Push summary notification
logger.info(timing_results_txt)
noti.send_code(timing_results_txt)
