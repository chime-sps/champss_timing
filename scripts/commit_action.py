###############################################################
# This is the script that will be run by after a commit event #
###############################################################

import os
import time
import glob
import datetime
import requests
from backend.pipecore import checker
from backend.utils.notification import notification
from backend.datastores.database import database
from cli.config import CLIConfig

# Load configuration
cli_config = CLIConfig()

# Define
TIMING_SOURCES = "./champss_timing_sources"
SUMMARY_TEXT = "./champss_timing_sources/timing_summary.txt"
MONITOR_URL = cli_config.config["user_defined"]["monitor_url"]
MONITOR_PASSWORD = cli_config.config["user_defined"]["monitor_password"]
UPTIME_URL = cli_config.config["user_defined"]["uptime_url"]
SLACK_TOKEN = cli_config.config["slack_token"]["chime"]


# Initialize the notification handler
noti = notification(SLACK_TOKEN)

# Trigger the monitor to update the database
try:
    requests.get(MONITOR_URL + "/public/api/update_psrdir")
except Exception as e:
    noti.send_urgent_message("Error while updating the timing monitor: " + str(e))

# Push notification to slack
noti.send_success_message("Timing sources committed to champss_timing_sources repo at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Run checker
all_checkers_passed = True
for source in glob.glob(os.path.join(TIMING_SOURCES, "*", "*.db")):
    try:
        with database(source) as db_hdl:
            if db_hdl.get_last_timing_info()["timestamp"] > time.time() - 43200:
                checker_res = checker(
                    psr_dir=os.path.dirname(source),
                    db_hdl=db_hdl,
                    noti_hdl=noti, 
                    psr_id=source.split("/")[-2]
                ).check()
                
                # check if all checkers are passed
                for checker_module in checker_res.keys():
                    for checker_key in checker_res[checker_module].keys():
                        if checker_res[checker_module][checker_key]["level"] > 0:
                            all_checkers_passed = False
    except Exception as e:
        noti.send_urgent_message(f"Error while checking {source}: {str(e)}")
        all_checkers_passed = False

# Send summary
if os.path.exists(SUMMARY_TEXT):
    with open(SUMMARY_TEXT, "r") as f:
        summary_content = f.read()
        if "False" in summary_content:
            noti.send_urgent_message("Not all pipeline processing finished successfully. Please check the following pipeline summary and processing log on Narval.")
        else:
            if all_checkers_passed:
                noti.send_success_message("The timing pipeline is healthy with no exceptions found in the last 24 hours of timing solutions.")
        noti.send_code(summary_content)
else:
    noti.send_message("No summary text file found.")

# End text
noti.send_message(f"For timing results, please refer to <{MONITOR_URL}|CHAMPSS Timing Monitor> (password: `{MONITOR_PASSWORD}`) or processing log on Narval. For pipeline status, please refer to <{UPTIME_URL}|CHAMPSS Timing Status Page>")
