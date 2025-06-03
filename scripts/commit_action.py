###############################################################
# This is the script that will be run by after a commit event #
###############################################################

import os
import time
import glob
import datetime
import requests
from champss_timing.backend.tools import monitoring
from backend.utils.notification import notification
from backend.datastores.database import database
from cli.config import CLIConfig

def parse_timing_results(text):
    timing_results = {}

    for line in text.strip().split('\n'):
        if "===" in line or not line.strip():
            continue

        if timing_results == {}:
            timing_results["id"] = []
            for header in line.split():
                timing_results[header] = []
            continue

        parts = line.split()
        if len(parts) != len(timing_results):
            raise ValueError(f"Line does not match header length: {line}")
        
        for i, header in enumerate(timing_results):
            timing_results[header].append(parts[i])

    return timing_results

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
psrdirs = glob.glob(os.path.join(TIMING_SOURCES, "*"))
mg = monitoring.Monitoring(noti_hdl=noti, verbose=True)
for psrdir in psrdirs:
    if not os.path.exists(os.path.join(psrdir, "champss_timing.sqlite3.db")):
        continue
    mg.add_psrdir(psrdir)
all_checkers_passed = mg.run_checkers(within_24h=True)

# Send summary
if os.path.exists(SUMMARY_TEXT):
    with open(SUMMARY_TEXT, "r") as f:
        summary_content = f.read()

        # if "False" in summary_content:
        #     noti.send_urgent_message("Not all pipeline processing finished successfully. Please check the following pipeline summary and processing log on Narval.")
        # else:
        #     if all_checkers_passed:
        #         noti.send_success_message("The timing pipeline is healthy with no exceptions found in the last 24 hours of timing solutions.")

        # Parse timing results
        try:
            timing_results = parse_timing_results(summary_content)
        except ValueError as e:
            noti.send_urgent_message("Error parsing timing summary. Please check the summary text file.")
            timing_results = {}

        # Check if all processing passed
        all_processing_passed = True
        try:
            for t in timing_results["success"]:
                if t.lower() == "false":
                    all_processing_passed = False
                    break
        except KeyError:
            all_processing_passed = False

        # Get number of files processed
        n_files_processed = 0
        try:
            for t in timing_results["n_timed"]:
                n_files_processed += int(t)
        except KeyError:
            n_files_processed = 0

        # Send message
        if not all_processing_passed:
            noti.send_urgent_message("Not all pipeline processing finished successfully. Please check the following pipeline summary and processing log on Narval.")
        elif n_files_processed == 0:
            noti.send_urgent_message("No files processed in the last 24 hours. If this is unexpected, please check the timing pipeline status.")
        else:
            if all_checkers_passed:
                noti.send_success_message("The timing pipeline is healthy with no exceptions found in the last 24 hours of timing solutions.")
            else:
                noti.send_message("Checkers found some issues in timing solutions. Please check the warning messages above.")

        noti.send_code(summary_content)
else:
    noti.send_message("No summary text file found.")

# End text
noti.send_message(f"For timing results, please refer to <{MONITOR_URL}|CHAMPSS Timing Monitor> (password: `{MONITOR_PASSWORD}`) or processing log on Narval. For pipeline status, please refer to <{UPTIME_URL}|CHAMPSS Timing Status Page>")
