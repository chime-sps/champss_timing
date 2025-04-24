import globus_sdk
import pickle
import requests
import time
import shutil
import glob
import subprocess
import argparse
import traceback
import os
from collections import deque
from globus_sdk.scopes import TransferScopes
from backend.datastores.tmg_master import tmg_master
from backend.utils.utils import utils
from backend.utils.notification import notification
from backend.utils.logger import logger
from cli.config import CLIConfig

###################################################
# Helper Functions                                #
###################################################

def _recursive_ls_helper(tc, ep, queue, max_depth, retries=5):
    while queue:
        abs_path, rel_path, depth = queue.pop()
        path_prefix = rel_path + "/" if rel_path else ""

        for i in range(retries):
            try:
                res = tc.operation_ls(ep, path=abs_path)
                time.sleep(2) # avoid rate limiting
                break
            except Exception as e:
                if i == retries - 1:
                    raise e
                print(f"Error while operation_ls: {e}")
                print(f"Retrying {i+1}/{retries}...")
                time.sleep(15)

        if depth < max_depth:
            queue.extend(
                (
                    res["path"] + item["name"],
                    path_prefix + item["name"],
                    depth + 1,
                )
                for item in res["DATA"]
                if item["type"] == "dir"
            )
        for item in res["DATA"]:
            item["name"] = path_prefix + item["name"]
            yield item

# tc: a TransferClient
# ep: an endpoint ID
# path: the path to list recursively
def recursive_ls(tc, ep, path, max_depth=3):
    queue = deque()
    queue.append((path, "", 0))
    yield from _recursive_ls_helper(tc, ep, queue, max_depth)

def do_submit(client):
    task_doc = client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    print(f"submitted transfer, task_id={task_id}")
    return task_id


# we will need to do the login flow potentially twice, so define it as a
# function
#
# we default to using the Transfer "all" scope, but it is settable here
# look at the ConsentRequired handler below for how this is used
def login_and_get_transfer_client(*, scopes=TransferScopes.all):
    auth_client.oauth2_start_flow(requested_scopes=scopes)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")

    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    # return the TransferClient object, as the result of doing a login
    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )

    
###################################################
# Initialize Variables                            #
###################################################

# parse arguments
parser = argparse.ArgumentParser(description='Transfer files using Globus')
parser.add_argument('--mute', action='store_true', help='Mute the slack message')
parser.add_argument('--refresh_token', action='store_true', help='Refresh the token')
parser.add_argument('--test_token', action='store_true', help='Test the token')
args = parser.parse_args()

cli_config = CLIConfig()
CLIENT_ID = cli_config.config["user_defined"]["globus"]["client_id"]
SLACK_TOKEN = cli_config.config["slack_token"]["chime"]
source_collection_id = cli_config.config["user_defined"]["globus"]["source_collection_id"]
dest_collection_id = cli_config.config["user_defined"]["globus"]["dest_collection_id"]
source_dir_fm = cli_config.config["user_defined"]["globus"]["source_dir_fm"]
source_dir_fil = cli_config.config["user_defined"]["globus"]["source_dir_fil"]
dest_dir = cli_config.config["user_defined"]["globus"]["dest_dir"]
transfer_scopes = cli_config.config["user_defined"]["globus"]["transfer_scopes"]
timing_sources_repo = cli_config.config["user_defined"]["globus"]["timing_sources_repo"]
heartbeat_url = cli_config.config["user_defined"]["heartbeat_url_cedar_narval"]


###################################################
# Initialize helper functions                     #
###################################################

if args.mute:
    SLACK_TOKEN = False

# Notification
slack = notification(SLACK_TOKEN)

# Logger
logger = logger()


###################################################
# Refresh token actions                           #
###################################################

if args.refresh_token:
    # Setup client
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

    # Authorize token
    client.oauth2_start_flow(refresh_tokens=True, requested_scopes=transfer_scopes)
    authorize_url = client.oauth2_get_authorize_url()
    logger.info(f"Please go to this URL and login:\n{authorize_url}")

    # Get auth code
    auth_code = input("Please enter the code here: ").strip()
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)

    # get credentials for the Globus Transfer service
    globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]

    # the refresh token and access token are often abbreviated as RT and AT
    transfer_rt = globus_transfer_data["refresh_token"]
    transfer_at = globus_transfer_data["access_token"]
    expires_at_s = globus_transfer_data["expires_at_seconds"]

    # construct a RefreshTokenAuthorizer
    # note that `client` is passed to it, to allow it to do the refreshes
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        transfer_rt, client, access_token=transfer_at, expires_at=expires_at_s
    )

    # and try using `tc` to make TransferClient calls. Everything should just
    # work -- for days and days, months and months, even years
    transfer_client = globus_sdk.TransferClient(authorizer=authorizer)

    # pickling the transfer_client
    with open('transfer_client.pickle', 'wb') as f:
        pickle.dump(transfer_client, f)

    exit() 


###################################################
# Check if Globus token is still valid            #
###################################################

logger.debug("Getting transfer_client...")

try:
    # Read transfer token pickle
    with open('transfer_client.pickle', 'rb') as f:
        transfer_client = pickle.load(f)
    
    # Try ls scratch space to see if it's still valid
    transfer_client.operation_ls(source_collection_id, path="/home/wenkexia/scratch")
    transfer_client.operation_ls(dest_collection_id, path="/home/wenkexia/scratch")

    logger.success("transfer_client loaded successfully from pickle")
except Exception as e:
    # Show the error message
    logger.error(traceback.format_exc())

    # Try to fetch required scope info
    if hasattr(e, "info"):
        if hasattr(e.info, "consent_required"): 
            logger.info("Required scope", e.info.consent_required.required_scopes)
    

    # Send slack notification
    slack.send_urgent_message(f"Globus token expired / not found. Please login to Globus to get a new token. ")
    exit()

if args.test_token:
    logger.success("Token is valid!")
    exit() # if only test token, exit


###################################################
# Prepare transfer task                           #
###################################################

logger.debug("Preparing transferring...")

# double check the source repo was cleanned up
shutil.rmtree("./champss_timing_sources", ignore_errors=True)

# clone a source repo
subprocess.run(["git", "clone", "--depth", "1", timing_sources_repo]) # have to clone before link file as link files reads masterdb


###################################################
# Find files needs to be transferred              #
###################################################

logger.debug("Searching files...")

pulsars = []
files = []
with tmg_master("./champss_timing_sources/TMGMaster.sqlite3.db", readonly=True) as tm:
    # Load pulsars to transfer
    pulsars = tm.get_psr_ids(table="timing") # load from master db
    for line in open("pulsars.txt", "r").read().split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        if line not in pulsars:
            pulsars.append(line.strip()) # check for additional pulsar data that needs to be transfered

    # Search data for each pulsar
    for i, pulsar in enumerate(pulsars):
        # Search files
        this_fm = glob.glob(source_dir_fm + f"/*{pulsar}*")
        this_fil = glob.glob(source_dir_fil + f"/*{pulsar}*")
        logger.debug("", f"[{i+1}/{len(pulsars)}]", "Searching files for", pulsar, f"(available: fold_mode -> {len(this_fm)}, filterbank -> {len(this_fil)})")

        # Cross-match fold mode
        for f in this_fm:
            if os.path.isfile(f):
                if len(tm.get_raw_data(psr_id=pulsar, ar_id=utils.get_archive_id(f), backend="chimepsr_fm")) == 0:
                    files.append({
                        "src": f, 
                        "dest": dest_dir + "/fold_mode/" + pulsar + "/" + f.split("/")[-1]
                    })
        
        # Cross-match filterbank
        for f in this_fil:
            if os.path.isfile(f):
                if len(tm.get_raw_data(psr_id=pulsar, ar_id=utils.get_archive_id(f), backend="chimepsr_fil")) == 0:
                    files.append({
                        "src": f, 
                        "dest": dest_dir + "/filterbank/" + pulsar + "/" + f.split("/")[-1]
                    })

###################################################
# Initial transfer task                           #
###################################################

logger.debug("Transferring files...")

# Check if there are files to be transferred
if len(files) == 0:
    logger.success("No files to transfer. ")
    slack.send_success_message(f"No files to transfer. No Globus task initiated.")
    exit()

# Create transfering task
task_data = globus_sdk.TransferData(
    source_endpoint=source_collection_id, destination_endpoint=dest_collection_id
)

# Add items from files
for f in files:
    task_data.add_item(
        f["src"],
        f["dest"]
    )

# Submit the transfer
try:
    task_id = do_submit(transfer_client)
except Exception as err:
    slack.send_urgent_message(f"Failed to initiate Globus transfer task. Please check if the token is still valid.  (`{err}`)")
    exit()

###################################################
# Finishing the script                            #
###################################################

# Send success message
slack.send_success_message(f"Globus transfer task initiated ({len(files)} files). Please check the status in the [Globus portal](https://app.globus.org/activity/{task_id}/overview).")

# send heartbeat
logger.debug("Sending heartbeat...")
requests.get(heartbeat_url)

# Cleanup source repo
logger.debug("Cleanning up...")
shutil.rmtree("./champss_timing_sources")

logger.success("Transfer script finished...")
