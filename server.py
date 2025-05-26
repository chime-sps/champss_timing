import os
import shutil
import threading
import time
import argparse

from web import server
from cli.config import CLIConfig

try:
    import git
except ImportError:
    import subprocess
    git = None
    print("gitpython is not installed. Using subprocess instead.")

# Load configuration
cli_config = CLIConfig(load_error=False)

# Initialize parser
parser = argparse.ArgumentParser(description="CHAMPSS Timing Pipeline Web Server")
parser.add_argument("-p", "--port", type=int, default=1508, help="Port number for the web server (default: 1508)")
parser.add_argument("-h", "--host", type=str, default="127.0.0.1", help="Host address for the web server (default: 127.0.0.1)")
parser.add_argument("-r", "--repo", type=str, default=None, help="Repository URL for the timing sources", required=False)
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode", default=False)
parser.add_argument("-k", "--ssh-key", type=str, help="SSH key for the repository", default="")
parser.add_argument("--slack", type=str, help="Slack token for the run notes service", default=None)
parser.add_argument("--password", type=str, help="Password for the repository", default="")
args = parser.parse_args()

# Initialize parameters
repo_url = args.repo
ssh_key = args.ssh_key
psr_dir = os.path.abspath(f"./timing_sources")
password = args.password if args.password else False

# Start the web server
def update_repo():
    '''
    Update the repository by removing the old directory and cloning a new one.
    '''

    if repo_url is None:
        if not os.path.exists(psr_dir):
            raise FileNotFoundError(f"Repository URL is not provided and the directory {psr_dir} does not exist.")
        print("No repository URL provided. Using existing directory.")
        return

    # Check if the directory exists
    if os.path.exists(psr_dir):
        os.system(f"rm -rf " + os.path.abspath("./timing_sources"))
        print("Remove the old directory: %s" % psr_dir)

    # Construct options
    multi_options = ["--depth", "1", "--single-branch", "--branch", "main"]
    if ssh_key != "":
        multi_options.append(f"--config core.sshCommand='ssh -i {ssh_key}'")

    # Clone the repository
    if git is not None:
        git.Repo.clone_from(repo_url, psr_dir, multi_options=multi_options, allow_unsafe_options=True)
        print("Clone the new directory: %s (gitpython)" % psr_dir)
    else:
        # Use subprocess to clone the repository
        cmd = ["git", "clone"] + multi_options + [repo_url, psr_dir]
        subprocess.run(cmd, check=True)
        print("Clone the new directory: %s (subprocess)" % psr_dir)

# Parse slack token
slack_token = None
if args.slack is not None:
    if args.slack in cli_config.config["slack_token"]:
        print(f"Using {args.slack} slack token.")
        slack_token = {
            "CHANNEL_ID": cli_config.config["slack_token"][args.slack]["CHANNEL_ID"],
            "SLACK_BOT_TOKEN": cli_config.config["slack_token"][args.slack]["SLACK_BOT_TOKEN"],
            "SLACK_APP_TOKEN": cli_config.config["slack_token"][args.slack]["SLACK_APP_TOKEN"]
        }
    else:
        raise ValueError("Known slack token name.")

# Start the web server
server.run(
    psr_dir=psr_dir, 
    update_hdl=update_repo, 
    host=args.host,
    port=args.port, 
    debug=args.debug,
    password=password, 
    slack_token=slack_token
)