import os
import shutil
import git
import threading
import time
from web import champss_monitor

# Initialize parser
parser = argparse.ArgumentParser(description="CHAMPSS Timing Pipeline Web Server")
parser.add_argument("-p", "--port", type=int, default=1508, help="Port number for the web server (default: 1508)")
parser.add_argument("-r", "--repo", type=str, help="Repository URL for the timing sources", required=True)
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode", default=False)
parser.add_argument("-k", "--ssh-key", type=str, help="SSH key for the repository", default="")
parser.add_argument("--password", type=str, help="Password for the repository", default="")
args = parser.parse_args()

# Initialize parameters
repo_url = args.repo
ssh_key = args.ssh_key
psr_dir = os.path.abspath(f"./timing_sources_{time.time()}")
password = args.password if args.password else False

# Start the web server
def update_repo():
    '''
    Update the repository by removing the old directory and cloning a new one.
    '''

    # Check if the directory exists
    if os.path.exists(psr_dir):
        os.system(f"rm -rf " + os.path.abspath("./timing_sources_*"))
        print("Remove the old directory: %s" % psr_dir)

    # Construct options
    multi_options = []
    if ssh_key != "":
        multi_options.append(f"--config core.sshCommand='ssh -i {ssh_key}'")

    # Clone the repository
    git.Repo.clone_from(repo_url, psr_dir, multi_options=multi_options, allow_unsafe_options=True)
    print("Clone the new directory: %s" % psr_dir)

champss_monitor.run(
    psr_dir=psr_dir, 
    update_hdl=update_repo, 
    port=args.port, 
    debug=args.debug,
    password=password
)