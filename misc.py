import argparse
import glob
import os
import time
import importlib

# Initialize parser
parser = argparse.ArgumentParser(description="Run a miscellaneous scripts.")
parser.add_argument(
    "-ls", 
    "--list",
    action="store_true",
    dest="list",
    help="List all available scripts.",
)
parser.add_argument(
    "-r", 
    "--run",
    type=str,
    dest="script",
    help="Run a specific script.",
)

# Parse arguments
args = parser.parse_args()

# List all available scripts
script_paths = glob.glob(os.path.dirname(__file__) + "/scripts/*.py")
script_ids = ["".join(os.path.basename(script_path)[:-3]) for script_path in script_paths]
if args.list:
    print("Available scripts:")
    for script_id in script_ids:
        print(f"  [{script_id}]")
    exit()

# If no script is specified, show help
if args.script is None:
    print("No script specified. Use -ls to list all available scripts and -s to run a specific script.")
    exit()

# Check if the script exists
if args.script not in script_ids:
    print(f"Script {args.script} does not exist.")
    exit()

# Get disired script path
script_path = script_paths[script_ids.index(args.script)]

# Wait 3 seconds
for i in range(3):
    print(f"Script [{args.script}] will start in {3 - i} seconds...", end="\r")
    time.sleep(1)

# Run the script
print(f"Running [{args.script}]..." + " " * 25)
importlib.import_module(f"scripts.{args.script}")