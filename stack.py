import os
import argparse
from cli.config import CLIConfig
from backend.datastores import tmg_master
from backend.tools.stack_utils import stack_utils

# Define
TIMING_SOURCES_PATH = "./timing_sources"
MASTER_DB_PATH = TIMING_SOURCES_PATH + "/TMGMaster.sqlite3.db"

# Load configuration
cli_config = CLIConfig(load_error=False)

# Parse arg
parser = argparse.ArgumentParser(description="Stack data from multiple observations.")
parser.add_argument("--psr", type=str, required=True, help="Pulsar name to stack data for.")
parser.add_argument("--output", type=str, required=True, help="Output file for stacked data.")
parser.add_argument("--nobs", type=int, required=False, default=32, help="Number of observations to stack (default: all).")
parser.add_argument("--mjd-range", type=int, required=False, help="MJD range to stack (START:FINISH).")
parser.add_argument("--npols", type=int, required=False, default=4, help="Number of polarizations to scrunch (default: 4).")
parser.add_argument("--nsubs", type=int, required=False, default=16, help="Number of frequency channels to scrunch (default: 16).")
parser.add_argument("--nfreqs", type=int, required=False, default=1024, help="Number of frequency bins to stack into (default: 1024).")
parser.add_argument("--nbins", type=int, required=False, default=1024, help="Number of phase bins to stack into (default: 1024).")
parser.add_argument("--ncpus", type=int, default=1, help="Number of parallel pools to use (default: 1).")
parser.add_argument("--no-normalize", action="store_true", default=False, help="Disable normalization in stacking.")
parser.add_argument("--backend", type=str, required=False, default="all", help="Backend to use for stacking (default: all).")
parser.add_argument("--parfile", type=str, required=False, help="Parfile to use for stacking (default: pipeline output parfile).")
parser.add_argument("--output-format", type=str, required=False, default="npy", help="Output file format (npy, pkl; default: npy).")
parser.add_argument("--tmpdir", type=str, required=False, default="/tmp", help="Temporary directory to use during stacking (default: /tmp).")
args = parser.parse_args()

# Sanity check
if args.output_format not in ["npy", "pkl"]:
    raise ValueError("Unsupported output format. Use 'npy' or 'pkl'.")

# Get parfile
if args.parfile is None:
    args.parfile = f"./{TIMING_SOURCES_PATH}/{args.psr}/pulsar.par"

# Get files to stack
with tmg_master.tmg_master(MASTER_DB_PATH, readonly=True) as tmgm:
    data = tmgm.get_raw_data(psr_id=args.psr)
    if data is None or len(data) == 0:
        raise ValueError(f"No data found for pulsar {args.psr} in master database.")

    # files = []
    # while len(files) < args.nobs and len(data) > 0:
    #     entry = data.pop(0)

    #     if args.backend != "all" and entry['backend'] != args.backend:
    #         continue

    #     files.append({"location": entry['location'], "backend": entry['backend']})

    # Filter data by backend
    data_ = []
    for entry in data:
        if args.backend == "all" or entry['backend'] == args.backend:
            data_.append(entry)
    data = data_

    # Filter data by MJD range
    if args.mjd_range is not None:
        try:
            mjd_start, mjd_end = map(float, args.mjd_range.split(":"))
        except:
            raise ValueError("Invalid MJD range format. Use START:FINISH.")
        
        data_ = []
        for entry in data:
            if mjd_start <= entry['mjd'] <= mjd_end:
                data_.append(entry)
        data = data_

    # Sort data by MJD (newest to oldest)
    data.sort(key=lambda x: x['mjd'], reverse=True)

    # Filter data by number of observations
    data = data[:args.nobs]

    # Extract file locations and backends
    files = []
    for entry in data:
        files.append({"location": entry['location'], "backend": entry['backend']})


# Load jump parameters
jumps = {}
config = cli_config.get_config()
for bknd in config['backends']:
    jumps[bknd] = config['backends'][bknd]['jump']

# Print info
print(f"Stacking data for pulsar: {args.psr}")
print(f"Number of files to stack: {len(files)}")
print(f"Number of polarizations: {args.npols}")
print(f"Number of frequency channels: {args.nsubs}")
print(f"Number of frequency bins: {args.nfreqs}")
print(f"Number of phase bins: {args.nbins}")
print(f"Number of CPUs: {args.ncpus}")
print(f"Using jumps: {jumps}")
print(f"Output file: {args.output} (format: {args.output_format})")
print(f"Parfile: {args.parfile}")
print("Files to be stacked:")
for f in files:
    print(f"  - {f['location']} (backend: {f['backend']})")

# Make sure tmpdir exists
if not os.path.exists(args.tmpdir):
    os.makedirs(args.tmpdir)

# Run stacking
su = stack_utils(
    files=files, 
    parfile=args.parfile, 
    n_subs=args.nsubs, 
    n_pols=args.npols, 
    n_freqs=args.nfreqs, 
    n_bins=args.nbins, 
    n_pools=args.ncpus, 
    jumps=jumps, 
    workspace=args.tmpdir
)
su.stack(normalize=not args.no_normalize)

# Save output
data = su.get_data()
if args.output_format == "npy":
    import numpy as np
    np.save(args.output, data)
elif args.output_format == "pkl":
    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
else:
    raise ValueError("Unsupported output format. Use 'npy' or 'pkl'.")
print(f"Stacked data saved to {args.output}.")