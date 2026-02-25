import argparse
import json
from backend import champss_timing
from backend.datastores import database, tmg_master
from cli.config import CLIConfig
from cli.pintk import PintkUtils

# Define
TIMING_SOURCES_PATH = "./timing_sources"

# Load configuration
cli_config = CLIConfig(load_error=False)

# Parse arguments
parser = argparse.ArgumentParser(description="PINTK Utility for CHAMPS Timing Data")
parser.add_argument("--psr", required=True, help="Pulsar name (e.g., J1234+5678)")
parser.add_argument("-o", "--output-dir", default="/tmp", help="Directory to save generated par and tim files (default: /tmp)")
parser.add_argument("--mjd-range", nargs=2, type=float, default=[0, 1e32], help="MJD range for TOAs (default: all)")
parser.add_argument("--initial-par", action="store_true", default=False, help="Use initial parfile instead of fitted parfile")
parser.add_argument("--no-pintk", action="store_true", help="Do not run pintk after generating files")
args = parser.parse_args()

psrdir = f"./{TIMING_SOURCES_PATH}/{args.psr}"
with database.database(f"{psrdir}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
    pintk_utils = PintkUtils(psrdir, db_hdl)

    parfile_path = f"{args.output_dir}/{args.psr}.par"
    timfile_path = f"{args.output_dir}/{args.psr}.tim"

    print(f"Generating par file at: {parfile_path} (using {'initial' if args.initial_par else 'fitted'} parfile)")
    pintk_utils.write_parfile(parfile_path, use_initial=args.initial_par)

    print(f"Generating tim file at: {timfile_path} with MJD range: {args.mjd_range}")
    pintk_utils.write_timfile(timfile_path, mjd_range=args.mjd_range)

    if not args.no_pintk:
        print("Starting pintk...")
        pintk_utils.run_pintk(parfile_path, timfile_path)