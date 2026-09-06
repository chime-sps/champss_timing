import argparse
from cli.config import CLIConfig
from cli.search import CLIInitialTimingSolutionSearch

# Define
TIMING_SOURCES_PATH = "./timing_sources"

# Load configuration
cli_config = CLIConfig(load_error=False)

# Parse arguments
parser = argparse.ArgumentParser(description="Search for initial timing solutions given a coarse spindown measurement. Either provide model parameters or a parfile. ")
parser.add_argument("archives", nargs='*', help="List of archive files to be used in the search")
parser.add_argument("--ra", required=False, help="Right ascension of the pulsar in degrees")
parser.add_argument("--dec", required=False, help="Declination of the pulsar in degrees")
parser.add_argument("--f0", required=False, help="Spin frequency of the pulsar in Hz")
parser.add_argument("--dm", required=False, help="Dispersion measure of the pulsar in pc/cm^3")
parser.add_argument("--f1", required=False, default=0.0, help="Spin-down rate of the pulsar in Hz/s (optional)")
parser.add_argument("--pepoch", required=False, default=None, help="Reference epoch of the pulsar in MJD (optional)")
parser.add_argument("--psr", required=False, default=None, help="Pulsar name (e.g., J1234+5678) (optional)")
parser.add_argument("--parfile", required=False, default=None, help="Parfile containing pulsar parameters (optional; ignore parameter inputs if provided)")
parser.add_argument("--output", required=False, default=None, help=f"Output directory for generated diagnostics, parfile, and stacked profile (default: ./{TIMING_SOURCES_PATH}/<psrname>)")
parser.add_argument("--ncpus", required=False, default=1, help="Number of CPU cores to use for the search (default: 1)")
args = parser.parse_args()

# Initialize the CLI search
cli_search = CLIInitialTimingSolutionSearch(
    archive_files=args.archives,
    parfile=args.parfile,
    params={
        "ra": args.ra,
        "dec": args.dec,
        "f0": args.f0,
        "dm": args.dm,
        "f1": args.f1,
        "pepoch": args.pepoch,
        "psrname": args.psr
    }
)

# Determine the output directory
make_directory = False
generated_psrname = False
if args.output is None:
    if args.psr is None:
        args.psr = cli_search.optimizer.model.PSR.value

        # Set flag for generated pulsar name
        if args.parfile is None:
                generated_psrname = True

    make_directory = True
    args.output = f"{TIMING_SOURCES_PATH}/{args.psr}"

# Print the parsed arguments
print("Parsed arguments:")
print(f" PSR: {args.psr} {'(generated from coordinates)' if generated_psrname else ''}")
print(f" RA: {cli_search.optimizer.model.RAJ.value / 24 * 360}")
print(f" Dec: {cli_search.optimizer.model.DECJ.value}")
print(f" F0: {cli_search.optimizer.model.F0.value}")
print(f" DM: {cli_search.optimizer.model.DM.value}")
print(f" F1: {cli_search.optimizer.model.F1.value}")
print(f" PEPOCH: {cli_search.optimizer.model.PEPOCH.value}")
print(f" Parfile: {args.parfile if args.parfile is not None else '(not provided)'}")
print(f" Output: {args.output}")
print(f" Number of CPU cores: {args.ncpus}")
print(f" Archives: ")
for archive in args.archives:
    print(f"  {archive}")

# Optimize the initial timing solution and save the results
cli_search.optimize(output_dir=args.output, make_directory=make_directory)