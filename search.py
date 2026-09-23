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
parser.add_argument("--max-n-obs", "-N", required=False, default=60, type=int, help=f"Maximum number of observations to use in the search (default: 60)")
parser.add_argument("--max-duty-cycle", required=False, default=None, type=float, help=f"Maximum duty cycle of the pulsar profile (default: None)")
parser.add_argument("--ncpus", required=False, default=1, help="Number of CPU cores to use for the search (default: 1)")
args = parser.parse_args()

# Check if archives is provided
if not args.archives:
    raise ValueError("No archive files provided. Please specify at least one archive file.")

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
    }, 
    max_n_obs=args.max_n_obs,
    max_duty_cycle=args.max_duty_cycle
)

# Determine the output directory
if args.output is None:
    if args.psr is None:
        args.psr = cli_search.optimizer.model.PSR.value

    args.output = f"{TIMING_SOURCES_PATH}/{args.psr}"

# Set flag for generated pulsar name
generated_psrname = False
if args.psr is None and args.ra is not None:
        generated_psrname = True

# Print the parsed arguments
print("Parsed arguments:")
print(f" PSR: {args.psr} {'(generated from coordinates)' if generated_psrname else ''}")
print(f" RA: {cli_search.optimizer.model.RAJ.value / 24 * 360}")
print(f" Dec: {cli_search.optimizer.model.DECJ.value}")
print(f" F0: {cli_search.optimizer.model.F0.value}")
print(f" DM: {cli_search.optimizer.model.DM.value}")
print(f" F1: {cli_search.optimizer.model.F1.value}")
print(f" PEPOCH: {cli_search.optimizer.model.PEPOCH.value}")
print(f" Max duty cycle: {args.max_duty_cycle} ({'not set' if args.max_duty_cycle is None else f'{cli_search.optimizer.max_width} bins'})")
print(f" Parfile: {args.parfile if args.parfile is not None else '(not provided)'}")
print(f" Output: {args.output}")
print(f" Number of CPU cores: {args.ncpus}")
print(f" Archives: ")
for archive in cli_search.optimizer.archive_files:
    print(f"  {archive}")

# Optimize the initial timing solution and save the results
cli_search.optimize(output_dir=args.output, make_directory=True)