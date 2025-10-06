import argparse
import traceback
from cli.mcmc import CLIMCMCFitter
from backend.utils.logger import logger

# Initialize parser
parser = argparse.ArgumentParser(description="Run MCMC fitting for a pulsar.")
parser.add_argument(
    "--psr",
    dest="psr",
    required=True,
    type=str,
    help="Name of the pulsar. "
)
parser.add_argument(
    "--ncpus",
    dest="ncpus",
    default=4,
    type=int,
    help="Number of CPUs to run in parallel (default=4). "
)
parser.add_argument(
    "-o",
    "--output",
    default=None,
    dest="output",
    help="Output file for the MCMC report (default=PSR_DIR/mcmc_report.pdf). "
)
parser.add_argument(
    "--nwalkers",
    dest="nwalkers",
    default=250,
    type=int,
    help="Number of walkers for MCMC (default=250). "
)
parser.add_argument(
    "--nsteps",
    dest="nsteps",
    default=2500,
    type=int,
    help="Number of steps for MCMC (default=2500). "
)

# Parse arguments
args = parser.parse_args()

# Initialize logger
logger = logger()

# Print parsed arguments
logger.info(f"Starting MCMC fitting for pulsar: {args.psr}")
logger.info(f"Output will be saved to: {args.output}")
logger.info(f"Number of CPUs to use: {args.ncpus}")

# Start MCMC fitting
try:
    with CLIMCMCFitter(
        psr=args.psr, 
        output=args.output, 
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        ncpus=args.ncpus, 
        logger = logger.copy()
    ) as fitter:
        fitter.fit()
except Exception as e:
    logger.error(f"An error occurred during MCMC fitting: {e}")
    logger.error(traceback.format_exc())
    raise e

# Print completion message
logger.info("MCMC fitting completed successfully.")