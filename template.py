import argparse

from cli.config import CLIConfig
from backend.datastores import database
from backend.tools.template_utils import StackTemplate
from backend.utils.logger import logger

# Load configuration
cli_config = CLIConfig(load_error=False)

# Initialize logger
logger = logger()

# Initialize parser
parser = argparse.ArgumentParser(description="CHAMPSS Timing Template Generator")
parser.add_argument("-p", "--psr", type=str, help="Pulsar name", required=True)
parser.add_argument("-o", "--output", type=str, help="Output template file name", default="paas.std")
parser.add_argument("-r", "--rcvr", type=str, help="Receiver name. Use all observations if not specified.", default=None)
parser.add_argument("--debug", action="store_true", help="Enable debug mode", default=False)
args = parser.parse_args()

logger.info(f"Generating template for {args.psr} > {args.output}")

# Get profiles and dm
pulse_profiles = {}
dm = None
with database.database(f"./timing_sources/{args.psr}/champss_timing.sqlite3.db", readonly=True, logger=logger.copy()) as db_hdl:
    logger.debug("Reading profiles from database...", layer=1)

    toas = db_hdl.get_all_toas()
    for toa in toas:
        if args.rcvr is not None:
            if toa["notes"]["rcvr"] != args.rcvr:
                continue

        this_filename = toa["filename"]
        this_mjd = toa["toa"]
        pulse_profiles[this_mjd] = db_hdl.get_archive_info_by_filename(this_filename)["psr_amps"]

    try:
        dm = db_hdl.get_last_timing_info()["fitted_params"]["DM"]
        logger.debug(f"DM = {dm} from last fitted parameters.", layer=1)
    except KeyError:
        dm = 0
        logger.warning("No DM found in the last fitted parameters. Using DM = 0.", layer=1)

        
# sort by key
logger.debug("Sorting profiles...")
pulse_profiles = dict(sorted(pulse_profiles.items()))

# Create template
logger.debug("Optimizing template...")
stpl = StackTemplate(list(pulse_profiles.values()), shift_meth="discrete", logger=logger.copy(), verbose=args.debug)
stpl.optimize()

# Save template
logger.debug("Writing template...")
stpl.to_archive(args.output, psr=args.psr, dm=dm)

logger.success(f"Template saved to {args.output}")