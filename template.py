import argparse

from cli.config import CLIConfig
from backend.datastores import database
from backend.tools.template_utils import StackTemplate, get_template_from_archive
from backend.utils.logger import logger

# Load configuration
cli_config = CLIConfig(load_error=False)

# Initialize logger
logger = logger()

# Initialize parser
# parser = argparse.ArgumentParser(description="Create a pulse template from stacked profiles.")
# parser.add_argument("action", type=str, help="Action to perform", choices=["stack", "center"])
# parser.add_argument("-p", "--psr", type=str, help="Pulsar name", required=True)
# parser.add_argument("-o", "--output", type=str, help="Output template file name", default="paas.std")
# parser.add_argument("-r", "--rcvr", type=str, help="Receiver name. Use all observations if not specified.", default=None)
# parser.add_argument("--debug", action="store_true", help="Enable debug mode", default=False)
# parser.add_argument("--center", action="store_true", help="Center the template profile", default=False)
# args = parser.parse_args()

parser = argparse.ArgumentParser(description="Utility for creating pulse profile templates.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
subparsers = parser.add_subparsers(dest="action", required=True, metavar="action")

# Subparser - stack
subparser_stack = subparsers.add_parser(
    "stack",
    help="Stack observations into a template",
    description="Stack all matching observations into a single pulse profile template.",
)
subparser_stack.add_argument("-p", "--psr", type=str, help="Pulsar name", required=True)
subparser_stack.add_argument("-o", "--output", type=str, help="Output template file name", default="paas.std")
subparser_stack.add_argument("-r", "--rcvr", type=str, help="Receiver name. Use all observations if not specified.", default=None)
subparser_stack.add_argument("--debug", action="store_true", help="Enable debug mode", default=False)
subparser_stack.add_argument("--center", action="store_true", help="Center the template profile", default=False)

# Subparser - center
subparser_center = subparsers.add_parser(
    "center",
    help="Center an existing template",
    description="Rotate an existing template so the pulse peak is centered. Either provide the template file or specify the pulsar name.",
)
subparser_center.add_argument("-p", "--psr", type=str, help="Pulsar name", required=False)
subparser_center.add_argument("-m", "--modify", action="store_true", help="Modify the template in place", default=False)
subparser_center.add_argument("-o", "--output", type=str, help="Output template file name", required=False)
subparser_center.add_argument("-t", "--template", type=str, help="Template file", required=False)

args = parser.parse_args()

if args.action == "stack":
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
    stpl.to_archive(args.output, psr=args.psr, dm=dm, center=args.center)

    logger.success(f"Template saved to {args.output}")

elif args.action == "center":
    logger.info(f"Centering template {args.template}")

    if args.template is None:
        if args.psr is None:
            raise ValueError("Either a template file or a pulsar name must be provided.")
        args.template = f"./timing_sources/{args.psr}/paas.std"
        logger.info(f"Using default template file for pulsar {args.psr}: {args.template}")
    
    if args.output is None:
        if args.modify:
            args.output = args.template
        else:
            raise ValueError("Output file must be specified unless modifying the template in place.")

    # Load template
    template = get_template_from_archive(args.template)

    # Center template
    template.center()

    # Save template
    template.to_archive(args.output)
    logger.success(f"Centered template saved to {args.output}")