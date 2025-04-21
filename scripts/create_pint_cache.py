import argparse
import glob
import os
import pint
from pint.models import get_model_and_toas
from pint.residuals import Residuals
pint.logging.setup(level="WARNING")

from backend.datastores.database import database
from backend.utils.logger import logger
from backend.utils.utils import utils
logger = logger()

def get_pulsars():
    return [os.path.basename(psr) for psr in glob.glob("./timing_sources/*") if os.path.isdir(psr)]

parser = argparse.ArgumentParser(description="Download PINT cache data prior to processing. ")
parser.add_argument("--psr", type=str, help="Pulsar name.", default=None)
parser.add_argument("--tmpdir", type=str, help="Temporary directory for parfile and timfile for script running.", default="/tmp")
args = parser.parse_args()

if args.psr is None:
    pulsars = get_pulsars()
else:
    pulsars = [args.psr]

logger.info(f"Start caching {len(pulsars)} pulsars...")

for psr in pulsars:
    # check if psrdir exists
    logger.info(f"Start caching {psr}...", layer=1)
    db_path = f"./timing_sources/{psr}/champss_timing.sqlite3.db"
    if not os.path.exists(db_path):
        logger.warning(f"Database for {psr} does not exist. Skipping.")
        continue

    # create parfile and timfile
    logger.debug(f"Creating parfile and timfile for {psr}...", layer=2)
    with database(db_path, readonly=True, logger=logger.copy()) as db_hdl:
        if db_hdl.is_blank_db():
            logger.debug(f"Database for {psr} is empty. Skipping.")
            continue

        open(f"{args.tmpdir}/{psr}.par", "w").write(
            db_hdl.create_parfile()
        )
        open(f"{args.tmpdir}/{psr}.tim", "w").write(
            db_hdl.create_timfile() + "\n" + f"./fake/path/to/fake/data/cand_000.00_0.00_fake_data.ar 600.000000 {utils.mjd_now()} 1000.000  chime  -rcvr fake_rcvr"
        )


    # initialize pint
    logger.debug(f"Initializing PINT for {psr}...", layer=2)
    m, t = get_model_and_toas(f"{args.tmpdir}/{psr}.par", f"{args.tmpdir}/{psr}.tim")

    # run prefit
    Residuals(t, m)

    # clean up
    logger.debug(f"Cleaning up {psr}...", layer=2)
    os.remove(f"{args.tmpdir}/{psr}.par")
    os.remove(f"{args.tmpdir}/{psr}.tim")

    logger.success(f"Finished caching {psr}.", layer=1)

logger.success("Finished caching all pulsars.")