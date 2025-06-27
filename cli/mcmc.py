import os
from pint.models import get_model_and_toas

from backend.datastores.database import database
from backend.utils.utils import utils
from backend.fitters.mcmc import MCMCFitter

class CLIMCMCFitter:
    """
    Command Line Interface for MCMC Fitter.
    """

    def __init__(self, psr, output, nwalkers, nsteps, ncpus, logger, tmpdir="/tmp"):
        # Initialize parameters
        self.psr = psr
        self.output = output
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.ncpus = ncpus
        self.logger = logger
        self.tmpdir = tmpdir + "/champss_timing__mcmc__" + utils.get_rand_string()
        self.psr_dir = None
        self.db_hdl = None

        # Create temporary directory
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)

        # Make sure the pulsar directory exists
        self.psr_dir = f"./timing_sources/{self.psr}"
        if not os.path.exists(self.psr_dir):
            raise FileNotFoundError(f"Pulsar directory {self.psr_dir} does not exist.")

        # Set output directory
        if self.output == None:
            self.output = self.psr_dir + "/mcmc_report.pdf"

        # Initialize database handler
        self.db_hdl = database(self.psr_dir + "/champss_timing.sqlite3.db", readonly=True)

    def __enter__(self):
        self.db_hdl.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.db_hdl:
            self.db_hdl.close()

    def fit(self):
        # Get paths
        parfile = os.path.join(self.psr_dir, "pulsar.par")
        timfile = os.path.join(self.psr_dir, "pulsar.tim")

        # Create parfile
        with open(parfile, "w") as f:
            f.write(
                self.db_hdl.create_parfile()
            )
        self.logger.debug(f"Temporary parfile created at {parfile}")

        # Get MJD range
        mjd_range = utils.read_start_end_from_parfile(
            parfile, 
            raise_exception=False
        )
        self.logger.info(f"Using MJD range: {mjd_range}")

        # Create timfile
        with open(timfile, "w") as f:
            f.write(
                self.db_hdl.create_timfile(
                    mjd_range=mjd_range
                )
            )
        self.logger.debug(f"Temporary timfile created at {timfile}")

        # Initialize model and toas
        m, t = get_model_and_toas(parfile, timfile)

        # Initialize fitter
        f = MCMCFitter(
            t, m, 
            nwalkers=self.nwalkers, 
            nsteps=self.nsteps, 
            n_pools=self.ncpus
        )

        # Run MCMC fitting
        f.fit_toas()

        # Save results
        f.plot(
            savefig=self.output
        )