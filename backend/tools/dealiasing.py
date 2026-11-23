import os
import json
import glob
import pickle
import shutil
import tqdm
import numpy as np
import astropy.units as u
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import gaussian_filter

from ..utils.utils import utils
from ..utils.exec import exec
from ..utils.logger import logger
from ..utils.dm import DMOptimizer
from ..utils.aliasing import AliasingFinder
from ..tools.stacking import Stacking
from ..pipecore.psrchive import psrchive_handler
from ..datastores.database import database

class alias_utils():
    def __init__(self, psrdir, ar_list, parfile, jumps={}, n_subints=32, n_bins=128, n_freqs=256, workspace="/tmp", cleanup=True, mode="auto", n_pools="auto", logger=logger()):
        """
        Initialize the alias_utils class.

        Parameters:
        psrdir (str): Directory containing pulsar data.
        ar_list (list): List of archive files.
        parfile (str): Parameter file.
        jumps (dict): Dictionary of jumps.
        n_subints (int): Number of sub-integrations.
        n_bins (int): Number of bins.
        n_freqs (int): Number of frequency channels (default to one, but can be set to more if frequency resolution is needed as a by-product).
        workspace (str): Workspace directory.
        cleanup (bool): Whether to clean up the workspace.
        mode (str): Mode of operation.
        n_pools (str): Number of pools for parallel processing.
        logger (logger): Logger instance.
        """

        # set parameters
        self.psrdir = psrdir
        self.ar_list = ar_list
        self.parfile = parfile
        self.mode = mode
        self.fs = []
        self.fs_prepared = []
        self.jumps = jumps
        self.n_subints = n_subints
        self.n_bins = n_bins
        self.n_freqs = n_freqs
        self.workspace = workspace + f"/{utils.get_time_string()}__{utils.get_rand_string()}"
        self.outdir = self.workspace + "/outfiles"
        self.sidereal_day = 0.99727 # day
        self.cleanup_workspace = cleanup
        self.alias_factor = None
        self.snr_stacked = 0.0
        self.avg_snr_per_obs = 0.0
        self.avg_snr_per_subints = 0.0
        self.data_stacked = []
        self.su = None
        self.summary = None

        # set methods
        self.logger = logger
        self.exec_handler = exec
        self.n_pools = n_pools
        self.psrchive = psrchive_handler(self)

        # # get update eph methods from archive_cache
        # self.update_eph = archive_cache("").exec_update_model
        # self.get_md5 = archive_cache("").get_md5

    def _initialize__copy_ar(self, ar):
        # check file size
        if os.path.getsize(ar) == 0:
            self.logger.warning(f"File size is 0: {ar}, skipping", layer=1)
            return None
        self.logger.debug(f"Copying {ar} to workspace", layer=1)
        shutil.copy(ar, self.workspace)
        return f"{self.workspace}/{os.path.basename(ar)}"

    def initialize(self):
        # create workspace
        self.logger.debug(f"Creating workspace: {self.workspace}")
        os.makedirs(self.workspace)
        os.makedirs(self.outdir)
        
        # Initialize stack_utils
        self.logger.debug(f"Stacking with {len(self.ar_list)} archives")
        self.logger.debug(f"Requesting: n_subints={self.n_subints}, n_pols=1, n_bins={self.n_bins}, n_freqs={self.n_freqs}")
        self.su = stack_utils(
            files=self.ar_list,
            parfile=self.parfile,
            workspace=self.workspace, 
            n_subs=self.n_subints, 
            n_pols=1, 
            n_freqs=self.n_freqs, 
            n_bins=self.n_bins, 
            n_pools=self.n_pools, 
            jumps=self.jumps, 
            remove_baseline=True,
            interpolate="minimal",
            logger=self.logger.copy()
        )

        # Stack
        self.su.stack()

        # Check if stacking was successful
        if self.su.n_stacked == 0:
            raise Exception("Stacking failed, no files stacked. This may be due to the input files being empty or all files being skipped due to failure in processing.")

        # Update n_subints, n_bins, n_freqs based on the stacked data
        if self.su.n_subs != self.n_subints:
            self.n_subints = self.su.n_subs
            self.logger.info(f"Updated n_subints from {self.n_subints} to {self.su.n_subs} from stacked data", layer=1)
        if self.su.n_bins != self.n_bins:
            self.n_bins = self.su.n_bins
            self.logger.info(f"Updated n_bins from {self.n_bins} to {self.su.n_bins} from stacked data", layer=1)
        if self.su.n_freqs != self.n_freqs:
            self.n_freqs = self.su.n_freqs
            self.logger.info(f"Updated n_freqs from {self.n_freqs} to {self.su.n_freqs} from stacked data", layer=1)

    def save_outfiles(self, filename="auto", pickle_only=False):
        # Get filename
        if filename == "auto":
            filename = self.psrdir + "/dealias"
            
        # Check if outdir exists
        if (not os.path.exists(self.outdir) or glob.glob(self.outdir) == []) and not pickle_only:
            raise Exception(f"Directory does not exist / empty: {self.outdir}. Was the alias_utils initialized or loaded from a pickle file?")

        # Save pickle
        with open(self.outdir + "/alias_utils.pkl", "wb") as f:
            pickle.dump(self, f)
            self.logger.success(f"Pickle saved to {self.outdir}/alias_utils.pkl")
            if pickle_only:
                return

        # Save stacked data
        self.su.save(
            filename=os.path.join(self.outdir, "stacked_data.npz"), 
            format="npz"
        )

        # Save summary to json 
        with open(os.path.join(self.outdir, "summary.json"), "w") as f:
            # Make sure everything is serializable
            summary_json = {}
            for key, value in self.summary.items():
                # np float 
                if isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, np.integer):
                    value = int(value)

                summary_json[key] = value
            json.dump(summary_json, f, indent=4)
            
        # Copy everything in outdir to filename
        if not os.path.isdir(filename):
            if os.path.exists(filename):
                raise Exception(f"Directory already exists and is not writable: {filename}. Is there a file with the same name?")
        else:
            # Remove the directory
            shutil.rmtree(filename)

        # Create the directory
        os.makedirs(filename)

        # Copy the contents of outdir to filename
        for f in glob.glob(self.outdir + "/*"):
            self.logger.debug(f"Copying {f} to {filename}", layer=1)
            if os.path.isdir(f):
                shutil.copytree(f, os.path.join(filename, os.path.basename(f)))
            else:
                shutil.copy2(f, filename)
        self.logger.success(f"Save outputs to {filename}")
        
        # self.cleanup_workspace = False

    def write_db(self, db_path="auto", db_hdl=None):
        # Check if summary is available
        if not self.summary:
            raise ValueError("No summary available. Please run dealias() first.")

        # Check if a database handle is provided
        if db_hdl is None:
            # Get db_path
            if db_path == "auto":
                db_path = f"{self.psrdir}/champss_timing.sqlite3.db"
            
            # Create a new database
            self.logger.debug(f"Creating new database at {db_path}")
            loaded_db_hdl = database(db_path)
        else: 
            loaded_db_hdl = db_hdl

        # Write database
        loaded_db_hdl.insert_dealias_history(
            n_stacked=float(self.summary["n_stacked"]), # ensure they are not np.float64
            alias_factor=float(self.summary["alias_factor"]),
            snr_stacked=float(self.summary["snr_stacked"]),
            notes=self.summary["notes"]
        )

        # Close db_hdl
        if db_hdl is None:
            self.logger.debug(f"Closing database at {db_path}")
            loaded_db_hdl.close()

        self.logger.success(f"Database updated: {db_path}")

    def cleanup(self):
        if self.cleanup_workspace:
            self.logger.debug("Cleaning up workspace")
            shutil.rmtree(self.workspace)
        else:
            self.logger.debug(f"Workspace will NOT be removed at {self.workspace} due to cleanup=False")

    def commit(self, db_hdl=None):
        '''
        Commit the changes to the psrdir and database

        Parameters:
        - db_hdl: Optional database handle. If not provided, a new database connection will be created and closed within the method.
        '''

        # Commit changes to psrdir
        self.logger.debug(f"Committing changes to psrdir: {self.psrdir}")
        self.logger.level_up()
        self.save_outfiles()
        self.logger.level_down()

        # Commit changes to database
        self.logger.debug(f"Committing changes to database: {self.psrdir}")
        self.logger.level_up()
        self.write_db(db_hdl=db_hdl)
        self.logger.level_down()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.error(f"Error while running find_alias. Workspace will NOT be removed at {self.workspace}")
            try:
                self.save(self.workspace + "/alias_utils.pkl")
                self.logger.error(f"Alias_utils object saved to {self.workspace}/alias_utils.pkl")
            except:
                self.logger.error(f"Error while saving alias_utils object to {self.workspace}/alias_utils.pkl")
            raise exc_type(exc_value).with_traceback(traceback)
        else:
            self.cleanup()

class DealiasingPayload:
    def __init__(self, logger=logger()):
        # Set logger
        self.logger = logger

        # Stacked data
        self.stacked_data = None

        # DM optimizing
        self.dmo = None
        self.best_dm_val = None
        self.best_dm_err = None

        # Aliasing finder
        self.af_res = None
        self.sol = None
        self.af_res_summary = None
        self.sol_info = None

    def set_stacked_data(self, stacked_data):
        self.stacked_data = stacked_data

        # Set dm values to the dm in stacked data as placeholder
        if self.best_dm_val is None or self.best_dm_err is None:
            self.best_dm_val = stacked_data.meta.dm
            self.best_dm_err = 0.0

    def set_optimized_dm(self, dmo_results):
        self.dmo = dmo_results
        self.best_dm_val = dmo_results.value
        self.best_dm_err = dmo_results.uncertainty

    def set_dealiased_solution(self, af_res, sol):
        self.af_res = af_res
        self.sol = sol

        # Get solution summary
        self.af_res_summary = af_res.get_summary()
        self.sol_info = sol.get_info()

    def validate(self):
        if self.stacked_data is None:
            raise ValueError("Stacked data is not set")
        if self.best_dm_val is None or self.best_dm_err is None:
            raise ValueError("Optimized DM values are not set")
        if self.sol is None or self.sol_info is None:
            raise ValueError("Dealiased solution or its info is not set")
        if self.af_res is None or self.af_res_summary is None:
            raise ValueError("Aliasing finder result or its summary is not set")

    def get_remark(self):
        return "DEALIAS_FITTING_OK" if self.sol.status else "DEALIAS_FITTING_FAILED"

    def unpack(self, outdir, overwrite=False):
        # Remove existing directory if requested
        if os.path.exists(outdir):
            if overwrite:
                shutil.rmtree(outdir)
            else:
                raise FileExistsError(f"Directory {outdir} already exists and overwrite is False")

        # Create the directory
        os.makedirs(outdir)

        # Stacked data
        self.stacked_data.save(os.path.join(outdir, "stacked_data.npz"))

        # Dealiasing summary
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            f.write(json.dumps(self.af_res_summary))

        # Aliasing finder plot
        self.af_res.plot(os.path.join(outdir, "diagnostics.pdf"))

        # Dealiased timing residual plot
        self.sol.plot(os.path.join(outdir, "timing_residuals.pdf"))

        # Dealiased timing solution
        if self.af_res_summary["alias_factor"] != 0:
            self.sol.write_dealiased_parfile(os.path.join(outdir, "pulsar.dealiased.par"))
        else:
            self.logger.debug("Timing solution is unaliased. Skip writing dealiased parfile")

        # DMO plot
        if self.dmo is not None:
            self.dmo.plot(os.path.join(outdir, "dmo.pdf"))
        else:
            self.logger.debug("DM optimizer result is not available. Skip plotting DM diagnostics.")

    def write_db(self, db):
        # Write database
        db.insert_dealias_history(
            n_stacked=float(self.af_res_summary["n_stacked"]), # ensure they are not np.float64
            alias_factor=float(self.af_res_summary["alias_factor"]),
            snr_stacked=float(self.af_res_summary["snr_stacked"]),
            notes={"remark": self.get_remark()}
        )

class Dealiasing:
    def __init__(
            self, 
            psrdir, 
            ar_list, 
            parfile=None, 
            jumps={}, 
            n_subints=8, 
            n_freqs=256, 
            n_bins=256, 
            workspace="/tmp", 
            ncpus=1, 
            logger=logger(), 
            verbose=False, 
            inj_ddm=None, 
            inj_af=None
        ):

        # Set input parameters
        self.psrdir = psrdir
        self.parfile = parfile
        self.ar_list = ar_list
        self.jumps = jumps
        self.n_subints = n_subints
        self.n_bins = n_bins
        self.n_freqs = n_freqs
        self.workspace = workspace
        self.ncpus = ncpus
        self.logger = logger
        self.verbose = verbose
        self.inj_ddm = inj_ddm
        self.inj_af = inj_af

        # Use psrdir parfile when no parfile is provided
        if self.parfile is None:
            self.parfile = os.path.join(self.psrdir, "pulsar.par")

        # Set up workspace
        self.workspace = workspace + f"/{utils.get_time_string()}__{utils.get_rand_string()}"

        # Setup payload
        self.payload = DealiasingPayload(logger=self.logger.copy())

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.verbose:
            self.cleanup()
        else:
            self.logger.info(f"Verbose mode enabled. Workspace retained at {self.workspace}")

    def setup(self):
        # Create workspace if it doesn't exist
        self.logger.debug(f"Setting up workspace at {self.workspace}")
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        # Copy parfile to workspace
        self.logger.debug(f"Copying parfile to workspace at {self.workspace}")
        shutil.copy(self.parfile, os.path.join(self.workspace, "pulsar.par"))
        self.parfile = os.path.join(self.workspace, "pulsar.par")

        # read the mjd_start and mjd_end from the parfile
        mjd_start, mjd_end = utils.read_start_end_from_parfile(self.parfile, raise_exception=False)

        # Generate timfile
        self.timfile = os.path.join(self.workspace, "pulsar.tim")
        with database(f"{self.psrdir}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
            open(self.timfile, "w").write(
                # create only mjds between the mjd range specified in the parfile
                db_hdl.create_timfile(mjd_range=[mjd_start, mjd_end])
            )

        # Create stacked data
        self.logger.debug("Creating stacked data")
        stacking = Stacking(
            files=self.ar_list, 
            parfile=self.parfile, 
            workspace=self.workspace,
            ncpus=self.ncpus, 
            logger=self.logger.copy()
        )

        # Store the stacked data in the payload
        self.payload.set_stacked_data(
            stacking.stack()
        )

        # Inject DM or aliasing if specified
        if self.inj_ddm is not None:
            self.logger.debug(f"DM before injection: {self.payload.stacked_data.meta.dm}")
            self.logger.debug(f"Injecting DM offset: {self.inj_ddm}")
            self.payload.stacked_data.refine_dm(ddm=-self.inj_ddm)
            self.logger.debug(f"DM after injection: {self.payload.stacked_data.meta.dm}")
        if self.inj_af is not None:
            self.logger.debug(f"Injecting aliasing factor: {self.inj_af}")
            self.payload.stacked_data.dealiasing(alias_factor=-self.inj_af)

        # Drop partial subints
        self.logger.debug("Dropping partial subints")
        self.payload.stacked_data.drop_partial_subints()

        # Scrunch to the desired sizes
        self.logger.debug("Scrunching stacked data to desired sizes")
        self.payload.stacked_data.scrunch(axis=1)
        self.payload.stacked_data.downsample_to_length(self.n_subints, axis=0)
        self.payload.stacked_data.downsample_to_length(self.n_freqs, axis=2)
        self.payload.stacked_data.downsample_to_length(self.n_bins, axis=3)

        # Verbose plotting
        if self.verbose:
            self.logger.debug("Generating stacked data plot")
            self.payload.stacked_data.plot(savefig=f"{self.workspace}/stacked_data.png")

    def cleanup(self):
        # Remove workspace if it exists
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
            self.logger.debug(f"Workspace removed at {self.workspace}")

    def optimize_dm(self):
        self.logger.debug("Optimizing DM")

        print("!!!!", self.payload.stacked_data.meta.dm)

        # Run DM optimization
        results = DMOptimizer(
            stacked_data=self.payload.stacked_data, 
            ncpus=self.ncpus,
            logger=self.logger.copy()
        ).optimize()

        # Unpack best DM measurement to payload
        self.payload.set_optimized_dm(results)

        # Verbose plot
        if self.verbose:
            self.logger.debug("Generating DM optimization plot")
            results.plot(savefig=f"{self.workspace}/dm_optimization.png")

        # Refine stacked data at the best DM found
        self.logger.debug(f"Refining stacked data at the best DM = {self.payload.best_dm_val} ± {self.payload.best_dm_err}")
        self.payload.stacked_data.dedisperse(dm=self.payload.best_dm_val)

    def resolve_aliasing(self, ignore_subints=[], smooth_sigma=None):
        self.logger.debug("Finding aliasing")

        # Find aliasing
        results = AliasingFinder(
            stacked_data=self.payload.stacked_data,
            ncpus=self.ncpus,
            logger=self.logger.copy()
        ).find_aliasing_by_shifts(
            ignore_subints=ignore_subints,
            smooth_sigma=smooth_sigma
        )

        # Print summary
        results.print_summary()

        # Resolve Aliasing
        print("Resolving aliasing...")
        sol = results.dealias(self.parfile, self.timfile)

        # Unpack summary to payload
        self.payload.set_dealiased_solution(results, sol)

        # Print information
        sol.print_info()

        # Verbose plot
        if self.verbose:
            self.logger.debug("Generating aliasing solution plot")
            sol.plot(savefig=f"{self.workspace}/aliasing_solution.png")

    def commit(self, db=None, overwrite=True):

        # Generate output directory path
        outdir = f"{self.psrdir}/dealias"

        # Initialize database 
        this_db = None
        if this_db is None:
            this_db = database(f"{self.psrdir}/champss_timing.sqlite3.db")

        # Commit changes to psrdir
        self.logger.debug(f"Committing changes to psrdir: {self.psrdir}")
        self.logger.level_up()
        self.payload.unpack(outdir, overwrite=overwrite)
        self.logger.level_down()

        # Commit changes to database
        self.logger.debug(f"Committing changes to database: {self.psrdir}")
        self.logger.level_up()
        self.payload.write_db(db=this_db)
        self.logger.level_down()

        # Close database connection
        if db is None: # only when we created this_db ourselves
            this_db.close()