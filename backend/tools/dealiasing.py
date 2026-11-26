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

def estimate_minimal_obs_for_subint_snr(obs_snrs, n_subints, subint_snr, min_n_obs=5):
    """
    Estimate the minimal number of observations required to achieve the desired SNR per subintegration.

    Parameters:
    - obs_snrs: List of SNRs for each observation.
    - n_subints: Number of subintegrations.
    - subint_snr: Desired SNR per subintegration.

    Returns:
    - Minimal number of observations required.
    """

    # Get mean snr for each archive file
    mean_snr = np.mean(obs_snrs)

    # Calculate the required number of files to reach the required snr per subint
    # Assuming the snr adds in quadrature for multiple observations / each subints: 
    #   SNR_subint = SNR_per_obs * sqrt(N_obs) / sqrt(N_subints)
    n_obs = (
        subint_snr * np.sqrt(n_subints) / mean_snr
    )**2

    # Ensure a minimum number of observations to robustly eliminate artifacts in single observation
    if n_obs < min_n_obs:
        n_obs = min_n_obs

    return int(np.ceil(n_obs)) # ceiling to the next integer

def estimate_minimal_obs_for_subint_snr_from_database(db, n_subints, subint_snr, min_n_obs=5):
    """
    Estimate the minimal number of observations required for each receiver to achieve the desired SNR per subintegration.

    Parameters:
    - db: Database path or handler.
    - n_subints: Number of subintegrations.
    - subint_snr: Desired SNR per subintegration.

    Returns:
    - Dictionary mapping receiver to minimal number of observations required.
    """

    # Initialize database if needed
    db_hdl = db
    if isinstance(db, str):
        db_hdl = database(db, readonly=True)
    
    # Get the snr of observations from database and index by rcvr
    obs_snrs_by_rcvr = {}
    for f in db_hdl.get_all_archive_info():
        # Unpack rcvr and snr
        rcvr, snr = f["notes"]["rcvr"], f["psr_snr"]

        if rcvr not in obs_snrs_by_rcvr:
            obs_snrs_by_rcvr[rcvr] = []
        obs_snrs_by_rcvr[rcvr].append(snr)

    # Estimate minimal number of observations for each receiver
    min_obs_by_rcvr = {}
    for rcvr, obs_snrs in obs_snrs_by_rcvr.items():
        min_obs_by_rcvr[rcvr] = estimate_minimal_obs_for_subint_snr(obs_snrs, n_subints, subint_snr, min_n_obs=min_n_obs)

    # Close the database if it was initialized in this session
    if isinstance(db, str):
        db_hdl.close()

    return min_obs_by_rcvr

def get_files_for_dealiasing_pipeline(min_obs_by_rcvr, ar_list_by_rcvr):
    # Sort the min_obs_by_rcvr dictionary by the number of required observations
    sorted_min_n_obs = sorted(min_obs_by_rcvr.items(), key=lambda x: x[1])

    for backend, n_required in sorted_min_n_obs:
        if backend in ar_list_by_rcvr and len(ar_list_by_rcvr[backend]) >= n_required:
            return backend, ar_list_by_rcvr[backend], min_obs_by_rcvr[backend]

    return None, [], 0

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

    def validate(self):
        if self.stacked_data is None:
            raise ValueError("Stacked data is not set")
        if self.best_dm_val is None or self.best_dm_err is None:
            raise ValueError("Optimized DM values are not set")
        if self.sol is None:
            raise ValueError("Dealiased solution is not set")
        if self.af_res is None or self.af_res_summary is None:
            raise ValueError("Aliasing finder result or its summary is not set")

    def get_remark(self):
        return "DEALIAS_FITTING_OK" if self.sol.status else "DEALIAS_FITTING_FAILED"

    def get_dm_info(self):
        return {
            "optimized": True if self.dmo is not None else False,
            "val": float(self.best_dm_val),
            "err": float(self.best_dm_err)
        }

    def get_summary(self):
        return {
            "psr_id": self.af_res_summary["psr_id"],
            "n_stacked": self.af_res_summary["n_stacked"],
            "alias_factor": self.af_res_summary["alias_factor"],
            "snr_stacked": self.af_res_summary["snr_stacked"],
            "avg_snr_per_subints": self.af_res_summary["avg_snr_per_subints"],
            "timestamp": self.af_res_summary["timestamp"],
            "remark": self.get_remark(),
            "best_dm": self.get_dm_info()
        }

    def get_summary_text(self):
        summary = self.get_summary()
        lines = []
        for key, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for subkey, subvalue in value.items():
                    lines.append(f"  {subkey}: {subvalue}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

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

        # Aliasing finder plot
        self.af_res.plot(os.path.join(outdir, "aliasing.pdf"))

        # Dealiased timing residual plot
        self.sol.plot(os.path.join(outdir, "timing_residuals.pdf"))

        # Input timing solution
        self.sol.write_input_parfile(
            os.path.join(outdir, "initial.par"), 
            comment="[Input parfile to CHAMPSS dealiasing pipeline]\n" + self.get_summary_text()
        )

        # Dealiased timing solution
        if self.af_res_summary["alias_factor"] != 0:
            self.sol.write_dealiased_parfile(
                os.path.join(outdir, "dealiased.par"), 
                comment="[Dealiased parfile generated by CHAMPSS dealiasing pipeline]\n" + self.get_summary_text()
            )
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
            notes={
                "remark": self.get_remark(), 
                "best_dm": self.get_dm_info()
            }
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

        # Scrunch to the desired sizes
        self.logger.debug("Scrunching stacked data to desired sizes")
        self.payload.stacked_data.scrunch(axis=1)
        self.payload.stacked_data.downsample_to_length(self.n_subints, axis=0, skip_if_shorter=True)
        self.payload.stacked_data.downsample_to_length(self.n_freqs, axis=2, skip_if_shorter=True)
        self.payload.stacked_data.downsample_to_length(self.n_bins, axis=3, skip_if_shorter=True)

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

        # Make a copy of the stacked data
        stacked_data = self.payload.stacked_data.copy()

        # Drop partial subints
        self.logger.debug("Dropping partial subints")
        stacked_data.drop_partial_subints()

        # Find aliasing
        results = AliasingFinder(
            stacked_data=stacked_data,
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

        # Verbose plot
        if self.verbose:
            self.logger.debug("Generating aliasing solution plot")
            sol.plot(savefig=f"{self.workspace}/aliasing_solution.png")

    def commit(self, db=None, overwrite=True):

        # Generate output directory path
        outdir = f"{self.psrdir}/dealiasing"

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