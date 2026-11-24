import os
import time
import random
import numpy as np

from ..utils.logger import logger
from ..utils.utils import utils
from ..tools.dealiasing import (
    Dealiasing, 
    estimate_minimal_obs_for_subint_snr_from_database, 
    get_files_for_dealiasing_pipeline
)

class dealias:
    def __init__(self, psr_dir, db_hdl, archive_files, jumps, potential_fit_params, n_subints=8, min_snr_per_subint=5.0, max_n_files=120, n_bins=128, n_freqs=256, smooth=0, recent_threshold=90 * 24 * 3600, workspace="/tmp", cleanup=True, n_pools=1, logger=logger()):
        self.psr_dir = psr_dir
        self.db_hdl = db_hdl
        self.jumps = jumps
        self.potential_fit_params = potential_fit_params
        self.n_subints = n_subints
        self.min_snr_per_subint = min_snr_per_subint
        self.n_bins = n_bins
        self.n_freqs = n_freqs
        self.smooth = smooth
        self.recent_threshold = recent_threshold
        self.workspace = workspace
        self.max_n_files = max_n_files
        self.cleanup = cleanup
        self.logger = logger
        self.n_pools = n_pools

        # Organize archive files by backend
        ar_list_by_backend = {}
        for mjd in archive_files:
            for f in archive_files[mjd]:
                if f["rcvr"] not in ar_list_by_backend:
                    ar_list_by_backend[f["rcvr"]] = []

                ar_list_by_backend[f["rcvr"]].append({
                    "location": f["path"],
                    "backend": f["rcvr"], # sorry but "rcvr" and "backend" are referring to the same thing in the pipeline... 
                    "mjd": f["mjd"]
                })

        # Get the minimal number of observation required to achieve the desired SNR per subint
        min_n_obs = estimate_minimal_obs_for_subint_snr_from_database(
            self.db_hdl, 
            self.n_subints, 
            self.min_snr_per_subint
        )

        # Get the best list of archive files to use
        self.rcvr, self.archive_files, self.n_files_required = get_files_for_dealiasing_pipeline(
            min_n_obs, 
            ar_list_by_backend
        )

        # Sort files by MJD
        self.archive_files.sort(key=lambda x: x["mjd"])

        self.logger.info(f"Selected receiver for generating dealias diagnostics: {self.rcvr}")

    def get_minimal_required_file(self):
        # Get mean snr for each archive file
        mean_snr = np.mean([f["snr"] for f in self.archive_files])

        # Calculate the required number of files to reach the required snr per subint
        # Assuming the snr adds in quadrature for multiple observations / each subints: 
        #   SNR_subint = SNR_per_obs * sqrt(N_obs) / sqrt(N_subints)
        n_obs = (
            self.min_snr_per_subint * np.sqrt(self.n_subints) / mean_snr
        )**2

        # Ensure a minimum number of observations to robustly eliminate artifacts in single observation
        if n_obs < 5:
            n_obs = 5

        return int(np.ceil(n_obs))

    def is_timing_model_robust(self):
        # Fetech the last timing information
        last_timing_info = self.db_hdl.get_last_timing_info()
        if last_timing_info is None:
            return False

        # Check if the timing model is robust based on some criteria
        unfreeze_params = last_timing_info["unfreeze_params"]
        if len(self.potential_fit_params) > len(unfreeze_params):
            return False

        # Check if the timing baseline is long enough to break spindown-position degeneracies
        fitted_mjds = last_timing_info["notes"]["fitted_mjds"]
        if np.max(fitted_mjds) - np.min(fitted_mjds) < 90:
            return False

        return True

    def is_dealiased_recently(self):
        # Fetch the last dealias information
        last_dealias_info = self.db_hdl.get_last_dealias_history()
        if last_dealias_info is None:
            return False

        # Check if dealiasing directory exists
        dealias_dir = f"{self.psr_dir}/dealias"
        if not os.path.exists(dealias_dir):
            return False

        # Check if the last dealias was recent enough
        last_dealias_time = last_dealias_info["timestamp"]
        time_since_last_dealias = time.time() - last_dealias_time
        if time_since_last_dealias > self.recent_threshold + random.uniform(0, max([7 * 24 * 3600, self.recent_threshold * 0.05])):  # Add random jitter up to 7 days to avoid too many dealias runs on the same day causing the processing delayed or congested. 
            return False
            
        return True

    def run(self):
        # Check if the timing model is robust
        if not self.is_timing_model_robust():
            self.logger.info("Timing model is not robust. Skipping dealias step.")
            return

        # Check if the data has been dealias recently
        if self.is_dealiased_recently():
            self.logger.info("Data has been dealias recently. Skipping dealias step.")
            return

        # Check whether enough files
        if len(self.archive_files) < self.n_files_required:
            self.logger.debug(f"Not enough archive files. Required: {self.n_files_required}, available: {len(self.archive_files)}")
            return
        
        # Cap at twice as much of the required number of files
        if len(self.archive_files) > 2 * self.n_files_required:
            self.archive_files = self.archive_files[:2 * self.n_files_required]
            self.logger.info(f"Capped archive files to twice the required number: 2 x {self.n_files_required} = {2 * self.n_files_required}")

        # Cap at max_n_files to avoid excessive processing
        if len(self.archive_files) > self.max_n_files:
            self.archive_files = self.archive_files[:self.max_n_files]
            self.logger.info(f"Capped archive files to {self.max_n_files} since there are still too many files to be processed.")

        # Run dealiasing pipeline
        self.logger.debug(f"Starting dealiasing with {len(self.archive_files)} archive files.")
        with Dealiasing(
            psrdir=self.psr_dir, 
            ar_list=self.archive_files,
            parfile=os.path.join(self.psr_dir, "pulsar.par"),
            jumps=self.jumps, 
            n_subints=self.n_subints, 
            n_bins=self.n_bins, 
            n_freqs=self.n_freqs, 
            workspace=self.workspace,
            ncpus=self.n_pools, 
            logger=self.logger
        ) as adg:
            # Optimize DM
            adg.optimize_dm()

            # Resolve aliasing
            adg.resolve_aliasing()

            # Get dealias results
            dealias_results = {
                "n_stacked": adg.payload.af_res_summary["n_stacked"],
                "alias_factor": adg.payload.af_res_summary["alias_factor"],
                "snr_stacked": adg.payload.af_res_summary["snr_stacked"],
                "remark": adg.payload.get_remark()
            }

            # Commit changes
            adg.commit(db=self.db_hdl)
            
        return dealias_results