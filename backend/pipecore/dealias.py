import os
import time
import random
import numpy as np

from ..tools.alias_utils import alias_utils
from ..utils.logger import logger
from ..utils.utils import utils

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

        # Determine the rcvr to use
        rcvr_counts = {}
        self.rcvr = None
        for mjd in archive_files:
            rcvr = archive_files[mjd][0]["rcvr"]
            if rcvr not in rcvr_counts:
                rcvr_counts[rcvr] = 0
            rcvr_counts[rcvr] += 1

        if rcvr_counts:
            self.rcvr = max(rcvr_counts, key=rcvr_counts.get)
            self.logger.info(f"Selected receiver for generating dealias diagnostics: {self.rcvr}")
            
        # Get snr and filter each archive file
        self.archive_files = []
        for mjd in archive_files:
            if archive_files[mjd][0]["rcvr"] != self.rcvr:
                continue

            this_arid = utils.get_archive_id(archive_files[mjd][0]["path"])
            this_info = self.db_hdl.get_archive_info_by_filename(this_arid)
            
            if len(this_info) == 0:
                self.logger.warning(f"No archive info found for {this_arid}. Ignoring. ")
                continue
            self.archive_files.append({
                "mjd": mjd,
                "arid": this_arid,
                "path": archive_files[mjd][0]["path"],
                "rcvr": archive_files[mjd][0]["rcvr"],
                "snr": this_info["psr_snr"]
            })

        # Sort files by MJD
        self.archive_files.sort(key=lambda x: x["mjd"])

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
        n_files_required = self.get_minimal_required_file()
        if len(self.archive_files) < n_files_required:
            self.logger.error(f"Not enough archive files. Required: {n_files_required}, available: {len(self.archive_files)}")
            return
        
        # Cap at twice as much of the required number of files
        if len(self.archive_files) > 2 * n_files_required:
            self.archive_files = self.archive_files[:2 * n_files_required]
            self.logger.info(f"Capped archive files to twice the required number: 2 x {n_files_required} = {2 * n_files_required}")

        # Cap at 120 files to avoid excessive processing
        if len(self.archive_files) > self.max_n_files:
            self.archive_files = self.archive_files[:self.max_n_files]
            self.logger.info(f"Capped archive files to {self.max_n_files} since there are still too many files to be processed.")

        # Run dealiasing pipeline
        self.logger.debug(f"Starting dealiasing with {len(self.archive_files)} archive files.")
        with alias_utils(
            psrdir = self.psr_dir, 
            ar_list = [{"location": f_info["path"], "backend": f_info["rcvr"]} for f_info in self.archive_files],
            parfile = os.path.join(self.psr_dir, "pulsar.par"),
            jumps=self.jumps, 
            n_subints=self.n_subints, 
            n_bins=self.n_bins, 
            n_freqs=self.n_freqs, 
            workspace=self.workspace, 
            cleanup=self.cleanup, 
            mode="auto", 
            n_pools=self.n_pools, 
            logger=self.logger
        ) as au:
            # Get alias factor
            au.cf_get_alias_factor(smooth_sigma=self.smooth)

            # Dealias
            au_summary = au.dealias()

            # Get dealias results
            dealias_results = {
                "psr_id": au_summary["psr_id"],
                "n_stacked": au_summary["n_stacked"],
                "alias_factor": au_summary["alias_factor"],
                "snr_stacked": au_summary["snr_stacked"],
                "remark": au_summary["notes"]["remark"],
            }

            # Commit changes to database
            au.commit(db_hdl=self.db_hdl)
            
        return dealias_results