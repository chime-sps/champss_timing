import numpy as np
import os
import matplotlib.pyplot as plt

from ..utils.stats_utils import stats_utils
from ..utils.logger import logger

class BasicChecker:
    def __init__(self, metric_mjds, metric_vals, metric_rcvrs=None, logger=logger(), verbose=False, verbose_title=""):
        """
        Initialize the BasicChecker class.

        Parameters
        ----------
        metric_vals : dict
            A dictionary containing the metric values to be checked.
        """
        
        # Set values
        self.metric_mjds = metric_mjds
        self.metric_vals = metric_vals
        self.metric_rcvrs = metric_rcvrs
        self.logger = logger
        self.verbose = verbose
        self.verbose_title = verbose_title

        # Assume all from the same receiver if not specified
        if self.metric_rcvrs is None:
            self.metric_rcvrs = [None] * len(self.metric_vals)

        # Sort the metric values by MJD
        sorted_indices = np.argsort(self.metric_mjds)
        self.metric_mjds = np.array(self.metric_mjds)[sorted_indices]
        self.metric_vals = np.array(self.metric_vals)[sorted_indices]
        self.metric_rcvrs = np.array(self.metric_rcvrs)[sorted_indices]

    def test(self, z_score_threshold=3, n_samples=365):
        """
        Perform the basic checks on the latest metric values.

        Parameters
        ----------
        z_score_threshold : float, optional
            The z-score threshold for outlier detection (default is 3).

        Returns
        -------
        dict
            A dictionary containing the results of the checks.
        """
        
        # Sanity check if there's any data
        if len(self.metric_vals) == 0:
            return "ok" # No data to check

        # Get the latest metric value
        latest_mjd = self.metric_mjds[-1]
        latest_val = self.metric_vals[-1]
        latest_rcvr = self.metric_rcvrs[-1]

        # Create samples
        samples = self.metric_vals[self.metric_rcvrs == latest_rcvr] # Get samples that has the same receiver as the latest value to avoid bias
        if len(samples) > n_samples:
            samples = samples[-n_samples:]

        if len(samples) < 7:
            return "ok" # Want a larger sample size to get a robust statistic

        # Run the test
        test_thresholds = stats_utils.mad_outlier_thresholds(samples, z_score=z_score_threshold, return_interval=True)

        if self.verbose:
            print(f"Latest MJD: {latest_mjd}, Latest Value: {latest_val}, Receiver: {latest_rcvr}")
            _, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].plot(samples, 'x', label='Metric Values')
            ax[0].axhline(test_thresholds[0], color='red', linestyle='--', label='Lower Threshold')
            ax[0].axhline(test_thresholds[1], color='green', linestyle='--', label='Upper Threshold')
            ax[0].set_ylabel('Metric Value')
            ax[0].set_title(f'Metric Values for Receiver {latest_rcvr} ({self.verbose_title})')
            ax[0].legend()
            ax[1].hist(samples, bins=30, alpha=0.7, label='Sample Distribution')
            ax[1].axvline(latest_val, color='orange', linestyle='--', label='Latest Value')
            ax[1].axvline(test_thresholds[0], color='red', linestyle='--', label='Lower Threshold')
            ax[1].axvline(test_thresholds[1], color='green', linestyle='--', label='Upper Threshold')
            ax[1].set_xlabel('Metric Value')
            ax[1].set_ylabel('Frequency')
            ax[1].set_title('Sample Distribution')
            ax[1].legend()
            plt.tight_layout()
            plt.show()
        
        # Check if the latest value is an outlier
        if latest_val < test_thresholds[0]:
            return "too_low"
        elif latest_val > test_thresholds[1]:
            return "too_high"

        return "ok"
    
    def test_95_997(self, n_samples=365):
        """
        Perform the basic checks on the latest metric values with a 95% and 99.7% confidence interval.
        """
        
        return self.test(z_score_threshold=1.96, n_samples=n_samples), self.test(z_score_threshold=3, n_samples=n_samples)

class Main:
    def __init__(self, db_hdl, basic_checker_results, psr_id, psr_dir, logger=logger()):
        """
        Initialize the Main class.

        Parameters
        ----------
        db : database
            The database object.
        config : dict
            The configuration dictionary.
        """

        # Get logger
        self.logger = logger

        # Get database handler
        self.db_hdl = db_hdl

        # Get pulsar info
        self.psr_id = psr_id
        self.psr_dir = psr_dir

        # Get basic checker results
        self.basic_checker_results = basic_checker_results

        # Get timing info
        self.timing_info = self.db_hdl.get_all_timing_info()

        # Get basic metric information
        self.metric_residuals = {"mjds": [], "vals": [], "rcvrs": []}
        self.metric_toa_errs = {"mjds": [], "vals": [], "rcvrs": []}
        self.metric_snrs = {"mjds": [], "vals": [], "rcvrs": []}
        self.metric_chi2rs = {"mjds": [], "vals": []}
        if len(self.timing_info) > 0:
            toas_mjd_idxed = {}
            for toa_entry in self.db_hdl.get_all_toas():
                toas_mjd_idxed[np.round(toa_entry["toa"], 5)] = toa_entry
            ## metric: residuals
            self.metric_residuals["mjds"] = self.timing_info[-1]["notes"]["fitted_mjds"]
            self.metric_residuals["vals"] = self.timing_info[-1]["residuals"]["val"]
            for mjd in self.metric_residuals["mjds"]:
                if round(mjd, 5) in toas_mjd_idxed:
                    self.metric_residuals["rcvrs"].append(toas_mjd_idxed[round(mjd, 5)]["notes"]["rcvr"])
                else:
                    self.metric_residuals["rcvrs"].append(None)
            ## metric: toa errors
            for toa_entry in self.db_hdl.get_all_toas():
                self.metric_toa_errs["mjds"].append(toa_entry["toa"])
                self.metric_toa_errs["vals"].append(toa_entry["toa_err"])
                self.metric_toa_errs["rcvrs"].append(toa_entry["notes"]["rcvr"])
            ## metric: snr
            self.metric_snrs["mjds"] = self.timing_info[-1]["obs_mjds"]
            for file in self.timing_info[-1]["files"]:
                this_ar_entry = self.db_hdl.get_archive_info_by_filename(file)
                self.metric_snrs["vals"].append(this_ar_entry["psr_snr"])
                self.metric_snrs["rcvrs"].append(this_ar_entry["notes"]["rcvr"])
            ## metric: chi2 reduced
            for timing in self.timing_info:
                # metric: chi2_reduced
                self.metric_chi2rs["vals"].append(timing["chi2_reduced"])
                self.metric_chi2rs["mjds"].append(np.max(timing["obs_mjds"]))
    
    def check(self):
        """
        Main entry point for the monitoring.
        """

        results = {"chi2r": {}, "residual": {}, "snr": {}, "fitting_failure": {}}

        for key in results:
            results[key] = getattr(self, f"check_{key}")()
        
        return results
    
    def check_chi2r(self):
        self.logger.debug("Checking chi2r...")

        # Basic distribution check
        if len(self.timing_info) > 180: # Need a longer time span to wait for the chi2r to stabilize
            bckr = BasicChecker(
                metric_mjds=self.metric_chi2rs["mjds"], 
                metric_vals=self.metric_chi2rs["vals"],
                metric_rcvrs=None,
                verbose_title="Chi2 Reduced (no receiver)"
            )
            bckr_res95, bckr_res997 = bckr.test_95_997(n_samples=30)
            if bckr_res997 != "ok":
                return {"level": 2, "id": "chi2r_very_sudden_increase", "message": f"Chi2r is out of 3-sigma range of all chi2rs in the last 30 days ({bckr_res997}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
            elif bckr_res95 != "ok":
                return {"level": 1, "id": "chi2r_sudden_increase", "message": f"Chi2r is out of 2-sigma range of all chi2rs in the last 30 days ({bckr_res95}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        else:
            bckr = BasicChecker(
                metric_mjds=self.metric_chi2rs["mjds"],
                metric_vals=self.metric_chi2rs["vals"],
                metric_rcvrs=None,
                verbose_title="Chi2 Reduced (no receiver)"
            )
            _, bckr_res997 = bckr.test_95_997(n_samples=7) # only check for very sudden increase
            if bckr_res997 != "ok":
                return {"level": 2, "id": "chi2r_very_sudden_increase", "message": f"Chi2r is out of 3-sigma range of all chi2rs in the last 7 days ({bckr_res997}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        
        # check if chi2r keeps increasing in the last 5 days
        if len(self.metric_chi2rs["vals"]) >= 8:
            chi2rs_7days = self.metric_chi2rs["vals"][-7:]
            if np.all(np.diff(chi2rs_7days) > 0):
                return {"level": 1, "id": "chi2r_keeps_increasing", "message": "Chi2r keeps increasing in the last 7 days.", "attachments": ["%DIAGNOSTIC_PLOT%"]}
            
        return {"level": 0, "id": "chi2r_ok", "message": "Chi2r is normal.", "attachments": []}
    
    def check_residual(self):
        self.logger.debug("Checking timing residuals...")

        # Check TOA_errors
        bckr = BasicChecker(
            metric_mjds=self.metric_toa_errs["mjds"],
            metric_vals=self.metric_toa_errs["vals"],
            metric_rcvrs=self.metric_toa_errs["rcvrs"],
            verbose_title="TOA Errors (residuals_err)"
        )
        bckr_res95, bckr_res997 = bckr.test_95_997(n_samples=365)
        if bckr_res997 != "ok":
            return {"level": 2, "id": "toa_error_very_sudden_increase", "message": f"TOA error is out of 3-sigma range of all TOA errors in the last 14 days ({bckr_res997}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif bckr_res95 != "ok":
            return {"level": 1, "id": "toa_error_sudden_increase", "message": f"TOA error is out of 2-sigma range of all TOA errors in the last 14 days ({bckr_res95}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}

        # Check residuals
        bckr = BasicChecker(
            metric_mjds=self.metric_residuals["mjds"],
            metric_vals=self.metric_residuals["vals"],
            metric_rcvrs=self.metric_residuals["rcvrs"],
            verbose_title="Residuals (residuals)"
        )
        bckr_res95, bckr_res997 = bckr.test_95_997(n_samples=365)
        if bckr_res997 != "ok":
            return {"level": 2, "id": "residual_very_sudden_increase", "message": f"Residual is out of 3-sigma range of all residuals in the last 14 days ({bckr_res997}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif bckr_res95 != "ok":
            return {"level": 1, "id": "residual_sudden_increase", "message": f"Residual is out of 2-sigma range of all residuals in the last 14 days ({bckr_res95}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}

        return {"level": 0, "id": "residual_ok", "message": "Residual is normal.", "attachments": []}

    def check_snr(self):
        self.logger.debug("Checking SNR...")

        # Basic distribution check
        bckr = BasicChecker(
            metric_mjds=self.metric_snrs["mjds"],
            metric_vals=self.metric_snrs["vals"],
            metric_rcvrs=self.metric_snrs["rcvrs"],
            verbose_title="SNR"
        )
        bckr_res95, bckr_res997 = bckr.test_95_997(n_samples=365)
        if bckr_res997 != "ok":
            return {"level": 2, "id": "snr_very_sudden_change", "message": f"SNR is out of 3-sigma range of all SNRs in the last 14 days ({bckr_res997}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif bckr_res95 != "ok":
            return {"level": 1, "id": "snr_sudden_change", "message": f"SNR is out of 2-sigma range of all SNRs in the last 14 days ({bckr_res95}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        
        # Check if SNR keeps increasing in the last 7 days
        if len(self.metric_snrs["vals"]) >= 8:
            snrs_7days = self.metric_snrs["vals"][-7:]
            if np.all(np.diff(snrs_7days) > 0):
                return {"level": 1, "id": "snr_keeps_increasing", "message": "SNR keeps increasing in the last 7 days.", "attachments": ["%DIAGNOSTIC_PLOT%"]}
            elif np.all(np.diff(snrs_7days) < 0):
                return {"level": 1, "id": "snr_keeps_decreasing", "message": "SNR keeps decreasing in the last 7 days.", "attachments": ["%DIAGNOSTIC_PLOT%"]}

        return {"level": 0, "id": "snr_ok", "message": "SNR is normal.", "attachments": []}

    def check_fitting_failure(self):
        self.logger.debug("Checking fitting status...")

        if len(self.timing_info) > 3:
            if ("FITTING_FAILED" in self.timing_info[-1]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-2]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-3]["notes"]["remark"]):
                return {"level": 2, "id": "fitting_failed", "message": "All PINT fittings failed in last 3 days. ", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        
        return {"level": 0, "id": "fitting_ok", "message": "Fitting status is normal.", "attachments": []}

    def _mad_outlier_test(self, samples, point):
        """
        Shortcut for stats_utils.mad_outlier_test
        """

        return stats_utils.mad_outlier_test(samples, point)