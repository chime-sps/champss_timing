import numpy as np
import os

from ..utils.stats_utils import stats_utils
from ..utils.logger import logger

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
    
    def check(self):
        """
        Main entry point for the monitoring.
        """

        results = {}

        results["chi2r"] = self.check_chi2r()
        results["residual"] = self.check_residual()
        results["snr"] = self.check_snr()
        results["fitting"] = self.check_fitting_failure()
        
        return results
    
    def check_chi2r(self):
        self.logger.debug("Checking chi2r...")

        if len(self.timing_info) < 7:
            return {"level": 0, "id": "too_few_toas", "message": "At least 7 days of timing info to get a reliable chi2r.", "attachments": []}

        chi2rs = []
        for timing in self.timing_info:
            chi2rs.append(timing["chi2_reduced"])

        if len(chi2rs) == 0:
            return {"level": 0, "id": "no_toas", "message": "No timing info available.", "attachments": []}
        
        # check if chi2r keeps increasing in the last 5 days
        if len(chi2rs) >= 7:
            if chi2rs[-1] > chi2rs[-2] and chi2rs[-2] > chi2rs[-3] and chi2rs[-3] > chi2rs[-4] and chi2rs[-4] > chi2rs[-5] and chi2rs[-5] > chi2rs[-6]:
                return {"level": 1, "id": "chi2r_keeps_increasing", "message": "Chi2r keeps increasing in the last 7 days.", "attachments": ["%DIAGNOSTIC_PLOT%"]}
            
        # check if the last chi2r is above 1.5 of the median in the last 7 days
        mean_chi2r = sum(chi2rs[-7:]) / 7
        std_chi2r = np.std(chi2rs[-7:])
        if chi2rs[-1] > 7 * std_chi2r + mean_chi2r:
            return {"level": 2, "id": "chi2r_very_sudden_increase", "message": f"The last chi2r is above mean + 7 * std in the last 7 days ({chi2rs[-1]:.3f} > mean={mean_chi2r:.3f} + 7 * std={7 * std_chi2r:.3f}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif chi2rs[-1] > 3 * std_chi2r + mean_chi2r:
            return {"level": 1, "id": "chi2r_sudden_increase", "message": f"The last chi2r is above mean + 3 * std in the last 7 days. ({chi2rs[-1]:.3f} > mean={mean_chi2r:.3f} + 3 * std={3 * std_chi2r:.3f}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}

        # # check the reduced chi2r of the last day
        # if len(chi2rs) >= 14:
        #     if self._mad_outlier_test(chi2rs[-14:], chi2rs[-1]) > 3: # 99.7% confidence
        #         if chi2rs[-1] > np.median(chi2rs[-14:]):
        #             return {"level": 2, "message": f"The last chi2r is out of 3-sigma range of chi2rs in the last 14 days ({chi2rs[-1]:.3f} > med={np.median(chi2rs[-14:]):.3f})."}
        #     elif self._mad_outlier_test(chi2rs[-14:], chi2rs[-1]) > 1.96: # 95% confidence
        #         if chi2rs[-1] > np.median(chi2rs[-14:]):
        #             return {"level": 1, "message": f"The last chi2r is out of 2-sigma range of chi2rs in the last 14 days ({chi2rs[-1]:.3f} > med={np.median(chi2rs[-14:]):.3f})."}
        
        return {"level": 0, "id": "chi2r_ok", "message": "Chi2r is normal.", "attachments": []}
    
    def check_residual(self):
        self.logger.debug("Checking timing residuals...")

        if len(self.timing_info) < 7:
            return {"level": 0, "id": "too_few_toas", "message": "At least 7 days of timing info to get a reliable residual.", "attachments": []}

        resid_val = self.timing_info[-1]["residuals"]["val"]
        resid_err = self.timing_info[-1]["residuals"]["err"]
        fitted_params = self.timing_info[-1]["fitted_params"]

        # check if error of residuals is too large
        last_error_in_phase = resid_err[-1] * 1e-6 * self.timing_info[-1]["fitted_params"]["F0"]
        if self._mad_outlier_test(resid_err, resid_err[-1]) > 3: # 99.7% confidence
            if last_error_in_phase > 0.01: # champss or chime/pulsar timing error is usually around 1%
                return {"level": 2,  "id": "toa_error_very_sudden_increase", "message": f"Error of the last residual is out of 3-sigma range of all residuals ({last_error_in_phase:.3f} in phase).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif self._mad_outlier_test(resid_err, resid_err[-1]) > 1.96: # 95% confidence
            if last_error_in_phase > 0.01: # champss or chime/pulsar timing error is usually around 1%
                return {"level": 1, "id": "toa_error_sudden_increase", "message": f"Error of the last residual is out of 2-sigma range of all residuals ({last_error_in_phase:.3f} in phase).", "attachments": ["%DIAGNOSTIC_PLOT%"]}

        # check if the last residual is too large (only do this when error looks good)
        last_resid_in_phase = resid_val[-1] * 1e-6 * self.timing_info[-1]["fitted_params"]["F0"]
        if self._mad_outlier_test(resid_val, resid_val[-1]) > 3: # 99.7% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 2, "id": "residual_very_sudden_increase", "message": f"The last residual is out of 3-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        elif self._mad_outlier_test(resid_val, resid_val[-1]) > 1.96: # 95% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 1, "id": "residual_sudden_increase", "message": f"The last residual is out of 2-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "attachments": ["%DIAGNOSTIC_PLOT%"]}
        
        return {"level": 0, "id": "residual_ok", "message": "Residual is normal.", "attachments": []}

    def check_snr(self):
        self.logger.debug("Checking SNR...")

        if len(self.timing_info) < 7:
            return {"level": 0, "id": "too_few_toas", "message": "At least 7 days of timing info to get a reliable SNR.", "attachments": []}

        # Get files from the last timing
        archive_ids = self.timing_info[-1]["files"]

        # Get SNR for each file
        snrs = []
        for this_id in archive_ids:
            # Get the file info
            this_info  = self.db_hdl.get_archive_info_by_filename(this_id)

            # Get the SNR
            snrs.append(this_info["psr_snr"])

        # Sanity check
        if len(snrs) == 0:
            return {"level": 0, "id": "no_data", "message": "No SNR info available.", "attachments": []}

        # Check if the last SNR is exceptionally high
        if self._mad_outlier_test(snrs, snrs[-1]) > 3: # 99.7% confidence
            median = np.median(snrs)
            if snrs[-1] > median:
                return {"level": 2, "id": "snr_sudden_high", "message": f"The signal from the last observation is exceptionally HIGH ({snrs[-1]:.3f} > {median:.3f} + 3 * {np.std(snrs):.3f}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}
            elif snrs[-1] < median:
                return {"level": 2, "id": "snr_sudden_low", "message": f"The signal from the last observation is exceptionally LOW ({snrs[-1]:.3f} < {median:.3f} - 3 * {np.std(snrs):.3f}).", "attachments": ["%DIAGNOSTIC_PLOT%"]}

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