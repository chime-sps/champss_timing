from .datastores.database import database

import numpy as np
import os
from .utils.stats_utils import stats_utils
from .utils.logger import logger
from .tools.glitch_utils import glitch_utils

class champss_checker:
    """
    CHAMPSS Timing Pipeline Checker
    This class is used to check the status of a timing result. 

    Parameters
    ----------
    psr_dir : str
        The directory of the pulsar.
    db_hdl : database
        The database handler. If None, a new database handler will be created.
    noti_hdl : Notifier
        The notifier handler. If None, no notification will be sent.
    psr_id : str
        The identifier of the pulsar. If None, the ID will be extracted from the directory name.
    """

    def __init__(self, psr_dir, db_hdl=None, noti_hdl=None, psr_id="psr_id_not_provided", logger=logger()):
        # Initial parameters and utilities
        self.psr_dir = psr_dir
        self.db_hdl = db_hdl
        self.psr_id = psr_id
        self.diagnostic_plot = psr_dir + "/champss_diagnostic.pdf"
        self.noti_hdl = noti_hdl
        self.logger = logger

        # Check if db_hdl is None
        if self.db_hdl == None:
            self.db_hdl = database(psr_dir + "/champss_timing.sqlite3.db")
        
        # Get timing info
        self.timing_info = self.db_hdl.get_all_timing_info()

    def check_chi2r(self):
        self.logger.debug("Checking chi2r...")

        if len(self.timing_info) < 7:
            return {"level": 0, "message": "At least 7 days of timing info to get a reliable chi2r."}

        chi2rs = []
        for timing in self.timing_info:
            chi2rs.append(timing["chi2_reduced"])

        if len(chi2rs) == 0:
            return {"level": 0, "message": "No timing info available."}
        
        # check if chi2r keeps increasing in the last 5 days
        if len(chi2rs) >= 7:
            if chi2rs[-1] > chi2rs[-2] and chi2rs[-2] > chi2rs[-3] and chi2rs[-3] > chi2rs[-4] and chi2rs[-4] > chi2rs[-5] and chi2rs[-5] > chi2rs[-6]:
                return {"level": 1, "message": "Chi2r keeps increasing in the last 7 days."}
            
        # check if the last chi2r is above 1.5 of the median in the last 7 days
        mean_chi2r = sum(chi2rs[-7:]) / 7
        std_chi2r = np.std(chi2rs[-7:])
        if chi2rs[-1] > 7 * std_chi2r + mean_chi2r:
            return {"level": 2, "message": f"The last chi2r is above mean + 7 * std in the last 7 days ({chi2rs[-1]:.3f} > mean={mean_chi2r:.3f} + 7 * std={7 * std_chi2r:.3f})."}
        elif chi2rs[-1] > 3 * std_chi2r + mean_chi2r:
            return {"level": 1, "message": f"The last chi2r is above mean + 3 * std in the last 7 days. ({chi2rs[-1]:.3f} > mean={mean_chi2r:.3f} + 3 * std={3 * std_chi2r:.3f})."}

        # # check the reduced chi2r of the last day
        # if len(chi2rs) >= 14:
        #     if self._mad_outlier_test(chi2rs[-14:], chi2rs[-1]) > 3: # 99.7% confidence
        #         if chi2rs[-1] > np.median(chi2rs[-14:]):
        #             return {"level": 2, "message": f"The last chi2r is out of 3-sigma range of chi2rs in the last 14 days ({chi2rs[-1]:.3f} > med={np.median(chi2rs[-14:]):.3f})."}
        #     elif self._mad_outlier_test(chi2rs[-14:], chi2rs[-1]) > 1.96: # 95% confidence
        #         if chi2rs[-1] > np.median(chi2rs[-14:]):
        #             return {"level": 1, "message": f"The last chi2r is out of 2-sigma range of chi2rs in the last 14 days ({chi2rs[-1]:.3f} > med={np.median(chi2rs[-14:]):.3f})."}
        
        return {"level": 0, "message": "Chi2r is normal."}
    
    def check_residual(self):
        self.logger.debug("Checking timing residuals...")

        if len(self.timing_info) < 7:
            return {"level": 0, "message": "At least 7 days of timing info to get a reliable residual.", "trigger_glitch_utils": False}

        resid_val = self.timing_info[-1]["residuals"]["val"]
        resid_err = self.timing_info[-1]["residuals"]["err"]
        fitted_params = self.timing_info[-1]["fitted_params"]

        # check if error of residuals is too large
        last_error_in_phase = resid_err[-1] * 1e-6 * self.timing_info[-1]["fitted_params"]["F0"]
        if self._mad_outlier_test(resid_err, resid_err[-1]) > 3: # 99.7% confidence
            if last_error_in_phase > 0.01: # champss or chime/pulsar timing error is usually around 1%
                return {"level": 2, "message": f"Error of the last residual is out of 3-sigma range of all residuals ({last_error_in_phase:.3f} in phase).", "trigger_glitch_utils": False}
        elif self._mad_outlier_test(resid_err, resid_err[-1]) > 1.96: # 95% confidence
            if last_error_in_phase > 0.01: # champss or chime/pulsar timing error is usually around 1%
                return {"level": 1, "message": f"Error of the last residual is out of 2-sigma range of all residuals ({last_error_in_phase:.3f} in phase).", "trigger_glitch_utils": False}

        # check if the last residual is too large (only do this when error looks good)
        last_resid_in_phase = resid_val[-1] * 1e-6 * self.timing_info[-1]["fitted_params"]["F0"]
        if self._mad_outlier_test(resid_val, resid_val[-1]) > 3: # 99.7% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 2, "message": f"The last residual is out of 3-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "trigger_glitch_utils": True}
        elif self._mad_outlier_test(resid_val, resid_val[-1]) > 1.96: # 95% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 1, "message": f"The last residual is out of 2-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "trigger_glitch_utils": True}
        
        return {"level": 0, "message": "Residual is normal.", "trigger_glitch_utils": False}

    def check_snr(self):
        self.logger.debug("Checking SNR...")

        if len(self.timing_info) < 7:
            return {"level": 0, "message": "At least 7 days of timing info to get a reliable SNR."}

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
            return {"level": 0, "message": "No SNR info available."}

        # Check if the last SNR is exceptionally high
        if self._mad_outlier_test(snrs, snrs[-1]) > 3: # 99.7% confidence
            median = np.median(snrs)
            if snrs[-1] > median:
                return {"level": 2, "message": f"The signal from the last observation is exceptionally HIGH ({snrs[-1]:.3f} > {median:.3f} + 3 * {np.std(snrs):.3f})."}
            elif snrs[-1] < median:
                return {"level": 2, "message": f"The signal from the last observation is exceptionally LOW ({snrs[-1]:.3f} < {median:.3f} - 3 * {np.std(snrs):.3f})."}

        return {"level": 0, "message": "SNR is normal."}

    def check_fitting_failure(self):
        self.logger.debug("Checking fitting status...")

        if len(self.timing_info) > 3:
            if ("FITTING_FAILED" in self.timing_info[-1]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-2]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-3]["notes"]["remark"]):
                return {"level": 2, "message": "All PINT fittings failed in last 3 days. "}
        
        return {"level": 0, "message": "Fitting status is normal."}
    
    def send_notification(self, check_result):
        if check_result["level"] == 0:
            self.logger.debug("Status is normal, no notification sent.", layer=1)
            return 
        elif check_result["level"] == 1:
            msg = f"*Checker Warning for {self.psr_id}*: `" + check_result["message"] + "`"
            self.noti_hdl.send_message(msg)
            self.logger.success(msg, layer=1)
        elif check_result["level"] == 2:
            msg = f"*Checker Important Warning for {self.psr_id}*: `" + check_result["message"] + "`"
            self.noti_hdl.send_urgent_message(msg)
            self.logger.success(msg, layer=1)
        
        if os.path.exists(self.diagnostic_plot):
            self.noti_hdl.send_file(self.diagnostic_plot)
            self.logger.success("Diagnostic plot sent.", layer=1)
        else:
            self.noti_hdl.send_urgent_message(f"*Checker Unexpected ({self.psr_id})*: `Diagnostic plot is missing.`")
            self.logger.warning("Diagnostic plot is missing. Warning sent.", layer=1)
    
    def check(self, send_noti=False):
        self.logger.debug("Checking...")

        # Run checkers
        self.logger.level_up()
        summary = {
            "fitting_failure": self.check_fitting_failure(),
            "chi2r": self.check_chi2r(),
            "residual": self.check_residual(), 
            "snr": self.check_snr()
        }
        self.logger.level_down()
        
        if send_noti and self.noti_hdl is not None:
            self.logger.debug("Sending notification...")
            self.logger.level_up()

            # Send regular notifications
            for key in summary:
                self.logger.debug(key)
                self.send_notification(summary[key])
            
            # Send the glitch notification
            if summary["residual"]["trigger_glitch_utils"]:
                self.logger.debug("Glitch utils triggered.")
                self.logger.level_up()

                try:
                    # Send text notification
                    self.noti_hdl.send_urgent_message(f"*Checker Important Warning ({self.psr_id})*: `Glitch-like event detected. Please check the following glitch diagnostic plot.`")
                    self.logger.success("Glitch notification sent.")

                    # Generate the diagnostic plot
                    with glitch_utils(db_hdl=self.db_hdl, logger=self.logger.copy()) as gu:
                        gu.estimate_glitch(savefig=f"glitch_diagnostic__{self.psr_id}.pdf")

                    # Check if glitch diagnostic plot exists
                    if os.path.exists(f"glitch_diagnostic__{self.psr_id}.pdf"):
                        self.noti_hdl.send_file(f"glitch_diagnostic__{self.psr_id}.pdf")
                        self.logger.success("Glitch diagnostic plot sent.")
                    else:
                        self.logger.warning("Glitch diagnostic plot is missing. Exception raised.")
                        raise Exception("Glitch utils failed to generate diagnostic plot.")
                except Exception as e:
                    self.noti_hdl.send_urgent_message(f"*Checker Unexpected ({self.psr_id})*: `Glitch utils triggered but failed to generate diagnostic plot. {e}`")
                    self.logger.warning("Error while triggering glitch utils. Warning sent.")

                self.logger.level_down()

            self.logger.level_down()

        self.logger.success("Checking finished.")

        return summary

    def _mad_outlier_test(self, samples, point):
        """
        Shortcut for stats_utils.mad_outlier_test
        """

        return stats_utils.mad_outlier_test(samples, point)