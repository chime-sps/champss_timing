from .datastores.database import database

import numpy as np
import os
from scipy.stats import median_abs_deviation

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

    def __init__(self, psr_dir, db_hdl=None, noti_hdl=None, psr_id="psr_id_not_provided"):
        self.psr_dir = psr_dir
        self.db_hdl = db_hdl
        self.psr_id = psr_id
        self.diagnostic_plot = psr_dir + "/champss_diagnostic.pdf"
        self.noti_hdl = noti_hdl
        if self.db_hdl == None:
            self.db_hdl = database(psr_dir + "/champss_timing.sqlite3.db")
        self.timing_info = self.db_hdl.get_all_timing_info()

    def check_chi2r(self):
        print("Checking chi2r...")

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
        print("Checking timing residuals...")

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
        print(last_resid_in_phase)
        print(self._mad_outlier_test(resid_val, resid_val[-1]))
        print(self._mad_outlier_test(resid_val, resid_val[-1]) > 3)
        print(last_resid_in_phase > 0.015)
        if self._mad_outlier_test(resid_val, resid_val[-1]) > 3: # 99.7% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 2, "message": f"The last residual is out of 3-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "trigger_glitch_utils": True}
        elif self._mad_outlier_test(resid_val, resid_val[-1]) > 1.96: # 95% confidence
            if np.abs(last_resid_in_phase) > 0.015: # could be timing noise
                return {"level": 1, "message": f"The last residual is out of 2-sigma range of all residuals ({last_resid_in_phase:.3f} in phase). ", "trigger_glitch_utils": True}
        
        return {"level": 0, "message": "Residual is normal.", "trigger_glitch_utils": False}

    def check_fitting_failure(self):
        print("Checking fitting status...")

        if len(self.timing_info) > 3:
            if ("FITTING_FAILED" in self.timing_info[-1]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-2]["notes"]["remark"]
            and "FITTING_FAILED" in self.timing_info[-3]["notes"]["remark"]):
                return {"level": 2, "message": "All PINT fittings failed in last 3 days. "}
        
        return {"level": 0, "message": "Fitting status is normal."}
    
    def send_notification(self, check_result):
        if check_result["level"] == 0:
            return 
        elif check_result["level"] == 1:
            self.noti_hdl.send_message(f"*Checker Warning for {self.psr_id}*: `" + check_result["message"] + "`")
        elif check_result["level"] == 2:
            self.noti_hdl.send_urgent_message(f"*Checker Important Warning for {self.psr_id}*: `" + check_result["message"] + "`")
        
        if os.path.exists(self.diagnostic_plot):
            self.noti_hdl.send_file(self.diagnostic_plot)
        else:
            self.noti_hdl.send_urgent_message(f"*Checker Unexpected ({self.psr_id})*: `Diagnostic plot is missing.`")
    
    def check(self, send_noti=False):
        print("Checking...")

        summary = {
            "fitting_failure": self.check_fitting_failure(),
            "chi2r": self.check_chi2r(),
            "residual": self.check_residual()
        }
        
        if send_noti and self.noti_hdl is not None:
            print("Sending notification...")
            self.send_notification(summary["fitting_failure"])
            self.send_notification(summary["residual"])
            self.send_notification(summary["chi2r"])
            
            if summary["residual"]["trigger_glitch_utils"]:
                try:
                    self.noti_hdl.send_urgent_message(f"*Checker Important Warning ({self.psr_id})*: `Glitch-like event detected. Please check the following glitch diagnostic plot.`")
                    from .tools.glitch_utils import glitch_utils
                    with glitch_utils(db_hdl=self.db_hdl) as gu:
                        gu.estimate_glitch(savefig=f"glitch_diagnostic__{self.psr_id}.pdf")
                    if os.path.exists(f"glitch_diagnostic__{self.psr_id}.pdf"):
                        self.noti_hdl.send_file(f"glitch_diagnostic__{self.psr_id}.pdf")
                    else:
                        raise Exception("Glitch utils failed to generate diagnostic plot.")
                except Exception as e:
                    self.noti_hdl.send_urgent_message(f"*Checker Unexpected ({self.psr_id})*: `Glitch utils triggered but failed to generate diagnostic plot. {e}`")

        print("Checking finished.")

        return summary

    def _mad_outlier_test(self, samples, point):
        """
        This function is used to determine if a point is an outlier or not. 
        It is used in the MAD outlier test. 

        Parameters
        ----------
        samples : array_like
            The sample of values to test the point against. 
        point : float
            The point to test. 

        Returns
        -------
        float
            The z-score of the point.
        """

        # get mad and median
        mad = median_abs_deviation(samples)
        median = np.median(samples)

        # calculate the z-score
        z_score = np.abs(point - median) / mad

        return z_score