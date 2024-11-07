from .database import database

import numpy as np
import os

class champss_checker:
    def __init__(self, psr_dir, db_hdl, noti_hdl, psr_id="psr_id_not_provided"):
        self.psr_dir = psr_dir
        self.db_hdl = db_hdl
        self.psr_id = psr_id
        self.diagnostic_plot = psr_dir + "/champss_diagnostic.pdf"
        self.noti_hdl = noti_hdl
        self.timing_info = self.db_hdl.get_all_timing_info()

    def check_chi2r(self):
        chi2rs = []
        for timing in self.timing_info:
            chi2rs.append(timing["chi2_reduced"])

        if len(chi2rs) == 0:
            return {"level": 0, "message": "No timing info available."}
        
        # check if chi2r keeps increasing in the last 3 days
        if len(chi2rs) >= 3:
            if chi2rs[-1] > chi2rs[-2] and chi2rs[-2] > chi2rs[-3]:
                return {"level": 1, "message": "Chi2r keeps increasing in the last 3 days."}
            
        # check if the last chi2r is above 1.5 of the median in the last 7 days
        median_chi2r = sum(chi2rs[-7:]) / 7
        if chi2rs[-1] > 1.5 * median_chi2r:
            return {"level": 2, "message": "The last chi2r is above 1.5 of the median in the last 7 days."}
        elif chi2rs[-1] > 1.25 * median_chi2r:
            return {"level": 1, "message": "The last chi2r is above 1.25 of the median in the last 7 days."}
        
        return {"level": 0, "message": "Chi2r is normal."}
    
    def check_residual(self):
        resid_val = self.timing_info[-1]["residuals"]["val"]
        resid_err = self.timing_info[-1]["residuals"]["err"]
        fitted_params = self.timing_info[-1]["fitted_params"]

        # check if error of residuals is too large
        if resid_err[-1] >= 10000:
            return {"level": 2, "message": "Error of the last is larger than 10000 us (error = {} us).".format(resid_err[-1])}

        # check if the last residual is too large
        if resid_val[-1] >= np.median(resid_val) * 1.5:
            return {"level": 2, "message": "The last residual is larger than 1.5 of the median in the last 7 days."}
        elif resid_val[-1] >= np.median(resid_val) * 1.25:
            return {"level": 1, "message": "The last residual is larger than 1.25 of the median in the last 7 days."}
        
        return {"level": 0, "message": "Residual is normal."}

    def check_fitting_failure(self):
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
            self.noti_hdl.send_message("=== Checker Warning === \n" + check_result["message"] + "\n=======================", psr_id = self.psr_id)
        elif check_result["level"] == 2:
            self.noti_hdl.send_urgent_message("=== Checker Important Warning === \n" + check_result["message"] + "\n=======================", psr_id = self.psr_id)
    
    def check(self):
        print("Checking fitting status...")
        self.send_notification(self.check_fitting_failure())

        print("Checking timing residuals...")
        self.send_notification(self.check_chi2r())

        print("Checking chi2r...")
        self.send_notification(self.check_residual())

        print("Sending notification...")
        self.noti_hdl.send_success_message("Timing finished. ", psr_id = self.psr_id)
        
        if os.path.exists(self.diagnostic_plot):
            self.noti_hdl.send_file(self.diagnostic_plot, psr_id = self.psr_id)
        else:
            self.noti_hdl.send_urgent_message("Checker Unexpected: \nDiagnostic plot is missing.", psr_id = self.psr_id)

        print("Checking finished.")