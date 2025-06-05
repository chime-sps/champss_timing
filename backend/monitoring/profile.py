import os
import numpy as np

from ..utils.utils import utils
from ..utils.logger import logger
from ..tools.profile_utils import ProfileAnalyzer, ProfilePeaks

class Main:
    def __init__(self, db_hdl, basic_checker_results, psr_id, psr_dir, logger=logger(), temp_dir="/tmp"):
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

        # Get the rcvr of the latest TOA
        toas = db_hdl.get_all_toas()
        latest_rcvr = toas[-1]["notes"]["rcvr"] if toas else None

        # Get profiles
        self.pulse_profiles = {}
        for toa in toas:
            if latest_rcvr is not None and toa["notes"]["rcvr"] != latest_rcvr:
                continue

            this_filename = toa["filename"]
            this_mjd = toa["toa"]
            self.pulse_profiles[this_mjd] = db_hdl.get_archive_info_by_filename(this_filename)["psr_amps"]

        # Sort by key
        self.pulse_profiles = dict(sorted(self.pulse_profiles.items()))

        # Get temp_id for diagnostic plots
        self.temp_id = utils.get_rand_string()
        self.temp_dir = temp_dir + f"/champss_timing_profile_checker__{self.temp_id}"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # ProfileAnalyzer placeholder
        self.paz = None
        
    def check(self):
        """
        Main entry point for the monitoring.
        """

        # Sanity check for number of profiles
        if len(self.pulse_profiles) < 45: # Minimum number of profiles to get 15 binned samples (i.e., minimal binsize is 3 in profile_utils.ProfileAnalyzer -> Get binned profiles)
            self.logger.warning(f"Not enough profiles to run the monitoring for {self.psr_id}. Found {len(self.pulse_profiles)} profiles.")
            return {
                "rms": {"level": 0, "id": "not_enough_profiles", "message": f"Not enough profiles to run the monitoring. ({len(self.pulse_profiles)} < 45)"},
                "chi2": {"level": 0, "id": "not_enough_profiles", "message": f"Not enough profiles to run the monitoring. ({len(self.pulse_profiles)} < 45)"},
                "peak_fluence": {"level": 0, "id": "not_enough_profiles", "message": f"Not enough profiles to run the monitoring. ({len(self.pulse_profiles)} < 45)"}
            }

        # Initialize ProfileAnalyzer
        self.paz = ProfileAnalyzer(list(self.pulse_profiles.values()), mjds=list(self.pulse_profiles.keys()), shift_meth="fourier", verbose=False)

        # Sanity check for binned profiles
        if len(self.paz.binned_profiles) < 15:
            return {
                "rms": {"level": 0, "id": "not_enough_binned_profiles", "message": f"Not enough profiles to run the monitoring or the SNR is too low ({len(self.paz.binned_profiles)} < 15)."},
                "chi2": {"level": 0, "id": "not_enough_binned_profiles", "message": f"Not enough profiles to run the monitoring or the SNR is too low ({len(self.paz.binned_profiles)} < 15)."},
                "peak_fluence": {"level": 0, "id": "not_enough_binned_profiles", "message": f"Not enough profiles to run the monitoring or the SNR is too low ({len(self.paz.binned_profiles)} < 15)."}
            }

        return {
            "rms": self.check_rms(),
            "chi2": self.check_chisquare(), 
            "peak_fluence": self.check_peak_fluence(), 
        }
        
    def check_rms(self):
        results = {
            "level": 0,
            "id": "no_change",
            "message": "No profile change event detected.", 
            "attachments": [], 
            "attachments_report_only": []
        }
        
        # Get RMS outliers
        _, outliers_idxes = self.paz.get_threshold_and_outliers(statistic="rms")

        # Sanity check for outliers
        if len(outliers_idxes) > 0.5 * len(self.paz.get_binned_profiles()):
            self.logger.warning(f"Too many outliers detected for {self.psr_id}. Skipping RMS check.")
            results["level"] = 1
            results["id"] = "too_many_outliers"
            results["message"] = "Too many outliers detected in RMS. Skipping RMS check."
        else:
            # Check if the last 3 samples are all outliers
            if len(outliers_idxes) >= 3:
                latest_3idx = [len(self.paz.get_binned_profiles()) - 1, len(self.paz.get_binned_profiles()) - 2, len(self.paz.get_binned_profiles()) - 3]
                if all(idx in outliers_idxes for idx in latest_3idx):
                    # Update results
                    results["level"] = 2
                    results["id"] = "continuous_high_rms"
                    results["message"] = "The last 3 profiles have a high RMS compared to previous profiles."

            # Check if the latest profile is an outlier
            if len(outliers_idxes) > 0 and results["level"] == 0:
                latest_idx = len(self.paz.get_binned_profiles()) - 1
                if latest_idx in outliers_idxes:
                    # Update results
                    results["level"] = 1
                    results["id"] = "high_rms"
                    results["message"] = "The latest profile has a high RMS compared to previous profiles."
        
        # Create diagnostics
        if results["level"] > 0:
            self.paz.plot(savefig=os.path.join(self.temp_dir, "basic_diagnostics.pdf"))
            results["attachments"].append(os.path.join(self.temp_dir, "basic_diagnostics.pdf")) 

        return results
    
    def check_chisquare(self):
        results = {
            "level": 0,
            "id": "no_change",
            "message": "No profile change event detected.", 
            "attachments": [], 
            "attachments_report_only": []
        }

        # Get chi2 outliers
        _, outliers_idxes = self.paz.get_threshold_and_outliers(statistic="chisquare")

        # Sanity check for outliers
        if len(outliers_idxes) > 0.33 * len(self.paz.get_binned_profiles()):
            self.logger.warning(f"Too many outliers detected for {self.psr_id}. Skipping chi2 check.")
            results["level"] = 1
            results["id"] = "too_many_outliers"
            results["message"] = "Too many outliers detected in chi2. Skipping chi2 check."
        else:
            # Check if the latest profile is an outlier
            if len(outliers_idxes) > 0:
                latest_idx = len(self.paz.get_binned_profiles()) - 1
                if latest_idx in outliers_idxes:
                    # Update results
                    results["level"] = 2
                    results["id"] = "high_chi2"
                    results["message"] = "The latest profile has a high chi2 compared to previous profiles."
                
            # Check if the last 3 samples are all outliers
            if len(outliers_idxes) >= 3:
                latest_3idx = [len(self.paz.get_binned_profiles()) - 1, len(self.paz.get_binned_profiles()) - 2, len(self.paz.get_binned_profiles()) - 3]
                if all(idx in outliers_idxes for idx in latest_3idx):
                    # Update results
                    results["level"] = 3
                    results["id"] = "continuous_high_chi2"
                    results["message"] = "The last 3 profiles have a high chi2 compared to previous profiles."

        # Create diagnostics
        if results["level"] > 0:
            self.paz.plot(savefig=os.path.join(self.temp_dir, "chi2_diagnostics.pdf"))
            results["attachments"].append(os.path.join(self.temp_dir, "chi2_diagnostics.pdf"))

        return results
    
    def check_peak_fluence(self):
        savefig_path95 = os.path.join(self.temp_dir, "peak_fluence_diagnostics_95.pdf")
        savefig_path997 = os.path.join(self.temp_dir, "peak_fluence_diagnostics_997.pdf")
        results = {
            "level": 0,
            "id": "no_change",
            "message": "No profile change event detected.", 
            "attachments": [], 
            "attachments_report_only": []
        }

        # Initialize peak fluence analyzer
        pfp = ProfilePeaks(template=self.paz.get_template(), aligned_profiles=self.paz.get_binned_profiles(), verbose=False)
        
        # Get 95% and 99.7% CL outliers
        _, outliers95 = pfp.get_outliers_thresholds_CL95(savefig=savefig_path95)
        _, outliers997 = pfp.get_outliers_thresholds_CL997(savefig=savefig_path997)

        # Sanity check for outliers
        for this_outliers95 in outliers95:
            if len(this_outliers95) > 0.33 * len(self.paz.get_binned_profiles()):
                self.logger.warning(f"Too many outliers detected for {self.psr_id}. Skipping peak fluence check.")
                results["level"] = 1
                results["id"] = "too_many_outliers"
                results["message"] = "Too many outliers detected in peak fluence. Skipping peak fluence check."
                results["attachments"].append(savefig_path95)
                return results
            
        # Check if the latest 3 profiles are 99.7% outliers
        for this_outliers997 in outliers997:
            if len(this_outliers997) >= 3:
                latest_3idx = [len(self.paz.get_binned_profiles()) - 1, len(self.paz.get_binned_profiles()) - 2, len(self.paz.get_binned_profiles()) - 3]
                if all(idx in this_outliers997 for idx in latest_3idx):
                    # Update results
                    results["level"] = 3
                    results["id"] = "continuous_very_high_peak_fluence"
                    results["message"] = "The last 3 profiles have a very high peak fluence compared to previous profiles."
                    results["attachments"].append(savefig_path997)
                    return results
            
        # Check if the latest 3 profiles are 95% outliers
        for this_outliers95 in outliers95:
            if len(this_outliers95) >= 3:
                latest_3idx = [len(self.paz.get_binned_profiles()) - 1, len(self.paz.get_binned_profiles()) - 2, len(self.paz.get_binned_profiles()) - 3]
                if all(idx in this_outliers95 for idx in latest_3idx):
                    # Update results
                    results["level"] = 2
                    results["id"] = "continuous_high_peak_fluence"
                    results["message"] = "The last 3 profiles have a high peak fluence compared to previous profiles."
                    results["attachments"].append(savefig_path95)

                    return results
            

        # Check if the latest profile is an 99.7% outlier
        for this_outliers997 in outliers997:
            if len(this_outliers997) > 0:
                latest_idx = len(self.paz.get_binned_profiles()) - 1
                if latest_idx in this_outliers997:
                    # Update results
                    results["level"] = 2
                    results["id"] = "sudden_very_high_peak_fluence"
                    results["message"] = "The latest profile has a high peak fluence compared to previous profiles."
                    results["attachments"].append(savefig_path997)
                    return results
            
        # Final check for 95% outliers
        for outliers95 in outliers95:
            if len(outliers95) > 0:
                latest_idx = len(self.paz.get_binned_profiles()) - 1
                if latest_idx in outliers95:
                    # Update results
                    results["level"] = 1
                    results["id"] = "high_peak_fluence"
                    results["message"] = "The latest profile has a high peak fluence compared to previous profiles."
                    results["attachments"].append(savefig_path95)
                    return results
            
        return results