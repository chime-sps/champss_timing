import numpy as np
import os
import matplotlib.pyplot as plt

from ..utils.logger import logger
from ..utils.utils import utils

from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

class SolarConjunctionDetector:
    def __init__(self, ra_deg, dec_deg, mjds, threshold_deg=4, logger=logger()):
        self.ra = ra_deg
        self.dec = dec_deg
        self.mjds = mjds
        self.threshold_deg = threshold_deg
        self.logger = logger

    def calc_separation(self, mjds):
        # Setup time object
        mjd_times = Time(mjds, format='mjd')

        # Compute Sun's position in icrs
        sun_positions = get_sun(mjd_times)

        # Find angular separation between the source and the Sun
        source_coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')
        separation = [
            source_coord.separation(
                SkyCoord(ra=sun_pos.ra, dec=sun_pos.dec, frame='icrs')
            ).deg for sun_pos in sun_positions]

        return np.array(separation)

    def detect_conjunctions(self):
        # Calculate the separation on the last observation mjd
        separation = self.calc_separation([np.max(self.mjds)])[0]

        return separation < self.threshold_deg, separation

    def plot_diagnostics(
            self, 
            snr_mjds, 
            snr_values, 
            residual_mjds,
            residual_vals, 
            residual_errs, 
            bad_residual_mjds,
            bad_residual_vals, 
            bad_residual_errs, 
            savefig=None
        ):

        # Setup the range of the plot
        plot_x_lim = [np.max(self.mjds) - 60, np.max(self.mjds)]
        plot_mjds = np.arange(plot_x_lim[0], plot_x_lim[1], 1)

        # Calculate separation
        separation = self.calc_separation(plot_mjds)

        # Plotting
        fig, ax = plt.subplots(3, 1, figsize=(10, 5), sharex=True, gridspec_kw={'hspace': 0})
        ax[0].errorbar(residual_mjds, residual_vals, yerr=residual_errs, fmt='kx', capsize=3)
        ax[0].errorbar(bad_residual_mjds, bad_residual_vals, yerr=bad_residual_errs, fmt='rx', capsize=3)
        ax[0].set_ylabel('Residuals (μs)')
        ax[1].plot(snr_mjds, snr_values, "kx")
        ax[1].set_ylabel('SNR')
        ax[2].plot(plot_mjds, separation, c="k")
        ax[2].axhline(self.threshold_deg, color='r', linestyle='--', label=f'Alert Threshold ({self.threshold_deg} deg)')
        ax[2].legend()
        ax[2].set_ylabel('Separation (deg)')
        ax[2].set_xlabel('MJD')
        ax[2].set_xlim(plot_x_lim)
        ax[2].set_yscale('log')
        fig.suptitle('Solar Conjunction Diagnostics')
        fig.tight_layout()

        if savefig:
            plt.savefig(savefig)   


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

        results = {
            "solconj": {
                "level": 0,
                "id": "no_solar_conjunction",
                "message": "No solar conjunction detected.", 
                "attachments": []
            }
        }

        if len(self.timing_info) < 1:
            return results # No timing info available

        # Initialize the solar conjunction detector
        solconj_detector = SolarConjunctionDetector(
            ra_deg=self.timing_info[-1]["fitted_params"]["RAJ"] / 24 * 360,
            dec_deg=self.timing_info[-1]["fitted_params"]["DECJ"],
            mjds=self.timing_info[-1]["notes"]["fitted_mjds"],
            threshold_deg=4,
            logger=self.logger.copy()
        )

        # Detect solar conjunctions
        is_conjunction, separation = solconj_detector.detect_conjunctions()
        if is_conjunction:
            # Get residuals and times
            resid_mjds = self.timing_info[-1]["notes"]["fitted_mjds"]
            resid_vals = self.timing_info[-1]["residuals"]
            bad_resid_mjds = self.timing_info[-1]["notes"]["bad_toa_mjds"]
            bad_resid_vals = self.timing_info[-1]["notes"]["bad_toa_residuals"]

            # Get snr values and times
            snr_mjds = []
            snr_vals = []
            for ar in self.db_hdl.get_all_archive_info():
                snr_mjds.append(ar["notes"]["init_epoch"])
                snr_vals.append(ar["psr_snr"])

            # Generate diagnostics
            diagnostic_path = f"/tmp/solar_conjunction_diagnostic__{self.psr_id}__{utils.get_time_string()}.pdf"
            solconj_detector.plot_diagnostics(
                snr_mjds=snr_mjds,
                snr_values=snr_vals,
                residual_mjds=resid_mjds,
                residual_vals=resid_vals["val"],
                residual_errs=resid_vals["err"],
                bad_residual_mjds=bad_resid_mjds,
                bad_residual_vals=bad_resid_vals["val"],
                bad_residual_errs=bad_resid_vals["err"],
                savefig=diagnostic_path
            )

            # Update results
            results["solconj"]["level"] = 1
            results["solconj"]["id"] = "solar_conjunction_detected"
            results["solconj"]["message"] = f"The source is in solar conjunction with the Sun (separation = {separation:.2f} deg). "
            
            # Attach the diagnostic plot if it exists
            if os.path.exists(diagnostic_path):
                results["solconj"]["attachments"] = [diagnostic_path]
                self.logger.info(f"Solar conjunction diagnostic plot generated successfully -> {diagnostic_path}")
            else:
                results["solconj"]["attachments"] = []
                self.logger.warning(f"Failed to generate solar conjunction diagnostic plot -> {diagnostic_path}")

                # Update results again
                results["solconj"]["level"] = 2
                results["solconj"]["id"] = "solar_conjunction_detected_no_plot"
                results["solconj"]["message"] += "However, the checker **failed** to generate the diagnostic plot. Please check the processing log for more details."

        return results
        