import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import datetime

from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import numpy as np

from ..utils.logger import logger
from ..utils.utils import utils


class TelescopePointingOffsetChecker:
    def __init__(self, db_hdl, threshold_deg, logger=logger()):
        self.db_hdl = db_hdl
        self.threshold_deg = threshold_deg
        self.logger = logger

        # Get telescope pointing information
        self.telescope_pointing = self.get_telescope_pointing()

        # Initialize states
        self.timing_ra_val = None
        self.timing_ra_err = None
        self.timing_dec_val = None
        self.timing_dec_err = None
        self.separation = None
        self.ellipse_points = None
        self.best_ellipse_point = None

    def get_telescope_pointing(self):
        """
        Get the telescope pointing information from the database.
        """

        # Get all archive info
        archive_info = self.db_hdl.get_all_archive_info()

        # Extract observation mjds and pointing information
        telescope_pointings = {}
        for ar in archive_info:
            # Read notes for each archive
            obs_mjd = ar["notes"]["init_epoch"]
            pointing_ra = ar["notes"]["ra_deg"]
            pointing_dec = ar["notes"]["dec_deg"]

            telescope_pointings[obs_mjd] = (pointing_ra, pointing_dec)

        # Find the latest pointing
        idx_latest = np.argmax(list(telescope_pointings.keys()))
        latest_ra, latest_dec = list(telescope_pointings.values())[idx_latest]

        return latest_ra, latest_dec

    def generate_ellipse(self, h, k, a, b, num_points=128):
        """
        Generate points on an ellipse with semi-major axis a and semi-minor axis b.
        
        Parameters:
        h (float): x-coordinate of the center of the ellipse
        k (float): y-coordinate of the center of the ellipse
        a (float): Semi-major axis length
        b (float): Semi-minor axis length
        num_points (int): Number of points to generate along the ellipse
        
        Returns:
        tuple: Arrays of x and y coordinates of the ellipse points
        """
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = h + a * np.cos(theta)
        y = k + b * np.sin(theta)
        return x, y

    def check_pointing_offset(self, timing_ra_val, timing_ra_err, timing_dec_val, timing_dec_err):
        """
        Check if the telescope is pointing nearby the timing position.

        Parameters
        ----------
        timing_ra_val : float
            The right ascension of the timing position in degrees.
        timing_ra_err : float
            The uncertainty of the right ascension of the timing position in degrees.
        timing_dec_val : float
            The declination of the timing position in degrees.
        timing_dec_err : float
            The uncertainty of the declination of the timing position in degrees.

        Returns
        -------
        bool
            True if the telescope is pointing nearby the timing position, False otherwise.
        """

        # Get latest telescope pointing
        latest_ra, latest_dec = self.telescope_pointing

        # Check if the latest pointing is the default value (3.33, 3.33)
        if latest_ra == 3.33 and latest_dec == 3.33:
            self.logger.warning("Telescope pointing information is not available in the database.")
            return True, 0

        # Generate the ellipse representing the uncertainty region around the timing position
        timing_ellipse_x, timing_ellipse_y = self.generate_ellipse(
            timing_ra_val, timing_dec_val, timing_ra_err, timing_dec_err
        )

        # Calculate angular separation between timing position and telescope pointing
        seperations = []
        for ra, dec in zip(timing_ellipse_x, timing_ellipse_y):
            ra = ra % 360  # Ensure RA is within [0, 360) degrees
            dec = np.clip(dec, -90, 90)  # Ensure Dec is within [-90, 90] degrees
            source_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
            pointing_coord = SkyCoord(ra=latest_ra * u.deg, dec=latest_dec * u.deg, frame='icrs')
            seperations.append(source_coord.separation(pointing_coord).deg)

        # Find best match on the ellipse
        best_idx = np.argmin(seperations)
        separation = seperations[best_idx]

        # Update internal state
        self.timing_ra_val = timing_ra_val
        self.timing_ra_err = timing_ra_err
        self.timing_dec_val = timing_dec_val
        self.timing_dec_err = timing_dec_err
        self.ellipse_points = [timing_ellipse_x, timing_ellipse_y]
        self.best_ellipse_point = [timing_ellipse_x[best_idx], timing_ellipse_y[best_idx]]
        self.separation = separation

        return separation < self.threshold_deg, separation

    def plot_diagnostics(self, savefig=None):
        """
        Plot the telescope pointing diagnostics.

        Parameters
        ----------
        timing_ra_val : float
            The right ascension of the timing position in degrees.
        timing_ra_err : float
            The uncertainty of the right ascension of the timing position in degrees.
        timing_dec_val : float
            The declination of the timing position in degrees.
        timing_dec_err : float
            The uncertainty of the declination of the timing position in degrees.
        savefig : str, optional
            The path to save the figure. If None, the figure will not be saved. Default is None.
        """

        if self.timing_ra_val is None or self.timing_dec_val is None:
            self.logger.warning("Timing position is not set. No diagnostics plot will be generated.")
            return

        # Get latest telescope pointing
        latest_ra, latest_dec = self.telescope_pointing

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(latest_ra, latest_dec, color='k', label='Telescope Pointing', s=100, marker='x')
        ax.scatter(self.timing_ra_val, self.timing_dec_val, color='r', label='Timing Position', s=100, marker='x')
        ax.errorbar(self.timing_ra_val, self.timing_dec_val, xerr=self.timing_ra_err, yerr=self.timing_dec_err, fmt='none', ecolor='r', capsize=5)
        ax.add_artist(
            plt.Circle((latest_ra, latest_dec), self.threshold_deg, color='k', fill=False, linestyle='-', label=f'Alert Threshold ({self.threshold_deg} deg)')
        )
        # if self.best_ellipse_point is not None and self.separation is not None and self.ellipse_points is not None:
        #     ax.plot([self.best_ellipse_point[0], latest_ra], [self.best_ellipse_point[1], latest_dec], color='k', linestyle=':')
        #     ax.text(
        #         (self.best_ellipse_point[0] + latest_ra) / 2, (self.best_ellipse_point[1] + latest_dec) / 2, f'Separation: {self.separation:.2f} deg', 
        #         fontsize=10, color='k', ha='center'
        #     )
        #     ax.plot(self.ellipse_points[0], self.ellipse_points[1], color='r', linestyle='-', label='Timing Position Uncertainty Ellipse')
        ax.set_xlabel('Right Ascension (deg)')
        ax.set_ylabel('Declination (deg)')
        ax.legend()
        ax.grid()

        # Plot pipeline info
        fig.text(0.001, 0, f"CHAMPSS Timing Pipeline ({utils.get_version_hash()}) position.TelescopePointingOffsetChecker | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")

        # Plot title
        fig.suptitle('Telescope Pointing Diagnostics (Separation = {:.2f} deg)'.format(self.separation))
        fig.tight_layout()

        # Save the figure if requested
        if savefig:
            plt.savefig(savefig)
            self.logger.info(f"Telescope pointing diagnostics plot saved to {savefig}")
    
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
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0})
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

        # Plot pipeline info
        fig.text(0.001, 0, f"CHAMPSS Timing Pipeline ({utils.get_version_hash()}) position.SolarConjunctionDetector | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")

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
        db_hdl : database handler
            The database handler object.
        basic_checker_results : dict
            The results from the basic checker.
        psr_id : str
            The pulsar ID.
        psr_dir : str
            The pulsar directory.
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
        Run the position checks.
        """

        results = {}

        if len(self.timing_info) < 1:
            return results # No timing info available

        # Get timing position
        try:
            # Try to use pint to read the model
            from io import StringIO
            from pint.models import get_model

            m = get_model(
                StringIO(self.timing_info[-1]["notes"]["fitted_parfile"])
            )

            self.timing_ra_val = m.RAJ.value / 24 * 360
            self.timing_ra_err = m.RAJ.uncertainty.value / 24 * 360
            self.timing_dec_val = m.DECJ.value
            self.timing_dec_err = m.DECJ.uncertainty.value
        except Exception as e:
            # If pint is not available or fails, use the values from the database
            self.logger.warning(f"Failed to read timing position from parfile using pint: {e}. Using values from the database instead.")
            self.timing_ra_val = self.timing_info[-1]["fitted_params"]["RAJ"] / 24 * 360
            self.timing_ra_err = 0.0
            self.timing_dec_val = self.timing_info[-1]["fitted_params"]["DECJ"]
            self.timing_dec_err = 0.0

            # Try to read the errors from the parfile if available
            for line in self.timing_info[-1]["notes"]["fitted_parfile"].split("\n"):
                if line.startswith("RAJ"):
                    if len(line.split()) == 4:
                        self.timing_ra_err = float(line.split()[3]) / 3600
                if line.startswith("DECJ"):
                    if len(line.split()) == 4:
                        self.timing_dec_err = float(line.split()[3]) / 3600

        # Check telescope pointing
        results.update(self.check_telescope_pointing())

        # Check for solar conjunctions
        results.update(self.check_solar_conjunction())

        return results

    def check_telescope_pointing(self):
        """
        Check if the telescope is pointing nearby the timing position. 
        """

        results = {
            "telescope_pointing": {
                "level": 0,
                "id": "no_pointing_offset",
                "message": "Telescope pointing is pointing nearby the timing position.",
                "attachments": []
            }
        }

        # Check if timing position is reliable
        if np.max(self.timing_info[-1]["notes"]["fitted_mjds"]) - np.min(self.timing_info[-1]["notes"]["fitted_mjds"]) < 90:
            self.logger.debug("Timing position is not reliable (fitted over less than 90 days). Skipping telescope pointing check.")
            return results

        # Initialize the telescope pointing offset checker
        pointing_checker = TelescopePointingOffsetChecker(
            db_hdl=self.db_hdl,
            threshold_deg=0.5,
            logger=self.logger.copy()
        )

        # Check against telescope pointing
        is_pointing_ok, separation = pointing_checker.check_pointing_offset(
            timing_ra_val=self.timing_ra_val,
            timing_ra_err=self.timing_ra_err,
            timing_dec_val=self.timing_dec_val, 
            timing_dec_err=self.timing_dec_err
        )

        if not is_pointing_ok:
            # Update results
            results["telescope_pointing"]["level"] = 1
            results["telescope_pointing"]["id"] = "large_pointing_offset"
            results["telescope_pointing"]["message"] = f"Telescope pointing is off by {separation:.2f} degrees from the timing position."

            # Generate diagnostics plot
            diagnostic_path = f"/tmp/telescope_pointing_diagnostic__{self.psr_id}__{utils.get_time_string()}.pdf"
            pointing_checker.plot_diagnostics(savefig=diagnostic_path)

            # Attach the diagnostic plot if it exists
            if os.path.exists(diagnostic_path):
                results["telescope_pointing"]["attachments"] = [diagnostic_path]
                self.logger.info(f"Telescope pointing diagnostic plot generated successfully -> {diagnostic_path}")
            else:
                results["telescope_pointing"]["attachments"] = []
                self.logger.warning(f"Failed to generate telescope pointing diagnostic plot -> {diagnostic_path}")

        return results
    
    def check_solar_conjunction(self):
        """
        Check for solar conjunctions based on the latest timing information.
        """

        results = {
            "solconj": {
                "level": 0,
                "id": "no_solar_conjunction",
                "message": "No solar conjunction detected.", 
                "attachments": []
            }
        }

        # Initialize the solar conjunction detector
        solconj_detector = SolarConjunctionDetector(
            ra_deg=self.timing_ra_val,
            dec_deg=self.timing_dec_val,
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
            results["solconj"]["id"] = "near_solar_conjunction"
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
        