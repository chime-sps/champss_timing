import numpy as np
import os
import copy
from scipy.stats import f as f_stats
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

from ..utils.logger import logger
from ..utils.utils import utils
from ..tools.glitch_utils import glitch_utils
from ..utils.stats_utils import stats_utils

class DiscontinuityDetectorState:
    def __init__(self, x, y, order=1):
        self.x = x
        self.y = y
        self.order = order
        self.coefficients = np.zeros(order + 1)

    def set_order(self, order):
        if order < 0:
            raise ValueError("Order must be non-negative")
        self.order = order
        self.coefficients = np.zeros(order + 1)

    def get_higher_order_state(self):
        self_copy = copy.deepcopy(self)
        self_copy.set_order(self.order + 1)
        return self_copy

    def fit(self):
        self.coefficients = np.polyfit(self.x, self.y, self.order)

    def predict(self, x):
        return np.polyval(self.coefficients, x)
    
    def residual(self, x, y):
        predicted = self.predict(x)
        return np.abs(y - predicted)
    
    def plot(self, ax=None, show=True):
        if ax is None:
            ax = plt.gca()
        x_range = np.linspace(np.min(self.x), np.max(self.x), 100)
        y_fit = self.predict(x_range)
        ax.plot(x_range, y_fit, label=f"Order {self.order} Fit")
        ax.scatter(self.x, self.y, color='red', s=10, label='Data Points')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Residuals')
        ax.legend()
        if show:
            plt.tight_layout()
            plt.show()

class DiscontinuityDetector:
    def __init__(self, mjds, residuals, logger=logger(), verbose=False):
        self.mjds = np.array(mjds)
        self.residuals = np.array(residuals)
        self.state = DiscontinuityDetectorState(self.mjds, self.residuals)
        self.logger = logger
        self.verbose = verbose
    
    def fit(self, max_iters=10):
        f_test_status = True # True = passed the f-test
        while f_test_status and max_iters > 0:
            # Fit the current state
            self.state.fit()

            # Fit a higher order state
            higher_order_state = self.state.get_higher_order_state()
            higher_order_state.fit()

            # Calculate residuals for both states
            residuals = self.state.residual(self.mjds, self.residuals)
            higher_order_residuals = higher_order_state.residual(self.mjds, self.residuals)

            # Calculate F-statistic
            ss_res = np.sum(residuals**2)
            ss_res_higher = np.sum(higher_order_residuals**2)
            df_res = len(self.mjds) - self.state.order - 1
            df_res_higher = len(self.mjds) - higher_order_state.order - 1
            f_statistic = (ss_res - ss_res_higher) / (df_res - df_res_higher) / (ss_res_higher / df_res_higher)

            # Calculate p-value
            p_value = f_stats.sf(float(f_statistic), dfn=float(df_res - df_res_higher), dfd=float(df_res_higher))

            if p_value < 0.05:
                # If p-value is less than 0.05, we reject the null hypothesis
                f_test_status = True
                self.state = higher_order_state
                if self.verbose:
                    self.logger.info(f"F-test passed: p-value = {p_value:.4f}, switching to higher order state.")
            else:
                # If p-value is greater than 0.05, we accept the null hypothesis
                f_test_status = False
                if self.verbose:
                    self.logger.info(f"F-test failed: p-value = {p_value:.4f}, keeping current state.")
            # self.state = higher_order_state

            if self.verbose:
                self.logger.info(f"Current order: {self.state.order}, F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
                self.state.plot()

            max_iters -= 1

        return self.state
    
    def is_discontinuous(self, mjd, val, threshold=3.0, get_details=False):
        """
        Check if the given value at mjd is discontinuous based on the fitted state.
        Parameters:
        - mjd: The modified Julian date to check.
        - val: The value at the given mjd to check for discontinuity.
        - threshold: The number of standard deviations to consider as a discontinuity.
        - get_details: If True, return additional details about the check.
        Returns:
        [get_details=False]:
        - is_discontinuous: Boolean indicating if the value is discontinuous.
        - sigma: The standardized residual (residual / noise).
        [get_details=True]:
        - is_discontinuous: Boolean indicating if the value is discontinuous.
        - predicted: The predicted value at the given mjd.
        - residual: The absolute residual value.
        - noise: The estimated noise level.
        - sigma: The standardized residual (residual / noise).
        """
        
        if self.state.order < 1:
            raise ValueError("State must be fitted before checking for discontinuity.")
        
        predicted = self.state.predict(mjd)
        residual = np.abs(val - predicted)
        # noise = np.std(self.state.residual(self.mjds, self.residuals))
        mad = median_abs_deviation(self.state.residual(self.mjds, self.residuals))

        # Inflate the noise by the time since the last fit
        # Based on the assumption that the model is less predictive as time goes on
        dt = np.min(np.abs(self.mjds - mjd))
        mad = mad * dt**2

        # Estimate the noise level
        # Ideally median is 0, but just in case of non-zero median, we add it to the noise
        noise = np.median(self.state.residual(self.mjds, self.residuals)) + stats_utils.mad_to_stdev(mad)
        
        is_discontinuous = residual > threshold * noise

        if get_details:
            return is_discontinuous, predicted, residual, noise, residual / noise

        return is_discontinuous, residual / noise

    def plot(self, mjd, val, threshold=3.0, ax=None, show=True):
        if ax is None:
            ax = plt.gca()

        # Get discontinuity status
        is_discontinuous, predicted, residual, noise, sigma = self.is_discontinuous(mjd, val, threshold=threshold, get_details=True)

        # Get predicted curve
        mjds_all = np.concatenate((self.mjds, [mjd]))
        pred_x = np.linspace(np.min(mjds_all), np.max(mjds_all), 100)
        pred_y = self.state.predict(pred_x)

        ax.plot(self.mjds, self.residuals, 'k.', label='Residuals', markersize=5)
        ax.plot(pred_x, pred_y, 'r--', label=f'Order {self.state.order} Fit')
        ax.plot(mjd, predicted, "rx", label='Predicted Value with Noise')
        ax.plot(mjd, val, "bx", label='Current Value', markersize=10)
        ax.axhline(predicted + threshold * noise, color='gray', linestyle='--', label='Upper Threshold')
        ax.axhline(predicted - threshold * noise, color='gray', linestyle='--', label='Lower Threshold')
        if is_discontinuous:
            ax.text(0.025, 0.95, f"Discontinuity Detected (σ={sigma:.2f})", 
                    transform=ax.transAxes, fontsize=9, color='red', bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.025, 0.95, f"No Discontinuity Detected (σ={sigma:.2f})", 
                    transform=ax.transAxes, fontsize=9, color='green', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_title("Discontinuity Detection State")
        if show:
            plt.tight_layout()
            plt.show()    


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
            "glitch": {
                "level": 0,
                "id": "no_glitch",
                "message": "No glitch detected.", 
                "attachments": []
            }
        }

        if len(self.timing_info) < 1:
            return results # No timing info available

        # Get residuals and times
        mjds = np.array(self.timing_info[-1]["notes"]["fitted_mjds"])
        resids = np.array(self.timing_info[-1]["residuals"]["val"])

        # Sanity check
        if len(mjds) < 15:
            return results # Not enough data points to detect a glitch
        
        # Sort samples by MJD
        sorted_indices = np.argsort(mjds)
        mjds = mjds[sorted_indices]
        resids = resids[sorted_indices]

        # Run discontinuity detection
        dd = DiscontinuityDetector(mjds[-15:-1], resids[-15:-1], verbose=False)
        dd.fit()
        is_discontinuous, sigma = dd.is_discontinuous(mjds[-1], resids[-1])

        # Trigger glitch alert if the residual is suddenly increasing
        # if (
        #     self.basic_checker_results["residual"]["id"] == "residual_sudden_increase" or 
        #     self.basic_checker_results["residual"]["id"] == "residual_very_sudden_increase"
        # ):
        if is_discontinuous:
            # Generate the diagnostic plot
            diagnostic_path = f"/tmp/glitch_diagnostic__{self.psr_id}__{utils.get_time_string()}.pdf"
            with glitch_utils(db_hdl=self.db_hdl, logger=self.logger.copy()) as gu:
                gu.estimate_glitch(savefig=diagnostic_path)

            # Check if glitch diagnostic plot exists
            if os.path.exists(diagnostic_path):
                results["glitch"]["level"] = 3
                results["glitch"]["id"] = "discontinuity_detected"
                results["glitch"]["message"] = f"Glitch-like event detected. Please check the posted glitch diagnostic plot (sigma={sigma}). "
                results["glitch"]["attachments"] = [diagnostic_path]
                self.logger.info(f"Glitch diagnostic plot generated successfully -> {diagnostic_path}")
            else:
                results["glitch"]["level"] = 3
                results["glitch"]["id"] = "discontinuity_detected"
                results["glitch"]["message"] = f"Glitch-like event detected, but **failed** to generate the diagnostic plot. Please check the processing log for more details  (sigma={sigma})."
                results["glitch"]["attachments"] = []
                self.logger.warning(f"Failed to generate glitch diagnostic plot -> {diagnostic_path}")
        
        return results
    