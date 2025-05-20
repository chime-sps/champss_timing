import numpy as np
import os

from ..utils.logger import logger
from ..utils.utils import utils
from ..tools.glitch_utils import glitch_utils
from ..utils.stats_utils import stats_utils

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

        # Trigger glitch alert if the residual is suddenly increasing
        if (
            self.basic_checker_results["residual"]["id"] == "residual_sudden_increase" or 
            self.basic_checker_results["residual"]["id"] == "residual_very_sudden_increase"
        ):
            # Generate the diagnostic plot
            diagnostic_path = f"/tmp/glitch_diagnostic__{self.psr_id}__{utils.get_time_string()}.pdf"
            with glitch_utils(db_hdl=self.db_hdl, logger=self.logger.copy()) as gu:
                gu.estimate_glitch(savefig=diagnostic_path)

            # Check if glitch diagnostic plot exists
            if os.path.exists(diagnostic_path):
                results["glitch"]["level"] = 2
                results["glitch"]["id"] = "glitch_detected"
                results["glitch"]["message"] = "Glitch-like event detected. Please check the following glitch diagnostic plot. "
                results["glitch"]["attachments"] = [diagnostic_path]
                self.logger.info(f"Glitch diagnostic plot generated successfully -> {diagnostic_path}")
            else:
                results["glitch"]["level"] = 2
                results["glitch"]["id"] = "glitch_detected"
                results["glitch"]["message"] = "Glitch-like event detected, but failed to generate the diagnostic plot. Please check the processing log for more details."
                results["glitch"]["attachments"] = []
                self.logger.warning(f"Failed to generate glitch diagnostic plot -> {diagnostic_path}")
        
        return results
    