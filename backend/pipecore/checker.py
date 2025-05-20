import os
import pkgutil
import importlib
import numpy as np
import traceback

from ..utils.logger import logger
from ..utils.notification import notification
from ..datastores.database import database

class checker:
    def __init__(self, psr_dir, db_hdl=None, noti_hdl=notification(), psr_id=None, logger=logger(), verbose=False):
        """
        Initialize the checker class.
        Parameters
        ----------
        psr_dir : str
            The directory of the pulsar.
        db_hdl : database
            The database handler.
        noti_hdl : notification
            The notification handler.
        psr_id : str
            The ID of the pulsar.
        logger : logger
            The logger object.
        verbose : bool
            If True, print debug messages.
        """

        # Initial parameters and utilities
        self.psr_dir = psr_dir
        self.db_hdl = db_hdl
        self.psr_id = psr_id
        self.diagnostic_plot = psr_dir + "/champss_diagnostic.pdf"
        self.noti_hdl = noti_hdl
        self.logger = logger
        self.verbose = verbose

        # Get psr_id
        if self.psr_id == None:
            self.psr_id = os.path.basename(psr_dir)

        # Initialize the database handler
        if self.db_hdl == None:
            self.db_hdl = database(psr_dir + "/champss_timing.sqlite3.db")

        # Load available checkers
        self.checkers = self.load_aval_checkers()

    def load_aval_checkers(self):
        """
        List all available checkers.
        """
        
        # List submodules under ../monitoring
        checkers = {}
        for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__) + "/../monitoring"]):
            try:
                checkers[name] = importlib.import_module(f"..monitoring.{name}", package=__package__).Main
            except ImportError as e:
                self.logger.warning(f"Failed to load [{name}] checker: {e}")
                self.send_notification({"level": 2, "message": f"Checker [{name}] failed to load. Please check the installation."})
                continue
            except AttributeError as e:
                self.logger.warning(f"Failed to load [{name}] checker: {e}. This may be due to a missing Main entry in the monitoring module.")
                self.send_notification({"level": 2, "message": f"Checker [{name}] unable to find the Main entry. Please check the installation."})
                continue

        # Make sure has basics checker
        if "basics" not in checkers:
            raise ImportError("No basics checker found. This should not happen, please check the installation.")   
        
        # Make sure the basics checker is the first one
        checker_keys = list(checkers.keys())
        checker_keys.remove("basics")
        checker_keys.insert(0, "basics")
        checkers = {key: checkers[key] for key in checker_keys}

        if self.verbose:
            self.logger.debug("Available checkers:")
            for checker in checkers:
                self.logger.debug(f" - {checker}")

        return checkers

    def check(self):
        """
        Run checkers
        """
        
        # Initialize the results
        results = {"basics": {}}

        # Loop over all checkers
        for checker_name, checker_class in self.checkers.items():
            self.logger.debug(f"Running [{checker_name}] checker...")

            # Run checker
            try:
                checker = checker_class(db_hdl=self.db_hdl, basic_checker_results=results["basics"], psr_id=self.psr_id, psr_dir=self.psr_dir, logger=self.logger.copy())
                results[checker_name] = checker.check()
            except Exception as e:
                self.logger.error(f"Failed to run {checker_name} checker: {e}")
                self.logger.error(traceback.format_exc())
                continue
                
            # Make sure the checker results are valid
            for key in results[checker_name]:
                if "id" not in results[checker_name][key]:
                    results[checker_name][key]["id"] = "no_id"
                if "message" not in results[checker_name][key]:
                    results[checker_name][key]["message"] = "no_message"
                if "level" not in results[checker_name][key]:
                    results[checker_name][key]["level"] = 2 # Assume the worst if not specified
                if "attachments" not in results[checker_name][key]:
                    results[checker_name][key]["attachments"] = []
                if type(results[checker_name][key]["attachments"] ) != list:
                    results[checker_name][key]["attachments"] = [results[checker_name][key]["attachments"]]

        # Send text notification
        self.logger.debug(f"Sending text notification...")
        attachments = []
        for checker_name, checker_results in results.items():
            for key, check_result in checker_results.items():
                # Send text notification
                attachments += self.send_text_notification(check_result) # The function will return the attachments to be sent

        # Eliminate duplicated attachments
        attachments = list(set(attachments))

        # Replace shortcuts in attachment paths
        for i, string in enumerate(attachments):
            if "%DIAGNOSTIC_PLOT%" in string:
                attachments[i] = string.replace("%DIAGNOSTIC_PLOT%", self.diagnostic_plot)
            if "%PSR_DIR%" in string:
                attachments[i] = string.replace("%PSR_DIR%", self.psr_dir)

        # Send attachments
        self.logger.debug(f"Sending attachments...")
        for att in attachments:
            self.send_attachment(att)

        return results

    def send_text_notification(self, check_result):
        """
        Send notification based on the check result.
        """

        # Check if notification is needed
        if check_result["level"] == 0:
            self.logger.debug("Status is normal, no notification sent.", layer=1)
            return []
        
        # Send text message
        if check_result["level"] == 1:
            msg = f"*Checker Warning for {self.psr_id}*: `" + check_result["message"] + "`"
            self.noti_hdl.send_message(msg)
            self.logger.success(msg, layer=1)
        elif check_result["level"] == 2:
            msg = f"*Checker Important Warning for {self.psr_id}*: `" + check_result["message"] + "`"
            self.noti_hdl.send_urgent_message(msg)
            self.logger.success(msg, layer=1)
        
        return check_result["attachments"]

    def send_attachment(self, attachment):
        """
        Send attachment to the notification handler.
        """

        # Send attachments
        if os.path.exists(attachment):
            self.noti_hdl.send_file(attachment)
            self.logger.success("Sent file: " + attachment, layer=1)
        else:
            self.noti_hdl.send_urgent_message(f"*Checker Unexpected ({self.psr_id})*: `Attachment {attachment} not found. Please check the processing log and make sure the file was generated successfully.`")
            self.logger.error("Attachment not found: " + attachment, layer=1)

        return 