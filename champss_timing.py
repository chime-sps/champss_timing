import os
import json
import time
import shutil
import traceback
import astropy.units as u

from .utils import utils
from .timing import timing
from .database import database
from .archive_utils import archive_utils
from .archive_cache import archive_cache
from .plot import plot
from .notification import notification
from .champss_checker import champss_checker
from .logger import logger

class champss_timing:
    def __init__(self, psr_dir, data_archives, slack_token, n_pools=4, workspace_cleanup=True, logger=logger()):
        """
        CHAMPSS timing pipeline class

        Parameters
        ----------
        psr_dir : str
            Path to the directory containing the pulsar data
        data_archives : dict
            List of data archives to be processed

        Returns
        -------
        None
        """

        # Parameters
        self.path_psr_dir = psr_dir
        self.psr_id = None
        self.path_data_archives = data_archives
        self.path_db = f"{self.path_psr_dir}/champss_timing.sqlite3.db"
        self.path_pulse_template = f"{self.path_psr_dir}/paas.std"
        self.path_timing_model = f"{self.path_psr_dir}/pulsar.par"
        self.path_timing_model_bakdir = f"{self.path_psr_dir}/parfile_bak"
        self.path_timing_config = f"{self.path_psr_dir}/champss_timing.config"
        self.path_diagnostic_plot = f"{self.path_psr_dir}/champss_diagnostic.pdf"
        self.info_ars_mjds = []
        self.info_ars_paths = []
        self.info_first_mjd = 0
        self.info_last_mjd = 0
        self.n_pools = n_pools
        self.workspace_cleanup = workspace_cleanup
        self.logger = logger

        # Format psr_dir
        if self.path_psr_dir.endswith("/"):
            self.path_psr_dir = self.path_psr_dir[0:-1]

        # Objects
        self.db_hdl = database(self.path_db)
        self.archive_cache = archive_cache(self.path_psr_dir, db_hdl=self.db_hdl)
        self.noti_hdl = notification(slack_token)

        # Timing config
        self.timing_config = {}

    def initialize(self):
        # Print git version
        self.logger.success(f"CHAMPSS Timing Pipeline (v_{utils.get_version_hash()})")

        # Initialize DB
        self.db_hdl.initialize()

        # Initialize archive_cache
        self.archive_cache.initialize()

        # Initialize archive_cache
        self.db_hdl.initialize()
        
        # Check number of data archives
        if len(self.path_data_archives) < 5:
            raise ValueError("No enough data archives to perform timing (at least 5 needed)")

        # Sort data archives by MJD
        self.path_data_archives = {k: v for k, v in sorted(self.path_data_archives.items(), key=lambda item: item[0])}

        # Get first and last MJD
        self.info_ars_mjds = list(self.path_data_archives.keys())
        self.info_ars_paths = list(self.path_data_archives.values())
        self.info_first_mjd = list(self.path_data_archives.keys())[0]
        self.info_last_mjd = list(self.path_data_archives.keys())[-1]
            
        # Check folder exists
        if not os.path.isdir(self.path_psr_dir):
            raise FileNotFoundError(f"Directory {self.path_psr_dir} not found for pulsar directory")
        if not os.path.isdir(self.path_timing_model_bakdir):
            os.makedirs(self.path_timing_model_bakdir)
            self.logger.debug(f"Created timing model backup directory {self.path_timing_model_bakdir}")

        # Check files exist
        if not os.path.isfile(self.path_pulse_template):
            raise FileNotFoundError(f"File {self.path_pulse_template} not found for pulse template")
        if not os.path.isfile(self.path_timing_model):
            raise FileNotFoundError(f"File {self.path_timing_model} not found for timing model")
        if not os.path.isfile(self.path_timing_config):
            raise FileNotFoundError(f"File {self.path_timing_config} not found for timing configuration")
        for mjd in self.path_data_archives:
            if not os.path.isfile(self.path_data_archives[mjd]):
                raise FileNotFoundError(f"File {self.path_data_archives[mjd]} not found for data archive")
        
        # Load config
        self.timing_config = json.load(open(self.path_timing_config, "r")) 

        # Get psr id
        self.psr_id = self.path_psr_dir.split("/")[-1]

    def run(self):
        n_timed = 0
        while True:
            timing_res = self.timing()
            if timing_res["status"] != "success":
                if timing_res["status"] == "error":
                    raise Exception("Timing failed. ")

                # if n_timed == 1:
                if n_timed == 0:
                    # plot(db_hdl=self.db_hdl).diagnostic(savefig=self.path_diagnostic_plot)
                    self.logger.debug(f"No additional file for timing. ")
                    break

                # update model for cached archives
                self.logger.debug(f"Updating model for all cached archives")
                self.archive_cache.update_model()

                # Create diagnostic plot
                self.logger.info(f"Creating diagnostic plot")
                plot(db_hdl=self.db_hdl).diagnostic(savefig=self.path_diagnostic_plot)

                # Run checker
                champss_checker(self.path_psr_dir, self.db_hdl, self.noti_hdl, self.psr_id).check()

                # End of the script
                self.logger.success("Script finished. ")
                self.noti_hdl.send_message(f"Timing finished for {n_timed} days of data. ", psr_id=self.psr_id)

                break
            n_timed += 1

        return {"n_timed": n_timed}
    
    def timing(self):
        self.logger.level_up()
        self.logger.debug("Starting timing... ")
        # Get last timing info
        last_timing_info = self.db_hdl.get_last_timing_info()

        # Check if last timing info exists
        mjds = []
        archives = []
        if last_timing_info["timestamp"] == 0:
            self.logger.info("No timing info found, starting from scratch. ")
            mjds = list(self.path_data_archives.keys())[0:5]
            archives = [self.path_data_archives[mjd] for mjd in mjds]
            fit_params = ["F0"]
            potential_fit_params = []
        else:
            self.logger.info(f"Last timing info found: ")
            self.logger.data("Timestamp", last_timing_info["timestamp"])
            self.logger.data("n_obs", len(last_timing_info["files"]))
            self.logger.data("obs_mjds", last_timing_info["obs_mjds"])
            self.logger.data("unfreeze_params", last_timing_info["unfreeze_params"])
            self.logger.data("chi2", last_timing_info["chi2"])
            self.logger.data("chi2_reduced", last_timing_info["chi2_reduced"])

            # find last mjd
            last_mjd = max(last_timing_info["obs_mjds"]) 

            # find the index of the next mjd to process
            mjds = []
            archives = []
            for idx, mjd in enumerate(list(self.path_data_archives.keys())): 
                if mjd >= last_mjd + self.timing_config["settings"]["fit_every_n_days"]:
                    mjds = list(self.path_data_archives.keys())[0:idx+1]
                    archives = [self.path_data_archives[i] for i in mjds]
                    break
            
            # check whether there are more files to process
            if mjds == [] or archives == []:
                return {"status": "no_files"}

            # get the fit_params
            fit_params = last_timing_info["unfreeze_params"]
            potential_fit_params = []
            n_days_to_fit = max(mjds) - min(mjds)
            for this_param_id, this_param_config in self.timing_config["params"].items():
                # if(this_param_config["min_days"] <= n_days_to_fit and this_param_config["max_days"] > n_days_to_fit):
                if(this_param_config["min_days"] <= n_days_to_fit):
                    if(this_param_id not in fit_params):
                        potential_fit_params.append(this_param_id)
                        # fit_params.append(this_param_id)
                        # break # one param a time

            if(len(fit_params) == 0):
                self.logger.error(f"No parameter to fit at n_days={n_days_to_fit}")
                return {"status": "error"} 

        # Run timing
        try:
            archives_untimed = self.db_get_untimed_archives(archives)
            self.logger.info(f"Timing module input parameters: ")
            self.logger.data(f"Timing {mjds} with archives: " + "\n -> " + "\n -> ".join(archives))
            self.logger.data(f"Fit params: {fit_params}")
            self.logger.data(f"Potential Fit params: {potential_fit_params}")
            self.logger.data(f"MJD range: {min(mjds)} - {max(mjds)}")
            self.logger.data(f"Number of Observations: {len(mjds)} ({len(archives_untimed)} untimed)")
            self.logger.data(f"Input timing model: {self.path_timing_model}")

            self.logger.success("======== Running timing modules ========")
            self.logger.level_up()
            with timing(
                ars = archives_untimed, # Only run for those archive that does not have any toas in the database
                par = self.path_timing_model,
                std = self.path_pulse_template,
                par_output = f"{self.path_timing_model}.timingoutput", 
                n_pools = self.n_pools, 
                workspace_cleanup = self.workspace_cleanup, 
                logger = self.logger.copy()
            ) as tim:
                # Processing initialize workspace
                self.logger.debug(f" > Initializing modules")
                tim.initialize()
                
                # Process archive as needed
                if len(archives_untimed) > 0:
                    ## Check if all archives are cached
                    if not self.archive_cache.archives_exists(tim.fs):
                        ### Process archives
                        self.logger.debug(f" > Preparing data")
                        tim.prepare()
                    else:
                        ### Copy from cache
                        self.logger.debug(f" > All archives are cached. Copying from cache... ")
                        for f in tim.fs:
                            self.archive_cache.get_archive(f, f"{f}.clfd.FTp")
                            self.logger.debug(f"[Archive] {f} -> {f}.clfd.FTp copied from cache. ", layer=1)

                    ## Getting TOAs
                    self.logger.debug(f" > Getting TOAs")
                    tim.get_toas()

                    ## Save TOAs to timing database
                    self.logger.debug(f" > Saving TOAs")
                    for f in tim.fs:
                        if(self.db_insert_timfile(f"{f}.clfd.FTp.tim") == 0):
                            self.logger.warning(f"No TOA created from {f}. Placeholder with INVALID_TOA remark has created. ", layer=1)
                            self.db_insert_invalid_toa(f)

                    ## Save cache archive and information to database
                    self.logger.debug(f" > Saving and caching archive information")
                    for f in tim.fs:
                        self.archive_cache.add_archive(f"{f}.clfd.FTp")

                # Create new timfile from database and overwrite the one in the workspace
                self.logger.debug(f" > Creating timfile")
                open(f"{tim.workspace}/pulsar.tim", "w").write(
                    self.db_create_timfile(archives)
                )

                # Run timing from PINT
                self.logger.debug(f" > Timing TOAs")
                tim.time(fit_params=fit_params, potential_params=potential_fit_params)
                # tim.time(fit_params=fit_params)
                
                # Insert timing info
                self.logger.debug(f"Saving timing info to database")
                self.db_insert_timing_info(archives, mjds, tim.pint)

                # Finishing and print summary
                self.logger.success(f"Timing module finished")
                self.logger.debug(tim.pint.f.get_summary(), layer=1)

                
            self.logger.level_down()
            self.logger.success("======== Timing completed ========")
        except Exception as e:
            self.logger.error(f"Timing failed for {self.path_psr_dir}. Please refer to the traceback below. ")
            self.logger.error(traceback.format_exc())
            self.noti_hdl.send_urgent_message(f"Timing failed for {self.path_psr_dir}. Please refer to the traceback in the following message. ", psr_id=self.psr_id)
            self.noti_hdl.send_code(traceback.format_exc(), psr_id=self.psr_id)
            return {"status": "error"}

        # Backup old timing model
        backup_filename = f"{self.path_timing_model_bakdir}/parfile__{time.strftime('%Y_%m_%d__%H_%M_%S', time.gmtime(last_timing_info['timestamp']))}.bak"
        backup_filename = utils.no_overwriting_name(backup_filename) # Avoid overwriting
        self.logger.debug(f" Backing up old timing model: {self.path_timing_model} > {backup_filename}")
        shutil.copy(self.path_timing_model, f"{backup_filename}")
        self.logger.debug(f" Writing new timing model > {self.path_timing_model}")
        shutil.copy(f"{self.path_timing_model}.timingoutput", self.path_timing_model)

        # Finish
        self.logger.success("Timing completed. ")
        self.logger.level_down()

        return {"status": "success"}

    def db_insert_timing_info(self, fs, mjds, pint):
        # Get PINT objects
        pint_f = pint.f
        pint_t = pint.t
        pint_bad_resids = pint.bad_resids
        pint_bad_toas = pint.bad_toas
        unfreezed_params = pint.get_unfreezed_params()

        # Get timing info
        fitted_params = pint_f.get_params_dict("all", "quantity")
        residuals = pint_f.resids.time_resids.to(u.us).value
        residuals_err = pint_t.get_errors().to(u.us).value
        residual_mjds = pint_t.get_mjds().value
        bad_residuals = pint_bad_resids.to(u.us).value
        bad_residuals_err = pint_bad_toas.get_errors().to(u.us).value
        bad_residual_mjds = pint_bad_toas.get_mjds().value
        
        # Prepare timing info for JSON dump
        fitted_params_dict = {}
        for key in fitted_params:
            try:
                fitted_params_dict[key] = float(fitted_params[key].value)
            except:
                fitted_params_dict[key] = str(fitted_params[key].value)
        residuals_list = [float(this_resid) for this_resid in residuals]
        residuals_err_list = [float(this_resid_err) for this_resid_err in residuals_err]
        residual_mjds_list = [float(this_mjd) for this_mjd in residual_mjds]
        bad_residuals_list = [float(this_resid) for this_resid in bad_residuals]
        bad_residuals_err_list = [float(this_resid_err) for this_resid_err in bad_residuals_err]
        bad_toa_mjds_list = [float(this_mjd) for this_mjd in bad_residual_mjds]

        # Prepare notes
        notes = {"remark": []}
        if not pint.f_status:
            notes["remark"].append("FITTING_FAILED")

        # Get archive ids
        archive_ids = []
        for f in fs:
            archive_ids.append(utils.get_archive_id(f))
        
        # Insert timing info
        self.db_hdl.insert_timing_info(
            files = archive_ids,
            obs_mjds = mjds,
            unfreeze_params = unfreezed_params,
            residuals = {"val": residuals_list, "err": residuals_err_list},
            chi2 = fitted_params["CHI2"].value,
            chi2_reduced = fitted_params["CHI2R"].value,
            fitted_params = fitted_params_dict,
            notes = {
                "fitted_parfile": pint_f.model.as_parfile(), 
                "fitted_summary": pint_f.get_summary(), 
                "fitted_mjds": residual_mjds_list, 
                "bad_toa_mjds": bad_toa_mjds_list, 
                "bad_toa_residuals": {"val": bad_residuals_list, "err": bad_residuals_err_list}
            }
        )
    
    def db_insert_timfile(self, timfile):
        n_inserted = 0
        timfile = open(timfile, "r").read()

        for this_toa in timfile.split("\n"):
            if "FORMAT 1" in this_toa or len(this_toa.strip()) == 0:
                continue

            splitted = []
            for this_param in this_toa.split(" "):
                if len(this_param.strip()) > 0:
                    splitted.append(this_param)
                    
            if(len(splitted) != 5):
                raise Exception("Unexpected .tim file format", this_param)
            
            self.logger.debug(f"[TOA] filename={splitted[0].split('/')[-1]}, freq={splitted[1]}, toa={splitted[2]}, toa_err={splitted[3]}, telescope={splitted[4]}", layer=1)

            self.db_hdl.insert_toa(
                filename = utils.get_archive_id(splitted[0]), # set only the filename as the index, otherwise the ws id will be different...
                freq = splitted[1], 
                toa = splitted[2], 
                toa_err = splitted[3], 
                telescope = splitted[4], 
                raw_tim = this_toa, 
                notes = {}
            )

            n_inserted += 1

        return n_inserted

    # def db_insert_archive_info(self, archive):
    #     archive_hdl = archive_utils(archive)

    #     print(f"  [Archive] {archive} -> database")
    #     self.db_hdl.insert_archive_info(
    #         filename = utils.get_archive_id(archive), 
    #         psr_amps = archive_hdl.get_amps(), 
    #         psr_snr = archive_hdl.get_snr(), 
    #         notes = {}
    #     )

        
    def db_insert_invalid_toa(self, archive):
        self.db_hdl.insert_toa(
            filename = utils.get_archive_id(archive), # set only the filename as the index, otherwise the ws id will be different...
            freq = 0, 
            toa = 0, 
            toa_err = 0, 
            telescope = "", 
            raw_tim = "", 
            notes = {"remark": "INVALID_TOA"}
        )
    
    def db_check_valid_toa(self, archive):
        toa = self.db_hdl.get_toa_by_filename(utils.get_archive_id(archive))

        if "remark" in toa["notes"]:
            if toa["notes"]["remark"] == "INVALID_TOA":
                return False
            
        return True
    
    def db_create_timfile(self, archives):
        timfile = ""

        for this_file in archives:
            this_toa = self.db_hdl.get_toa_by_filename(utils.get_archive_id(this_file))

            if(not self.db_check_valid_toa(this_file)):
                self.logger.warning(f"INVALID_TOA remark was found for {this_file}. Skipped while creating timfile...", layer=1)
                continue

            if(this_toa["timestamp"] == 0 or this_toa["raw_tim"].strip() == ""):
                raise Exception(f"TOA from archive [{this_file}] does not exist in database. ")
            
            timfile += this_toa["raw_tim"] + "\n"

        return timfile
    
    def db_get_untimed_archives(self, archives):
        untimed_archives = []

        for this_file in archives:
            if not self.db_hdl.check_toa_exists(utils.get_archive_id(this_file)):
                untimed_archives.append(this_file)

        return untimed_archives

    def cleanup(self):
        # Close DB
        self.db_hdl.close()

        # Cleanup archive_cache
        self.archive_cache.cleanup()

    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.error(f"A fetal error occurred while timing for {self.path_psr_dir}. Script exited.")
            # self.noti_hdl.send_urgent_message(f"A fetal error occurred while timing for {self.path_psr_dir}. Script exited.")
            raise exc_type(exc_value).with_traceback(traceback)
        self.cleanup()
