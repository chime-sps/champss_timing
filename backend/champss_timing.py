import os
import json
import time
import shutil
import traceback
import astropy.units as u
import numpy as np

from .champss_checker import champss_checker
from .io.archive import ArchiveReader
from .pipecore.timing import timing
from .pipecore.plot import plot
from .pipecore.config import config
from .datastores.database import database
from .datastores.archive_cache import archive_cache
from .utils.logger import logger
from .utils.utils import utils
from .utils.notification import notification

class champss_timing:
    def __init__(self, psr_dir, data_archives, toa_jumps={}, slack_token=False, timing_mode="opd", n_pools=4, workspace_cleanup=True, logger=logger()):
        """
        CHAMPSS timing pipeline

        Parameters
        ----------
        psr_dir : str
            Path to the directory containing the pulsar data
        data_archives : dict
            List of data archives to be processed
        slack_token : dict
            Slack token for notification
        timing_mode : str
            Timing mode to be used (opd, mpd)
            opd: One TOA a day
            mpd: Multiple TOAs a day
        n_pools : int
            Number of pools to be used for multiprocessing
        workspace_cleanup : bool
            Clean up workspace after timing
        logger : logger
            Logger object for logging

        Returns
        -------
        None
        """

        # Parameters
        self.psr_id = None
        self.toa_jumps = toa_jumps
        self.path_psr_dir = psr_dir
        self.path_data_archives = data_archives
        self.path_db = f"{self.path_psr_dir}/champss_timing.sqlite3.db"
        self.path_pulse_template = f"{self.path_psr_dir}/paas.std"
        self.path_timing_model = f"{self.path_psr_dir}/pulsar.par"
        self.path_timing_model_bakdir = f"{self.path_psr_dir}/parfile_bak"
        self.path_timing_model_initial = f"{self.path_timing_model_bakdir}/initial_parfile.bak"
        self.path_timing_config = f"{self.path_psr_dir}/champss_timing.config"
        self.path_diagnostic_plot = f"{self.path_psr_dir}/champss_diagnostic.pdf"
        self.path_mcmc_report = f"{self.path_psr_dir}/mcmc_report.pdf"
        self.info_ars_mjds = []
        self.info_ars_paths = []
        self.info_first_mjd = 0
        self.info_last_mjd = 0

        # Settings
        self.n_pools = n_pools
        self.logger = logger
        self.workspace_root = "./__champss_timing__workspace"
        self.tempfolder = "auto"
        self.workspace_cleanup = workspace_cleanup
        self.timing_mode = timing_mode

        # If slurm is used, set workspace to $SLURM_TMPDIR
        if "SLURM_TMPDIR" in os.environ:
            if os.environ["SLURM_TMPDIR"] != "" and os.path.isdir(os.environ["SLURM_TMPDIR"]):
                self.workspace = os.environ["SLURM_TMPDIR"]
                self.tempfolder = f"{self.workspace}/temp"
                self.workspace_cleanup = False

                logger.info(f"SLURM detected. Setting workspace to {self.workspace} and tempfolder to {self.tempfolder}")

        # Format psr_dir
        if self.path_psr_dir.endswith("/"):
            self.path_psr_dir = self.path_psr_dir[0:-1]

        # Objects
        self.db_hdl = database(self.path_db, logger=self.logger)
        self.archive_cache = archive_cache(self.path_psr_dir, db_hdl=self.db_hdl)
        self.noti_hdl = notification(slack_token)

        # Timing config
        self.timing_config = {}

    def initialize(self):
        # Print git version
        self.logger.success(f"CHAMPSS Timing Pipeline ({utils.get_version_hash()})")

        # Initialize DB
        self.db_hdl.initialize()

        # Initialize archive_cache
        self.archive_cache.initialize()

        # Initialize archive_cache
        self.db_hdl.initialize()

        # Clear logger cache
        self.logger.clear_log_cache()
        
        # Load config
        self.timing_config = config(self.path_timing_config, self.logger.copy(), db_hdl=self.db_hdl).to_dict()

        # Get psr id
        self.psr_id = self.path_psr_dir.split("/")[-1]
        
        # Check number of data archives
        if len(self.path_data_archives) < 2:
            raise ValueError("No enough data archives to perform timing (at least 2 needed)")

        # Sort data archives by MJD key
        self.path_data_archives = {k: v for k, v in sorted(self.path_data_archives.items(), key=lambda item: item[0])}
        # self.path_data_archives = dict(sorted(self.path_data_archives.items()))

        # ignore archive that has mjds earlier than self.timing_config["ignore_mjds"]["earlier_than"]
        for mjd in list(self.path_data_archives.keys()):
            if mjd < self.timing_config["ignore_mjds"]["earlier_than"]:
                self.logger.debug(f"Archive mjd:{mjd} is ignored due to the earlier_than setting in the config file. ")
                del self.path_data_archives[mjd]
            if mjd > self.timing_config["ignore_mjds"]["later_than"]:
                self.logger.debug(f"Archive mjd:{mjd} is ignored due to the later_than setting in the config file. ")
                del self.path_data_archives[mjd]

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
        for mjd in self.path_data_archives:
            for this_archive_info in self.path_data_archives[mjd]:
                if not os.path.isfile(this_archive_info["path"]):
                    raise FileNotFoundError(f"File {this_archive_info['path']} not found for data archive")

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
                    self.logger.success(f"No additional file for timing. ")
                    break

                # Remove existing diagnostic plot, then will be created again below
                if os.path.isfile(self.path_diagnostic_plot):
                    os.remove(self.path_diagnostic_plot)

                break
            n_timed += 1

        # If no diagnostic plot, create one
        if not os.path.isfile(self.path_diagnostic_plot):
            # Update model for cached archives
            self.logger.debug(f"Updating model for all cached archives")
            self.archive_cache.update_model(jumps=self.toa_jumps, n_pools=self.n_pools, tempdir=self.tempfolder)

            # Create diagnostic plot
            self.logger.info(f"Creating diagnostic plot")
            plot(db_hdl=self.db_hdl).diagnostic(savefig=self.path_diagnostic_plot)

            # Run checker
            champss_checker(self.path_psr_dir, self.db_hdl, self.noti_hdl, self.psr_id).check(send_noti=True)

            # End of the script
            self.logger.success("Script finished. ")
            self.noti_hdl.send_message(f"Timing finished for {n_timed} days of data. ", psr_id=self.psr_id)
                
            # Save log
            self.logger.info(f"Saving log to {self.path_psr_dir}/champss_timing.log")
            self.logger.save_log(f"{self.path_psr_dir}/champss_timing.log")

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
            mjds = list(self.path_data_archives.keys())[0:2]
            # mjds = self.get_densiest_mjds(list(self.path_data_archives.keys()))
            archives = [self.path_data_archives[mjd] for mjd in mjds]
            fit_params = ["F0"]
            potential_fit_params = []

            # Create initial parfile
            if os.path.isfile(self.path_timing_model_initial):
                if open(self.path_timing_model_initial).read() != open(self.path_timing_model).read():
                    raise Exception(f"Initial parfile {self.path_timing_model_initial} exists and does not match the current timing model. Please remove the file or update the file to match the current timing model. ")
            else:
                shutil.copy(self.path_timing_model, self.path_timing_model_initial)
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
            
            if mjds == [] or archives == []:
                if len(last_timing_info["obs_mjds"]) != len(self.path_data_archives):
                    missing_mjds = []
                    for mjd in self.path_data_archives:
                        if mjd not in last_timing_info["obs_mjds"]:
                            missing_mjds.append(mjd)
                    self.logger.warning(f"No additional file since the last timing. However, not all files are processed. ")
                    self.logger.warning(f"This warning may be fixed by the next timing. ")
                    self.logger.warning(f"If this warning presists, restart timing for this source from scratch may fix the issue. ")
                    self.noti_hdl.send_urgent_message(f"No additional file since the last timing, but not all files are processed. Restart timing for this source from scratch would fix the issue (missing_mjds={missing_mjds}). ", psr_id=self.psr_id)
                return {"status": "no_files"}

            # The method below may not work properly with plotting and not very useful in current implementation (since we run timing every day)
            # # find the nearest mjd to the last mjd
            # mjds = last_timing_info["obs_mjds"] + [self.get_nearest_mjd(list(self.path_data_archives.keys()), last_timing_info["obs_mjds"])]
            # archives = [self.path_data_archives[i] for i in mjds]
            # # check whether there are more files to process
            # if len(last_timing_info["obs_mjds"]) == len(self.path_data_archives):
            #     return {"status": "no_files"}

            # get the fit_params
            # fit_params = last_timing_info["unfreeze_params"]
            # potential_fit_params = []
            # n_days_to_fit = max(mjds) - min(mjds)
            # for this_param_id, this_param_config in self.timing_config["params"].items():
            #     # if(this_param_config["min_days"] <= n_days_to_fit and this_param_config["max_days"] > n_days_to_fit):
            #     if(this_param_config["min_days"] <= n_days_to_fit):
            #         if(this_param_id not in fit_params):
            #             potential_fit_params.append(this_param_id)
            #             # fit_params.append(this_param_id)
            #             # break # one param a time
            fit_params, potential_fit_params = self.get_fit_parameters(last_timing_info, n_days_to_fit=max(mjds) - min(mjds))

            if len(fit_params) == 0:
                self.logger.error(f"No parameter to fit at n_days={n_days_to_fit}")
                return {"status": "error"} 

        # Check timing mode
        ar_list = []
        if self.timing_mode == "opd":
            for i, ar_info in enumerate(archives):
                ar_list.append(ar_info[0])
                self.logger.debug(f"OPD: MJD{ar_info[0]['mjd']} -> {ar_info[0]['label']} ({utils.get_archive_id(ar_info[0]['path'])})")
        elif self.timing_mode == "mpd":
            for ar_info in archives:
                ar_list += ar_info
        else:
            raise ValueError("Invalid timing mode. ")

        # Check if mcmc report is needed
        mcmc_report = None # None = not generate the report
        if os.path.isfile(self.path_mcmc_report):
            # Get last mcmc report modified date
            mcmc_report_mtime = os.path.getmtime(self.path_mcmc_report)

            # Re-generate the report every 30 days after the first report been generated
            if time.time() - mcmc_report_mtime > 30 * 24 * 60 * 60:
                mcmc_report = self.path_mcmc_report
        else:
            # Generate the first report once there are more than 3 fit params
            if len(fit_params) >= 3:
                mcmc_report = self.path_mcmc_report
                

        # Run timing
        try:
            ar_list_untimed = self.db_get_untimed_archives(ar_list)
            self.logger.info(f"Timing module input parameters: ")
            self.logger.data(f"Timing {mjds} with archives: " + "\n -> " + "\n -> ".join([f"{this_ar['path']}" for this_ar in ar_list]))
            self.logger.data(f"Fit params: {fit_params}")
            self.logger.data(f"Potential Fit params: {potential_fit_params}")
            self.logger.data(f"MJD range: {min(mjds)} - {max(mjds)}")
            self.logger.data(f"Number of Observations: {len(mjds)} ({len(ar_list_untimed)} untimed)")
            self.logger.data(f"Input timing model: {self.path_timing_model}")

            self.logger.success("======== Running timing modules ========")
            self.logger.level_up()
            with timing(
                ars = ar_list_untimed, # Only run for those archive that does not have any toas in the database
                par = self.path_timing_model,
                par_initial = self.path_timing_model_initial,
                std = self.path_pulse_template,
                jumps = self.toa_jumps,
                dedisperse = True, 
                par_output = f"{self.path_timing_model}.timingoutput", 
                n_pools = self.n_pools, 
                reset_params = self.timing_config["settings"]["reset_params"],
                filters = self.timing_config["settings"]["use_filters"],
                workspace_cleanup = self.workspace_cleanup, 
                logger = self.logger.copy(),
                workspace_root = self.workspace_root
            ) as tim:
                # Processing initialize workspace
                self.logger.debug(f" > Initializing modules")
                tim.initialize()
                
                # Process archive as needed
                if len(ar_list_untimed) > 0:
                    ## Check if all archives are cached
                    if not self.archive_cache.archives_exists(tim.fs):
                        ### Process archives
                        self.logger.debug(f" > Preparing data")
                        tim.prepare()
                        ### Sanity check for bad_percentage
                        if (np.array(tim.psrchive.bad_percentage) > 0.70).any():
                            self.noti_hdl.send_urgent_message(f"Bad channel percentage > 70% (usually ~30% for CHIME/Pulsar, ~50% for CHAMPSS). Please check the diagnostic plot. ", psr_id=self.psr_id)
                            self.noti_hdl.send_code(tim.psrchive.bad_percentage, psr_id=self.psr_id)
                        if (np.array(tim.psrchive.bad_percentage) < 0.65).any():
                            self.noti_hdl.send_urgent_message(f"Bad channel percentage < 5% (usually ~30% for CHIME/Pulsar, ~50% for CHAMPSS). Please check the diagnostic plot. ", psr_id=self.psr_id)
                            self.noti_hdl.send_code(tim.psrchive.bad_percentage, psr_id=self.psr_id)
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
                        if(self.db_insert_timfile(f"{f}.clfd.FTp.tim", ar_list) == 0):
                            self.logger.warning(f"No TOA created from {f}. Placeholder with INVALID_TOA remark has created. ", layer=1)
                            self.db_insert_invalid_toa(f)

                    ## Save cache archive and information to database
                    self.logger.debug(f" > Saving and caching archive information")
                    for i, f in enumerate(tim.fs):
                        self.archive_cache.add_archive(f"{f}.clfd.FTp", tim.rcvrs[i])

                # Create new timfile from database and overwrite the one in the workspace
                self.logger.debug(f" > Creating timfile")
                open(f"{tim.workspace}/pulsar.tim", "w").write(
                    self.db_create_timfile(ar_list)
                )

                # Run timing from PINT
                self.logger.debug(f" > Timing TOAs")
                tim.time(fit_params=fit_params, potential_params=potential_fit_params, mcmc_report=mcmc_report)
                # tim.time(fit_params=fit_params)
                
                # Insert timing info
                self.logger.debug(f"Saving timing info to database")
                self.db_insert_timing_info(ar_list, mjds, tim.pint)

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

    def db_insert_timing_info(self, ar_list, mjds, pint):
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
        for ar_info in ar_list:
            archive_ids.append(utils.get_archive_id(ar_info["path"]))
        
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
    
    def db_insert_timfile(self, timfile, ar_list):
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

            ar_info = {}
            for ar_info_ in ar_list:
                if utils.get_archive_id(ar_info_["path"]) == utils.get_archive_id(splitted[0]):
                    ar_info = ar_info_
                    break
            
            if ar_info == {}:
                raise Exception(f"Archive {splitted[0]} not found in the archive list (unknown TOA). ")
            
            self.logger.debug(f"[TOA] filename={utils.get_archive_id(splitted[0])}, freq={splitted[1]}, toa={splitted[2]}, toa_err={splitted[3]}, telescope={splitted[4]}, label={ar_info['label']}", layer=1)

            self.db_hdl.insert_toa(
                filename = utils.get_archive_id(splitted[0]), # set only the filename as the index, otherwise the ws id will be different...
                freq = splitted[1], 
                toa = splitted[2], 
                toa_err = splitted[3], 
                telescope = splitted[4], 
                raw_tim = this_toa, 
                notes = {
                    "label": ar_info["label"], 
                    "rcvr": ar_info["rcvr"]
                }
            )

            n_inserted += 1

        return n_inserted

    # def db_insert_archive_info(self, archive):
    #     archive_hdl = ArchiveReader(archive)

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
    
    # def db_check_valid_toa(self, archive):
    #     toa = self.db_hdl.get_toa_by_filename(utils.get_archive_id(archive))

    #     if "remark" in toa["notes"]:
    #         if toa["notes"]["remark"] == "INVALID_TOA":
    #             return False
            
    #     return True
    
    def db_create_timfile(self, ar_list):
        return self.db_hdl.create_timfile(ar_list=ar_list)
        # timfile = ""

        # for ar_info in ar_list:
        #     this_toa = self.db_hdl.get_toa_by_filename(utils.get_archive_id(ar_info["path"]))

        #     if(not self.db_check_valid_toa(ar_info["path"])):
        #         self.logger.warning(f"INVALID_TOA remark was found for {ar_info['path']}. Skipped while creating timfile...", layer=1)
        #         continue

        #     if(this_toa["timestamp"] == 0 or this_toa["raw_tim"].strip() == ""):
        #         raise Exception(f"TOA from archive [{ar_info['path']}] does not exist in database. ")
            
        #     timfile += this_toa["raw_tim"] + f" -rcvr {this_toa['notes']['rcvr']} " + "\n"

        # return timfile
    
    def db_get_untimed_archives(self, ar_list):
        untimed_archives = []

        # for ar_info in archives:
        #     for i, this_ar_info in enumerate(ar_info):
        #         if self.db_hdl.check_toa_exists(utils.get_archive_id(this_ar_info["path"])):
        #             break
        #         if i == len(ar_info) - 1:
        #             # If loop reaches the end, then the archive is untimed since no TOA is found so that no break is called.
        #             untimed_archives.append(ar_info)

        for ar_info in ar_list:
            if not self.db_hdl.check_toa_exists(utils.get_archive_id(ar_info["path"])):
                untimed_archives.append(ar_info)

        return untimed_archives

    def get_densiest_mjds(self, mjds):
        if len(mjds) <= 5:
            return mjds
        
        stds = []
        for i in range(len(mjds) - 5):
            stds.append(np.std(mjds[i:i+5]))
        min_std_i = np.where(stds == np.min(stds))[0][0]
        return mjds[min_std_i:min_std_i+5]

    def get_nearest_mjd(self, mjds, last_mjds):
        for mjd in last_mjds:
            if mjd in mjds:
                # remove the mjd from the list
                mjds.remove(mjd)

        if len(mjds) == 0:
            return []

        return mjds[np.argmin(np.abs(np.array(mjds) - np.mean(last_mjds)))]

    def get_fit_parameters(self, last_timing_info, n_days_to_fit):
        fit_params = last_timing_info["unfreeze_params"]
        potential_fit_params = []

        if "F0" in self.timing_config["settings"]["fit_params"]:
            if "F0" not in fit_params:
                fit_params.append("F0")

        if "DECJ" in self.timing_config["settings"]["fit_params"]:
            if "DECJ" not in fit_params:
                if n_days_to_fit >= 30:
                    potential_fit_params.append("DECJ")
        
        if "RAJ" in self.timing_config["settings"]["fit_params"]:
            if "RAJ" not in fit_params:
                if n_days_to_fit >= 30:
                    potential_fit_params.append("RAJ")

        if "F1" in self.timing_config["settings"]["fit_params"]:
            if "F1" not in fit_params:
                if n_days_to_fit >= 60:
                    potential_fit_params.append("F1")

        if "PX" in self.timing_config["settings"]["fit_params"]:
            if "PX" not in fit_params:
                if n_days_to_fit >= 300:
                    potential_fit_params.append("PX")

        if "F2" in self.timing_config["settings"]["fit_params"]:
            if "F2" not in fit_params and "F1" in fit_params:
                if n_days_to_fit >= 500:
                    potential_fit_params.append("F2")

        if "F3" in self.timing_config["settings"]["fit_params"]:
            if "F3" not in fit_params and "F2" in fit_params and "F1" in fit_params:
                if n_days_to_fit >= 600:
                    potential_fit_params.append("F3")

        if "PMDEC" in self.timing_config["settings"]["fit_params"]:
            if "PMDEC" not in fit_params:
                if n_days_to_fit >= 700:
                    potential_fit_params.append("PMDEC")

        if "PMRA" in self.timing_config["settings"]["fit_params"]:
            if "PMRA" not in fit_params:
                if n_days_to_fit >= 800:
                    potential_fit_params.append("PMRA")

        return fit_params, potential_fit_params

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
