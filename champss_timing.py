import os
import json
import shutil
import astropy.units as u

from .utils import utils
from .timing import timing
from .database import database
from .archive_utils import archive_utils

class champss_timing:
    def __init__(self, psr_dir, data_archives, n_pools=4, workspace_cleanup=True):
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
        self.path_data_archives = data_archives
        self.path_db = f"{self.path_psr_dir}/champss_timing.sqlite3.db"
        self.path_pulse_template = f"{self.path_psr_dir}/paas.std"
        self.path_timing_model = f"{self.path_psr_dir}/pulsar.par"
        self.path_timing_config = f"{self.path_psr_dir}/champss_timing.config"
        self.info_ars_mjds = []
        self.info_ars_paths = []
        self.info_first_mjd = 0
        self.info_last_mjd = 0
        self.n_pools = n_pools
        self.workspace_cleanup = workspace_cleanup

        # Objects
        self.db_hdl = database(self.path_db)

        # Timing config
        self.timing_config = {}

    def initialize(self):
        # Initialize DB
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

    def run(self):
        while True:
            if(self.timing()["status"] != "success"):
                break
    
    def timing(self):
        print("Starting timing... ")
        # Get last timing info
        last_timing_info = self.db_hdl.get_last_timing_info()

        # Check if last timing info exists
        mjds = []
        archives = []
        if last_timing_info["timestamp"] == 0:
            utils.print_info(" No timing info found, starting from scratch. ")
            mjds = list(self.path_data_archives.keys())[0:5]
            archives = [self.path_data_archives[mjd] for mjd in mjds]
            fit_params = ["F0"]
        else:
            utils.print_info(f" Last timing info found: ")
            print("  Timestamp", last_timing_info["timestamp"])
            print("  n_obs", len(last_timing_info["files"]))
            print("  obs_mjds", last_timing_info["obs_mjds"])
            print("  unfreeze_params", last_timing_info["unfreeze_params"])
            print("  chi2", last_timing_info["chi2"])
            print("  chi2_reduced", last_timing_info["chi2_reduced"])

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
            fit_params = []
            n_days_to_fit = max(mjds) - min(mjds)
            for this_param_id, this_param_config in self.timing_config["params"].items():
                # if(this_param_config["min_days"] <= n_days_to_fit and this_param_config["max_days"] > n_days_to_fit):
                if(this_param_config["min_days"] <= n_days_to_fit):
                    fit_params.append(this_param_id)

            if(len(fit_params) == 0):
                utils.print_error(f"No parameter to fit at n_days={n_days_to_fit}")
                return {"status": "error"} 

        # Run timing
        try:
            archives_untimed = self.db_get_untimed_archives(archives)
            utils.print_info(f" Timing module input parameters: ")
            print(f"  Timing {mjds} with archives: " + "\n -> " + "\n -> ".join(archives))
            print(f"  Fit params: {fit_params}")
            print(f"  MJD range: {min(mjds)} - {max(mjds)}")
            print(f"  Number of Observations: {len(mjds)} ({len(archives_untimed)} untimed)")
            print(f"  Input timing model: {self.path_timing_model}")

            utils.print_success("======== Running timing modules ========")
            with timing(
                ars = archives_untimed, # Only run for those archive that does not have any toas in the database
                par = self.path_timing_model,
                std = self.path_pulse_template,
                par_output = f"{self.path_timing_model}.timingoutput", 
                n_pools = self.n_pools, 
                workspace_cleanup = self.workspace_cleanup
            ) as tim:
                # Processing initialize workspace
                print(f" > Initializing modules")
                tim.initialize()
                
                # Process archive as needed
                if len(archives_untimed) > 0:
                    ## Getting TOAs
                    print(f" > Preparing data")
                    tim.prepare()

                    ## Save TOAs to timing database
                    print(f" > Saving TOAs")
                    for f in tim.fs:
                        if(self.db_insert_timfile(f"{f}.clfd.FTp.tim") == 0):
                            utils.print_warning(f"No TOA created from {f}. Placeholder with INVALID_TOA remark has created. ")
                            self.db_insert_invalid_toa(f)
                    
                    ## Update timing model for archives (for pulse alignment purpose in archive info)
                    print(f" > Updating timing model for archives")
                    tim.update_model()

                    ## Save archive information to database
                    print(f" > Saving archive information")
                    for f in tim.fs:
                        self.db_insert_archive_info(f"{f}.clfd.FTp")

                # Create new timfile from database and overwrite the one in the workspace
                print(f" > Creating timfile")
                open(f"{tim.workspace}/pulsar.tim", "w").write(
                    self.db_create_timfile(archives)
                )

                # Run timing from PINT
                print(f" > Timing TOAs")
                tim.time(fit_params=fit_params)

                # Finishing and print summary
                print(f" Timing completed")
                tim.pint.f.print_summary()

                # Insert timing info
                self.db_insert_timing_info(archives, mjds, fit_params, tim.pint.f, tim.pint.t)
            utils.print_success("======== Timing completed ========")
        except Exception as e:
            print(" Timing failed")
            print(f" Error: {e}")
            return {"status": "error"}

        # Backup old timing model
        print(f" Backing up old timing model: {self.path_timing_model} > {self.path_timing_model}.bak{last_timing_info['timestamp']}")
        shutil.copy(self.path_timing_model, f"{self.path_timing_model}.bak{last_timing_info['timestamp']}")
        print(f" Writing new timing model > {self.path_timing_model}")
        shutil.copy(f"{self.path_timing_model}.timingoutput", self.path_timing_model)

        # Finish
        utils.print_success("Timing completed. ")

        return {"status": "success"}

    def db_insert_timing_info(self, fs, mjds, fit_params, pint_f, pint_t):
        # Get timing info
        fitted_params = pint_f.get_params_dict("all", "quantity")
        residuals = pint_f.resids.time_resids.to(u.us).value
        residual_mjds = pint_t.get_mjds().value
        
        # Prepare timing info for JSON dump
        fitted_params_dict = {}
        for key in fitted_params:
            try:
                fitted_params_dict[key] = float(fitted_params[key].value)
            except:
                fitted_params_dict[key] = str(fitted_params[key].value)
        residuals_list = [float(this_resid) for this_resid in residuals]
        residual_mjds_list = [float(this_mjd) for this_mjd in residual_mjds]

        # Get archive ids
        archive_ids = []
        for f in fs:
            archive_ids.append(self.utils.get_archive_id(f))
        
        # Insert timing info
        self.db_hdl.insert_timing_info(
            files = archive_ids,
            obs_mjds = mjds,
            unfreeze_params = fit_params,
            residuals = residuals_list,
            chi2 = fitted_params["CHI2"].value,
            chi2_reduced = fitted_params["CHI2R"].value,
            fitted_params = fitted_params_dict,
            notes = {
                "fitted_parfile": pint_f.model.as_parfile(), 
                "fitted_summary": pint_f.get_summary(), 
                "fitted_mjds": residual_mjds_list
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
            
            print(f"  [TOA] filename={splitted[0].split('/')[-1]}, freq={splitted[1]}, toa={splitted[2]}, toa_err={splitted[3]}, telescope={splitted[4]}")

            self.db_hdl.insert_toa(
                filename = self.utils.get_archive_id(splitted[0]), # set only the filename as the index, otherwise the ws id will be different...
                freq = splitted[1], 
                toa = splitted[2], 
                toa_err = splitted[3], 
                telescope = splitted[4], 
                raw_tim = this_toa, 
                notes = {}
            )

            n_inserted += 1

        return n_inserted

    def db_insert_archive_info(self, archive):
        archive_hdl = archive_utils(archive)

        print(f"  [Archive] {archive} -> database")
        self.db_hdl.insert_archive_info(
            filename = self.utils.get_archive_id(archive), 
            psr_amps = archive_hdl.get_amps(), 
            psr_snr = archive_hdl.get_snr(), 
            notes = {}
        )

        
    def db_insert_invalid_toa(self, archive):
        self.db_hdl.insert_toa(
            filename = self.utils.get_archive_id(archive), # set only the filename as the index, otherwise the ws id will be different...
            freq = 0, 
            toa = 0, 
            toa_err = 0, 
            telescope = "", 
            raw_tim = "", 
            notes = {"remark": "INVALID_TOA"}
        )
    
    def db_check_valid_toa(self, archive):
        toa = self.db_hdl.get_toa_by_filename(self.utils.get_archive_id(archive))

        if "remark" in toa["notes"]:
            if toa["notes"]["remark"] == "INVALID_TOA":
                return False
            
        return True
    
    def db_create_timfile(self, archives):
        timfile = ""

        for this_file in archives:
            this_toa = self.db_hdl.get_toa_by_filename(self.utils.get_archive_id(this_file))

            if(not self.db_check_valid_toa(this_file)):
                utils.print_warning(f"INVALID_TOA remark was found for {this_file}. Skipped while creating timfile...")
                continue

            if(this_toa["timestamp"] == 0 or this_toa["raw_tim"].strip() == ""):
                raise Exception(f"TOA from archive [{this_file}] does not exist in database. ")
            
            timfile += this_toa["raw_tim"] + "\n"

        return timfile
    
    def db_get_untimed_archives(self, archives):
        untimed_archives = []

        for this_file in archives:
            if not self.db_hdl.check_toa_exists(self.utils.get_archive_id(this_file)):
                untimed_archives.append(this_file)

        return untimed_archives

    def cleanup(self):
        # Close DB
        self.db_hdl.close()

    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            utils.print_error("champss_timing failed with error. ")
            raise exc_type(exc_value).with_traceback(traceback)
        self.cleanup()