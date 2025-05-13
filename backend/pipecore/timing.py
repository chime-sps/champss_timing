from .psrchive import psrchive_handler
from .pint import pint_handler
from .filters import Filters
from ..utils.exec import exec
from ..utils.utils import utils
from ..utils.logger import logger
from ..processing.dspsr_shutils import dspsr_shutils

import os
import time
import glob
import itertools
from shutil import copyfile, rmtree

class timing():
    def __init__(self, ars, std, par, par_initial, jumps, dedisperse=True, par_output=False, n_pools=4, reset_params=False, filters=[], workspace_cleanup=True, logger=logger(), workspace_root="./__champss_timing__workspace"):
        # Paths & Settings
        self.ars = ars # data archives
        self.std = std # pulse template
        self.par = par # input timing model
        self.par_initial = par_initial # initial timing model
        self.jumps = jumps # jump cpt
        self.dedisperse = dedisperse # dedisperse data before timing
        self.par_output = par_output # output timing model
        self.n_pools = n_pools
        self.reset_params = reset_params
        self.filters = filters
        self.workspace_root = workspace_root
        self.workspace_cleanup = workspace_cleanup

        # Workspace
        self.workspace = f"{self.workspace_root}/{utils.get_time_string()}__{utils.get_rand_string()}"
        self.fs = []
        self.labels = []
        self.rcvrs = []

        # Functions
        self.logger = logger # logging function
        self.utils = utils # ultilities
        self.exec_handler = exec # exec handler
        self.psrchive = psrchive_handler(self) # psrchive handler
        self.pint = pint_handler(self, initialize=False) # pint handler
        self.dspsr_shutils = dspsr_shutils(n_pools=self.n_pools) # dspsr utils

        # Control
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return 
        
        # Check archive exist
        for f in self.ars:
            if not os.path.exists(f["path"]):
                raise Exception(f"Archive {f['path']} does not exist")

        # Check psrdir files exist
        for f in [self.par, self.std]:
            if not os.path.exists(f):
                raise Exception(f"Psrdir file {f} does not exist")
            
        # Create workspace
        self.logger.debug(f"Creating workspace at {self.workspace}")
        self.logger.debug(f"Press Ctrl+C to cancel")
        time.sleep(1)
        os.makedirs(self.workspace, exist_ok=True)

        # Move files to workspace
        self.logger.debug("Copying par and std to workspace... ", layer=1)
        # os.system(f"cp {self.par} {self.workspace}/pulsar.par")
        copyfile(self.par, f"{self.workspace}/pulsar.par")
        copyfile(self.par_initial, f"{self.workspace}/pulsar.par.initial")
        # os.system(f"cp {self.std} {self.workspace}/paas.std")
        copyfile(self.std, f"{self.workspace}/paas.std")

        # Getting fs, labels, rcvrs
        for i, f in enumerate(self.ars):
            self.fs.append(f"{self.workspace}/{os.path.basename(f['path'])}")
            self.labels.append(f["label"])
            self.rcvrs.append(f["rcvr"])

        # Check if parfile has name
        if not self.__parfile_has_name(f"{self.workspace}/pulsar.par"):
            raise Exception("Parfile does not have PSR name. This is required for timing for several functionalities in the pipeline. ")

        # Add jump cpt to parfile
        for rcvr in self.jumps:
            if self.jumps[rcvr][0] == 0:
                continue
            self.__parfile_add_jump_cpt(f"{self.workspace}/pulsar.par", rcvr, self.jumps[rcvr][0], self.jumps[rcvr][1])

        self.logger.debug("Initialized workspace")
        self.initialized = True

    def cleanup(self, verbose=False):
        if not self.initialized:
            raise Exception("Workspace not initialized")
        
        # Remove workspace
        self.logger.debug(f"Removing workspace at {self.workspace}")
        self.logger.debug(f"Press Ctrl+C to cancel")
        time.sleep(1)
        
        if verbose:
            # Remove files except *.log
            for f in glob.glob(f"{self.workspace}/*"):
                if ".log" not in f:
                    # os.system(f"rm {f}")
                    os.remove(f)
        else:
            # os.system(f"rm -r {self.workspace}")
            rmtree(self.workspace)
        
            # If workspace root is empty, remove it
            if(len(os.listdir(self.workspace_root)) == 0):
                # os.system(f"rm -r {self.workspace_root}")
                rmtree(self.workspace_root)

    def prepare(self):
        if not self.initialized:
            raise Exception("Workspace not initialized")

        # Copy archives to workspace
        self.logger.debug("Copying archives to workspace... ")
        for i, f in enumerate(self.ars):
            self.logger.debug(f"Copying file {i+1}/{len(self.ars)}", end="\r", layer=2)
            copyfile(f['path'], self.fs[i])

        # Dedisperse
        if self.dedisperse:
            self.logger.debug("Dedispersing... ")
            fs_no_filterbank = []
            for f in self.fs:
                if not f.endswith(".fil"):
                    fs_no_filterbank.append(f)

            # Dedisperse
            self.logger.debug(f"Dedispersing... ")
            self.psrchive.dedisperse(fs_no_filterbank, f"{self.workspace}/pulsar.par.initial")

            # # get dm from parfile
            # self.logger.debug("Reading DM from parfile... ")
            # parfile_dm = self.utils.read_dm_from_parfile(f"{self.workspace}/pulsar.par.initial", raise_exception=True)

            # # Dedisperse (legacy)
            # self.logger.debug(f"Dedispersing with DM = {parfile_dm}... ")
            # self.psrchive.dedisperse(self.fs, parfile_dm)
        
        # Convert filterbank to ar as needed. 
        for i, f in enumerate(self.fs):
            if f.endswith(".fil"):
                self.logger.debug(f"Converting {f} -> {f}.ar... ")
                self.dspsr_shutils.fil2ar(f, f, f"{self.workspace}/pulsar.par.initial")
                self.fs[i] = f"{f}.ar"
        
        self.logger.debug("Zapping bad channels...")
        self.psrchive.zap_bad_channel(self.fs)
        self.logger.debug("Zapping bad channels... Done. ")

        self.logger.debug("Scrunching...")
        self.psrchive.scrunch(self.fs)
        self.logger.debug("Scrunching... Done. ")

        if "champss" in self.filters:
            self.logger.debug("Applying CHAMPSS filter... ")
            Filters(
                ars=[f + ".clfd.FTp" for f in self.fs], 
                filters=["champss"], 
                std_profile=f"{self.workspace}/paas.std", 
                n_pools=self.n_pools,
                logger=self.logger.copy()
            ).filter()
            self.logger.debug("Applying CHAMPSS filter... Done. ")
    
    def get_toas(self):
        self.logger.debug("Getting TOAs...")
        self.psrchive.get_toas(self.fs, template=f"{self.workspace}/paas.std", output=f"{self.workspace}/pulsar.tim")
        self.logger.debug("Getting TOAs... Done. ")

    def time(self, fit_params="auto", potential_params=[], mcmc_report=None):
        if not self.initialized:
            raise Exception("Workspace not initialized")
        
        self.logger.debug("Initializing PINT handler... ")
        self.pint.initialize()

        if fit_params != "auto":
            self.logger.debug(f"Unfreezing parameters... {fit_params}", layer=1)
            self.pint.freeze_all()
            for p in fit_params:
                self.pint.unfreeze(p)

        if self.pint.check_toa_gaps(latest_n_days=3, threshold=15):
            potential_params = [] # Not adding parameter after a huge gap

        if self.pint.check_toa_gaps(latest_n_days=5, threshold=30):
            potential_params = [] # Not adding parameter after a huge gap
        
        if len(potential_params) > 0:
            # Run F-test
            f_test_res = {"params": [], "p_values": []}
            for i in range(len(potential_params)): # Loop through all combinations of parameters
                for param_comb in itertools.combinations(potential_params, i + 1):
                    # Skip if F-test is not possible
                    if "F2" in param_comb and "F1" not in fit_params:
                        continue
                    if "F3" in param_comb and ("F2" not in fit_params or "F1" not in fit_params):
                        continue

                    # Run trial fit and get p-value
                    this_pass, this_p_value = self.pint.f_test(param_comb)

                    # Append results
                    if this_pass: # Only append if the null hypothesis is rejected (this_pass=True)
                        f_test_res["params"].append(param_comb)
                        f_test_res["p_values"].append(this_p_value)

                    self.logger.debug(f"Testing parameter {param_comb}... Done. (passed = {this_pass}, p-value = {this_p_value}")
            
            # Find lowest p-value combination
            if(len(f_test_res["p_values"]) > 0):
                best_comb = f_test_res["params"][f_test_res["p_values"].index(min(f_test_res["p_values"]))]
                for this_param in best_comb:
                    self.pint.unfreeze(this_param)
                self.logger.debug(f"Best combination of parameters by adding {best_comb} (from p-values) ")

        # if(len(potential_params) > 1):
        #     # Run trial fit
        #     for this_param in potential_params:
        #         self.logger.debug(f"Running trial fit for {this_param}... ")
        #         if self.pint.trial_fit([this_param]):
        #             fit_params.append(this_param)
        #             self.logger.debug(f"Running trial fit for {this_param}... Passed. ")
        #             break # add one param each time. 
        #         else:
        #             self.logger.warning(f"Running trial fit for {this_param}... Failed. ")
        #     self.logger.info(f"Fit parameters: {fit_params}. ")
        #     for p in fit_params:
        #         self.pint.unfreeze(p)

        self.logger.debug("Filtering TOAs... ")
        self.pint.filter()

        self.logger.debug("Fitting TOAs... ")
        self.pint.fit(raise_exception=False)

        if mcmc_report is not None:
            self.logger.debug("Running MCMC... ")
            self.pint.fit_mcmc_report(mcmc_report)

        self.logger.debug("Plotting residuals... ")
        self.pint.plot()

        self.logger.debug("Saving model... ")
        self.pint.save()

        self.logger.debug("Done. ")

    def __parfile_has_name(self, parfile):
        parfile_content = open(parfile, "r").read()
        for line in parfile_content.split("\n"):
            if line.strip().startswith("PSR") or line.strip().startswith("PSRJ"): 
                return True
        return False
    
    def __parfile_add_jump_cpt(self, parfile, rcvr, jump, jump_err=0):
        self.logger.info(f"Adding JUMP ({jump} +/- {jump_err}) to {rcvr} in {parfile}")

        # read parfile
        parfile_content = open(parfile, "r").read()

        # check if jump exists
        for line in parfile_content.split("\n"):
            if line.strip().strip().startswith("JUMP"):
                if rcvr in line:
                    return parfile

        # add jump cpt
        if not parfile_content.endswith("\n"):
            parfile_content += "\n"
        parfile_content += f"JUMP       -rcvr {rcvr}       {jump} 0 {jump_err}\n"
        
        # overwrite parfile
        open(parfile, "w").write(parfile_content)

        return parfile

    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.error(f"Error while running timing. Workspace will NOT be removed at {self.workspace}")
            raise exc_type(exc_value).with_traceback(traceback)
        else:
            if self.workspace_cleanup:
                self.cleanup()
        return False