from .psrchive import psrchive_handler
from .pint import pint_handler
from .exec import exec
from .utils import utils
from .logger import logger

import os
import time
import glob
import itertools
from shutil import copyfile, rmtree

class timing():
    def __init__(self, ars, std, par, par_output=False, n_pools=4, workspace_cleanup=True, logger=logger()):
        # Paths & Settings
        self.ars = ars # data archives
        self.std = std # pulse template
        self.par = par # input timing model
        self.par_output = par_output # output timing model
        self.n_pools = n_pools
        self.workspace_cleanup = workspace_cleanup

        # Workspace
        self.workspace_root = "./__champss_timing__workspace"
        self.workspace = f"{self.workspace_root}/{utils.get_time_string()}__{utils.get_rand_string()}"
        self.fs = []

        # Functions
        self.logger = logger # logging function
        self.utils = utils # ultilities
        self.exec_handler = exec # exec handler
        self.psrchive = psrchive_handler(self) # psrchive handler
        self.pint = pint_handler(self, initialize=False) # pint handler

        # Control
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return 
        
        # Check files exist
        for f in self.ars + [self.par, self.std]:
            if not os.path.exists(f):
                raise Exception(f"File {f} does not exist")
            
        # Create workspace
        self.logger.debug(f"Creating workspace at {self.workspace}")
        self.logger.debug(f"Press Ctrl+C to cancel")
        time.sleep(1)
        os.makedirs(self.workspace, exist_ok=True)

        # Move files to workspace
        self.logger.debug("Copying files to workspace... ", layer=1)
        for i, f in enumerate(self.ars):
            self.logger.debug(f"Copying file {i+1}/{len(self.ars)}", end="\r", layer=2)
            # os.system(f"cp {f} {self.workspace}")
            copyfile(f, f"{self.workspace}/{os.path.basename(f)}")
            self.fs.append(f"{self.workspace}/{os.path.basename(f)}")
        # os.system(f"cp {self.par} {self.workspace}/pulsar.par")
        copyfile(self.par, f"{self.workspace}/pulsar.par")
        # os.system(f"cp {self.std} {self.workspace}/paas.std")
        copyfile(self.std, f"{self.workspace}/paas.std")

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
        
        self.logger.debug("Zapping bad channels...")
        self.psrchive.zap_bad_channel(self.fs)
        self.logger.debug("Zapping bad channels... Done. ")

        self.logger.debug("Scrunching...")
        self.psrchive.scrunch(self.fs)
        self.logger.debug("Scrunching... Done. ")
    
    def get_toas(self):
        self.logger.debug("Getting TOAs...")
        self.psrchive.get_toas(self.fs, template=f"{self.workspace}/paas.std", output=f"{self.workspace}/pulsar.tim")
        self.logger.debug("Getting TOAs... Done. ")

    def time(self, fit_params="auto", potential_params=[]):
        if not self.initialized:
            raise Exception("Workspace not initialized")
        
        self.logger.debug("Initializing PINT handler... ")
        self.pint.initialize()

        if fit_params != "auto":
            self.logger.debug(f"Unfreezing parameters... {fit_params}", layer=1)
            self.pint.freeze_all()
            for p in fit_params:
                self.pint.unfreeze(p)

        if self.pint.check_toa_gaps():
            potential_params = [] # Not adding parameter after a huge gap
        
        if len(potential_params) > 0:
            # Run F-test
            f_test_res = {"params": [], "p_values": []}
            for i in range(len(potential_params)):
                for param_comb in itertools.combinations(potential_params, i + 1):
                    this_pass, this_p_value = self.pint.f_test(param_comb)
                    if this_pass:
                        f_test_res["params"].append(param_comb)
                        f_test_res["p_values"].append(this_p_value)
                    self.logger.debug(f"Testing parameter {param_comb}... Done. (passed = {this_pass}, p-value = {this_p_value}")
            
            # Find lowest p-value
            if(len(f_test_res["p_values"]) > 0):
                best_comb = f_test_res["params"][f_test_res["p_values"].index(min(f_test_res["p_values"]))]
                for this_param in best_comb:
                    self.pint.unfreeze(this_param)
                self.logger.debug(f"Best parameter to add: {best_comb} due to lowest p-value ")

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

        self.logger.debug("Plotting residuals... ")
        self.pint.plot()

        self.logger.debug("Saving model... ")
        self.pint.save()

        self.logger.debug("Done. ")

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