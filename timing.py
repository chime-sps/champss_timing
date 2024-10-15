from .psrchive import psrchive_handler
from .pint import pint_handler
from .exec import exec
from .utils import utils

import os
import time
import glob
from shutil import copyfile, rmtree

class timing():
    def __init__(self, ars, std, par, par_output=False, n_pools=4, workspace_cleanup=True, logger=print):
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
        self.logger(f"Creating workspace at {self.workspace}")
        self.logger(f"Press Ctrl+C to cancel")
        time.sleep(1)
        os.makedirs(self.workspace, exist_ok=True)

        # Move files to workspace
        self.logger(" Copying files to workspace... ")
        for i, f in enumerate(self.ars):
            self.logger(f"  Copying file {i+1}/{len(self.ars)}", end="\r")
            # os.system(f"cp {f} {self.workspace}")
            copyfile(f, f"{self.workspace}/{os.path.basename(f)}")
            self.fs.append(f"{self.workspace}/{os.path.basename(f)}")
        # os.system(f"cp {self.par} {self.workspace}/pulsar.par")
        copyfile(self.par, f"{self.workspace}/pulsar.par")
        # os.system(f"cp {self.std} {self.workspace}/paas.std")
        copyfile(self.std, f"{self.workspace}/paas.std")

        self.logger("Initialized workspace")
        self.initialized = True

    def cleanup(self, verbose=False):
        if not self.initialized:
            raise Exception("Workspace not initialized")
        
        # Remove workspace
        self.logger(f"Removing workspace at {self.workspace}")
        self.logger(f"Press Ctrl+C to cancel")
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
        
        self.logger("Zapping bad channels...")
        self.psrchive.zap_bad_channel(self.fs)
        self.logger("Zapping bad channels... Done. ")

        self.logger("Scrunching...")
        self.psrchive.scrunch(self.fs)
        self.logger("Scrunching... Done. ")
    
    def get_toas(self):
        self.logger("Getting TOAs...")
        self.psrchive.get_toas(self.fs, template=f"{self.workspace}/paas.std", output=f"{self.workspace}/pulsar.tim")
        self.logger("Getting TOAs... Done. ")

    def time(self, fit_params="auto"):
        if not self.initialized:
            raise Exception("Workspace not initialized")
        
        self.logger("Initializing PINT handler... ")
        self.pint.initialize()

        if(fit_params != "auto"):
            self.logger(f"Unfreezing parameters... {fit_params}")
            self.pint.freeze_all()
            for p in fit_params:
                self.pint.unfreeze(p)

        self.logger("Fitting TOAs... ")
        self.pint.fit()

        self.logger("Plotting residuals... ")
        self.pint.plot()

        self.logger("Saving model... ")
        self.pint.save()

        self.logger("Done. ")

    # def update_model(self): # optional. 
    #     if not self.initialized:
    #         raise Exception("Workspace not initialized")
        
    #     # Remove TZRSITE to fix a problem with psrchive for CHIME observations
    #     open(f"{self.workspace}/pulsar.par.pam", "w").write(
    #         open(f"{self.workspace}/pulsar.par").read().replace("TZRSITE", "# TZRSITE")
    #     )
        
    #     self.logger("Updating timing model for scrunched archives... ")
    #     self.psrchive.update_model(self.fs, f"{self.workspace}/pulsar.par.pam")
    #     self.logger("Updating timing model for scrunched archives... Done. ")

    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            utils.print_error(f"Error while running timing. Workspace will NOT be removed at {self.workspace}")
            raise exc_type(exc_value).with_traceback(traceback)
        else:
            if self.workspace_cleanup:
                self.cleanup()
        return False