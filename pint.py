import astropy.units as u
import matplotlib.pyplot as plt

import pint.fitter
from pint.models import get_model_and_toas
from pint.residuals import Residuals
import pint.logging

from scipy.stats import median_abs_deviation
import numpy as np
import shutil
import time

pint.logging.setup(level="INFO")


class pint_handler():
    def __init__(self, self_super, initialize=True):
        self.toas = f"{self_super.workspace}/pulsar.tim"
        self.model = self_super.par
        self.model_output = self_super.par_output
        self.logger = self_super.logger
        self.m, self.t = False, False
        self.f = False

        self.initialized = False
        if initialize:
            self.initialize()

    def initialize(self):
        self.m, self.t = get_model_and_toas(self.model, self.toas)
        self.mad_filter()
        self.initialized = True

    def mad_filter(self):
        resids = np.abs(np.array(Residuals(self.t, self.m).phase_resids))
        mad = median_abs_deviation(resids)
        self.t = self.t[((resids - np.median(resids)) / mad) < 7]

        return self.t

    def freeze(self, param):
        if not self.initialized:
            self.initialize()

        self.m[param].frozen = True

    def unfreeze(self, param):
        if not self.initialized:
            self.initialize()

        self.m[param].frozen = False

    def freeze_all(self):
        if not self.initialized:
            self.initialize()

        for param in self.m.params:
            self.m[param].frozen = True
        
    def fit(self):
        if not self.initialized:
            self.initialize()
        
        try:
            self.f = pint.fitter.Fitter.auto(self.t, self.m)
            self.f.fit_toas()
            # self.f.print_summary()
            self.logger(self.f.get_summary())
        except Exception as e:
            self.logger("Fitting failed. ", e)
            self.f = {
                "fail": True, 
                "error": e
            }

    def plot(self):
        if not self.initialized:
            self.initialize()

        rs = Residuals(self.t, self.m).time_resids
        xt = self.t.get_mjds()

        plt.figure()
        plt.errorbar(
            xt.value,
            rs.to(u.us).value,
            self.t.get_errors().to(u.us).value,
            fmt="x",
            label="Pre-fit",
            alpha=0.55, 
            c="k", 
            capsize=3
        )

        if(self.f != False):
            plt.errorbar(
                xt.value,
                self.f.resids.time_resids.to(u.us).value,
                self.t.get_errors().to(u.us).value,
                fmt="x",
                label="Post-fit", 
                alpha=0.75, 
                c="r", 
                capsize=3
            )

        plt.title(f"{self.m.PSR.value} Timing Residuals")
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\\rm \\mu s$)")
        plt.grid()
        plt.legend()

        if(self.model_output != False):
            plt.savefig(f"{self.model_output}.png")
    
    def save(self, fmt="tempo2"):
        if not self.initialized:
            self.initialize()
            
        if not self.f:
            raise Exception("TOAs are not fitted. ")
        
        if self.model_output != False:
            if(self.model == self.model_output):
                self.logger("Overwriting", self.model)
                shutil.copyfile(self.model, self.model + f".bak{int(time.time())}")
            if isinstance(self.f, dict):
                with open(self.model_output, "w") as f:
                    f.write(f"# Fitting failed. \n# Error: {self.f['error']}")
            else:
                self.f.model.write_parfile(self.model_output, format=fmt)
        
        if isinstance(self.f, dict):
            return f"# Fitting failed. \n# Error: {self.f['error']}"
        
        return self.f.model.as_parfile()