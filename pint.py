import astropy.units as u
import matplotlib.pyplot as plt

import pint.fitter
from pint.models import get_model_and_toas
from pint.residuals import Residuals
import pint.logging

from scipy.stats import median_abs_deviation
from scipy.stats import f as f_stats
import numpy as np
import shutil
import time
import copy

from .utils import utils

pint.logging.setup(level="ERROR")

class pint_handler():
    def __init__(self, self_super, initialize=True):
        self.toas = f"{self_super.workspace}/pulsar.tim"
        self.model = self_super.par
        self.model_output = self_super.par_output
        self.logger = self_super.logger.copy()
        self.m, self.t = False, False
        self.f = False
        self.prefit_resids = False
        self.bad_toas = []
        self.bad_resids = []

        self.initialized = False
        if initialize:
            self.initialize()

    def initialize(self):
        # Initialize model and toas
        self.m, self.t = get_model_and_toas(self.model, self.toas)

        # Run prefit
        self.prefit_resids = Residuals(self.t, self.m)

        # Set initialized
        self.initialized = True
    

    def filter(self, error=True, dropout=True):
        # MAD filter
        # if mad:
            # self.mad_filter()

        # Error filter
        if error:
            self.error_filter()

        # Dropout filter
        if dropout:
            self.dropout_chi2r_filter()

    def mad_filter(self, threshold=7):
        # get mad
        resids = np.abs(np.array(self.prefit_resids))
        mad = median_abs_deviation(resids)

        # filter
        toas_bad = np.where((resids - np.median(resids)) / mad > threshold)[0]
        toas_good = np.where((resids - np.median(resids)) / mad <= threshold)[0]

        # get toas and mjds
        self.bad_toas = self.t[toas_bad]
        self.bad_resids = resids[toas_bad]
        self.t = self.t[toas_good]

        return self.t

    def error_filter(self, threshold=0.5):
        # get error_ok
        threshold_phase = threshold * (1 / self.m.F0.value) * u.s
        error_ok = self.t.get_errors() < threshold_phase.to(u.us)

        # filter
        toas_bad = np.where(error_ok == False)[0]
        toas_good = np.where(error_ok == True)[0]

        # get toas and mjds
        self.bad_toas = self.t[toas_bad]
        self.bad_resids = self.prefit_resids.time_resids[toas_bad]
        self.t = self.t[toas_good]

        return self.t

    
    def dropout_chi2r_filter(self, threshold=7):
        utils.print_info("Running dropout_chi2r_filter, the following PINT output is coming from dropout trials. ")
        
        dropout_chi2rs = []
        for i in range(len(self.t)):
            self.logger.debug("Dropout filter trial", i, "/", len(self.t), layer=1, end="\r")
            try:
                # copy self
                self_tmp = copy.deepcopy(self)
                
                # remove toa
                self_tmp.t = self.t[:i] + self.t[i+1:]

                # fit
                f_tmp = pint.fitter.Fitter.auto(self_tmp.t, self_tmp.m)
                f_tmp.fit_toas()
            
                # append chi2r
                dropout_chi2rs.append(f_tmp.get_params_dict("all", "quantity")["CHI2R"].value)
            
            except Exception as e:
                self.logger.error(f"Dropout trial failed for TOA {i}. ", e)
                dropout_chi2rs.append(np.inf)

        # fit model without dropout
        self_tmp = copy.deepcopy(self)
        self_tmp.fit()

        # calculate threshold
        dropout_chi2rs = np.array(dropout_chi2rs)
        # ref_chi2r = self_tmp.f.get_params_dict("all", "quantity")["CHI2R"].value
        ref_chi2r = np.median(dropout_chi2rs)
        threshold_chi2r = ref_chi2r - median_abs_deviation(np.abs(dropout_chi2rs - ref_chi2r)) * threshold

        # filter
        toas_bad = np.where(dropout_chi2rs < threshold_chi2r)[0]
        toas_good = np.where(dropout_chi2rs >= threshold_chi2r)[0]
        
        # get toas resid, err, and mjds
        self.bad_toas = self.t[toas_bad]
        self.bad_resids = Residuals(self.t, self.m).time_resids[toas_bad]
        self.t = self.t[toas_good]

        return self.t
    
    def f_test(self, additional_params, p_value_threshold=0.05, beamsize=0.87376064): # chime beam size
        # Ref: [1] https://sites.duke.edu/bossbackup/files/2013/02/NonLinSummary.pdf
        #      [2] https://online.stat.psu.edu/stat501/lesson/6/6.2

        utils.print_info("Running f_test, the following PINT output is coming from f_test trials. ")

        def get_rss(resids):
            return np.sum([resid**2 for resid in resids])

        # fit current model
        self_current = copy.deepcopy(self)
        self_current.fit()

        # fit model with additional params
        self_additional = copy.deepcopy(self)
        for param in additional_params:
            self_additional.unfreeze(param)
        try:
            self_additional.fit()
        except:
            self.logger.warning("F-test failed. ")
            return False

        # # get residuals and rsses
        # rss_current = get_rss(self_current.f.resids.time_resids)
        # rss_additional = get_rss(self_additional.f.resids.time_resids)

        # # get number of unfreezed params
        # n_current = len(self_current.m.free_params)
        # n_additional = len(self_additional.m.free_params)

        # # calculate df
        # df_current = len(self_current.t) - n_current
        # df_additional = len(self_additional.t) - n_additional

        # # calculate f
        # F = ((rss_current - rss_additional) / (df_current - df_additional)) / (rss_additional / df_additional)
        
        # # calculate p-value
        # p_value = 1 - f_stats.cdf(float(F), dfn=float(df_current - df_additional), dfd=float(df_additional))

        def f_test(resid_simple_mode, resid_complex_mode, n_free_params_simple, n_free_params_complex, n_points):
            def get_rss(resid):
                return np.sum(resid**2)
            
            rss_simple = get_rss(resid_simple_mode)
            rss_complex = get_rss(resid_complex_mode)

            f_stat = ((rss_simple - rss_complex) / (n_free_params_complex - n_free_params_simple)) / (rss_complex / (n_points - n_free_params_complex))
            p_value = 1 - f_stats.cdf(float(f_stat), float(n_free_params_complex - n_free_params_simple), float(n_points - n_free_params_complex))

            return f_stat, p_value
        
        F, p_value = f_test(
            self_current.f.resids.time_resids, 
            self_additional.f.resids.time_resids, 
            len(self_current.m.free_params), 
            len(self_additional.m.free_params), 
            len(self_additional.t)
        )

        # p-value check
        self.logger.debug(f"Parameters: {additional_params}, F: {F}, p-value: {p_value}")
        if p_value > p_value_threshold: 
            return False, p_value
            
        # sanity check
        ## check if freq-dot from more complex model is negative
        # if self_additional.m.F1.value > 0:
        #     self.logger.warning("F-test passed, but freq-dot from more complex model is positive. ")
        #     return False, p_value
        ## check if the ra and dec changes are too large
        if np.abs(self_current.m.RAJ.value - self_additional.m.RAJ.value) > beamsize or np.abs(self_current.m.DECJ.value - self_additional.m.DECJ.value) > beamsize:
            self.logger.warning("F-test passed, but ra and dec changes are too large (> 1 beamsize). ")
            return False, p_value
        
        return True, p_value

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
    
    def get_unfreezed_params(self):
        if not self.initialized:
            self.initialize()

        params = []
        for param in self.m.params:
            if not self.m[param].frozen:
                params.append(param)

        return params

        
    def fit(self):
        if not self.initialized:
            self.initialize()

        # Sanity check: if F1 > 0, then set it into 0
        if self.m["F1"].value > 0:
            self.m["F1"].value = 0
        
        try:
            self.f = pint.fitter.Fitter.auto(self.t, self.m)
            self.f.fit_toas()
            # self.f.print_summary()
            self.logger.info(self.f.get_summary())
        except Exception as e:
            self.logger.error("Fitting failed. ", e)
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
                self.logger.warning("Overwriting", self.model)
                shutil.copyfile(self.model, self.model + f".bak{int(time.time())}")
            if isinstance(self.f, dict):
                with open(self.model_output, "w") as f:
                    f.write(f"# Fitting failed. \n# Error: {self.f['error']}")
            else:
                self.f.model.write_parfile(self.model_output, format=fmt)
        
        if isinstance(self.f, dict):
            return f"# Fitting failed. \n# Error: {self.f['error']}"
        
        return self.f.model.as_parfile()