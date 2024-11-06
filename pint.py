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
import traceback

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
        self.f_status = None
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

    def error_filter(self, threshold=0.05):
        while True:
            # get error_ok
            threshold_phase = threshold * (1 / self.m.F0.value) * u.s
            if threshold_phase < 5000 * u.s:
                threshold_phase = 5000 * u.s
            error_ok = self.t.get_errors() < threshold_phase.to(u.us)

            # filter
            toas_bad = np.where(error_ok == False)[0]
            toas_good = np.where(error_ok == True)[0]

            # do not filter out more than 10% of points
            if len(toas_bad) / len(error_ok) < 0.10:
                break

            threshold += 0.05
            self.logger.warning(f"More than 10% of points were filtered out by the error filter. Lowering threshold to {threshold}")

        # get toas and mjds
        self.bad_toas = self.t[toas_bad]
        self.bad_resids = self.prefit_resids.time_resids[toas_bad]
        self.t = self.t[toas_good]

        return self.t

    
    def dropout_chi2r_filter(self, threshold=30):
        # if len(self.t) < 10:
        #     return self.t
        
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
                self.logger.warning(f"Dropout trial failed for TOA {i}. ", e)
                dropout_chi2rs.append(np.inf)

        # fit model without dropout
        self_tmp = copy.deepcopy(self)
        try:
            self_tmp.fit()
            # no need to filter if chi2r < 1
            if self_tmp.f.get_params_dict("all", "quantity")["CHI2R"].value < 1:
                return self.t
        except Exception as e:
            self.logger.warning(f"Trial fit in dropout filter failed. ", e)

        # If all chi2rs are the same value
        if np.all(np.array(dropout_chi2rs) == dropout_chi2rs[0]):
            self.logger.warning(f"Chi2r for all dropout trials are the same ({dropout_chi2rs[0]}). Not TOA will be removed. ")
            return self.t

        while True:
            # calculate threshold
            dropout_chi2rs = np.array(dropout_chi2rs)
            # ref_chi2r = self_tmp.f.get_params_dict("all", "quantity")["CHI2R"].value
            ref_chi2r = np.median(dropout_chi2rs)
            # threshold_chi2r = ref_chi2r - median_abs_deviation(np.abs(dropout_chi2rs - ref_chi2r)) * threshold
            threshold_chi2r = ref_chi2r - median_abs_deviation(np.abs(dropout_chi2rs)) * threshold

            # filter
            toas_bad = np.where(dropout_chi2rs < threshold_chi2r)[0]
            toas_good = np.where(dropout_chi2rs >= threshold_chi2r)[0]
        
            # do not include the lastest 3 TOAs in bad TOAs if there's a huge gap in the lastest 3 TOAs
            if self.check_toa_gaps(latest_n_days=3, threshold=30):
                self.logger.warning("Huge gap ( > 30 days) in the lastest 3 TOAs. Do not filter the lastest 3 TOAs out since the model might not be able to fit the gap. ")
                i_del = np.where(toas_bad >= len(self.t) - 3)[0]
                toas_good = np.append(toas_good, toas_bad[i_del])
                toas_bad = np.delete(toas_bad, i_del)

            # # do not include the lastest 1 TOAs in bad TOAs
            # i_del = np.where(toas_bad >= len(self.t) - 1)[0]
            # toas_good = np.append(toas_good, toas_bad[i_del])
            # toas_bad = np.delete(toas_bad, i_del)

            # sanity check: if there are too many points get filtered out
            if (len(toas_bad) / len(self.t)) < 0.25:
                break

            threshold += 1
            self.logger.warning(f"More than 25% of points were filtered out by the dropout filter. Lowering threshold to ref_chi2 - mad * {threshold}")
        
        # get toas resid, err, and mjds
        self.logger.debug(f"Bad TOAs: {toas_bad}")
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
        self_current.dropout_chi2r_filter()
        self_current.fit()

        # fit model with additional params
        self_additional = copy.deepcopy(self)
        for param in additional_params:
            self_additional.unfreeze(param)
        try:
            self_additional.dropout_chi2r_filter()
            self_additional.fit()
        except:
            self.logger.warning("F-test failed. ")
            return False, 1.0

        # get residuals and rsses
        rss_current = get_rss(self_current.f.resids.time_resids)
        rss_additional = get_rss(self_additional.f.resids.time_resids)

        # get number of unfreezed params
        n_current = len(self_current.m.free_params)
        n_additional = len(self_additional.m.free_params)

        # calculate df
        df_current = len(self_current.t) - n_current
        df_additional = len(self_additional.t) - n_additional

        # calculate f
        F = ((rss_current - rss_additional) / (df_current - df_additional)) / (rss_additional / df_additional)
        
        # calculate p-value
        p_value = 1 - f_stats.cdf(float(F), dfn=float(df_current - df_additional), dfd=float(df_additional))

        # p-value check
        self.logger.debug(f"Parameters: {additional_params}, F: {F}, p-value: {p_value}")
        if p_value > p_value_threshold: 
            return False, p_value
            
        # # sanity check
        # ## check if freq-dot from more complex model is negative
        # if self_additional.m.F1.value > 0:
        #     self.logger.warning("F-test passed, but freq-dot from more complex model is positive. ")
        #     return False, p_value
        # ## check if the ra and dec changes are too large
        # if np.abs(self_current.m.RAJ.value - self_additional.m.RAJ.value) > beamsize or np.abs(self_current.m.DECJ.value - self_additional.m.DECJ.value) > beamsize:
        #     self.logger.warning("F-test passed, but ra and dec changes are too large (> 1 beamsize). ")
        #     return False, p_value
        
        return True, p_value
    
    def trial_fit(self, additional_params):
        utils.print_info("Running trial fit, the following PINT output is not comming from real fit. ")

        # fit current model
        self_current = copy.deepcopy(self)
        self_current.dropout_chi2r_filter()
        self_current.fit()

        # fit model with additional params
        self_additional = copy.deepcopy(self)
        self_additional.dropout_chi2r_filter()
        for param in additional_params:
            self_additional.unfreeze(param)
        try:
            self_additional.fit()
        except:
            self.logger.warning("Trial fit failed. ")
            return False
        
        # sanity check: check if new chi2 is better
        # chi2r_current = self_current.f.get_params_dict("all", "quantity")["CHI2R"].value
        # chi2r_additional = self_current.f.get_params_dict("all", "quantity")["CHI2R"].value

        # if chi2r_current <= chi2r_additional:
        #     return False
        
        return True


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
    
    def check_toa_gaps(self, latest_n_days=2, threshold=15):
        mjds = self.t.get_mjds().value
        mjds.sort()
        mjds = mjds[-latest_n_days:]

        if np.max(np.diff(mjds)) > threshold:
            return True

        return False
        
    def fit(self, raise_exception=True):
        if not self.initialized:
            self.initialize()

        # Sanity check: if F1 > 0, then set it into 0
        # if self.m["F1"].value > 0:
        #     self.m["F1"].value = 0

        try:
            self.f = pint.fitter.Fitter.auto(self.t, self.m)
        except Exception as e:
            self.logger.warning("Failed to initialize fitter. ")
            self.logger.warning("Error", e)
            self.logger.warning("Parameters", self.get_unfreezed_params())
            self.logger.warning("TOAs", self.t)
            self.logger.warning("Model", self.m)
            raise Exception("Fitting failed. Please resolve this error manually. ", e)

        try:
            self.f.fit_toas()
            self.logger.info(self.f.get_summary())
            self.f_status = True
        except Exception as e:
            self.logger.warning("Fitting failed. ")
            self.logger.warning("Error", e)
            self.logger.warning("Parameters", self.get_unfreezed_params())
            self.logger.warning("Timing Skipped. If this warning persists, it must be resolved manually.")
            # self.f = {
            #     "fail": True, 
            #     "error": e
            # }
            self.f_status = False
            if raise_exception:
                raise Exception("Fitting failed. Please resolve this error manually. ", e)

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