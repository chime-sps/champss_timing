# PINT
import pint.logging
from pint.models import get_model_and_toas
from pint.models.timing_model import Component
from pint.residuals import Residuals

# Fitters
from ..fitters.wls import WLSFitter
from ..fitters.mcmc import MCMCFitter

# Other packages
from multiprocessing import Pool
from scipy.stats import median_abs_deviation
from scipy.stats import f as f_stats
import numpy as np
import shutil
import time
import copy
import tqdm
import os
import traceback
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

# Other local packages
from ..utils.utils import utils
from ..utils.stats_utils import stats_utils

# Set logging level
pint.logging.setup(level="WARNING")

###################################################################
# PINT Handler                                                    #
###################################################################

class pint_handler():
    def __init__(self, self_super, initialize=True):
        self.toas = f"{self_super.workspace}/pulsar.tim"
        self.model = f"{self_super.workspace}/pulsar.par"
        self.model_output = self_super.par_output
        self.reset_params = self_super.reset_params
        self.logger = self_super.logger.copy()
        self.n_pools = self_super.n_pools

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

        # Check if required params are present
        if "F0" not in self.m.params or "F1" not in self.m.params:
            raise Exception("Spindown parameters F0 and F1 are required. ")
        if "RAJ" not in self.m.params or "DECJ" not in self.m.params:
            raise Exception("Position parameters RAJ and DECJ are required. ")

        # Run prefit
        self.prefit_resids = Residuals(self.t, self.m)
        self.bad_toas = self.t[0:0]
        self.bad_resids = self.prefit_resids.time_resids[self.bad_toas]

        # Set initialized
        self.initialized = True

    def filter(self, error=True, dropout=True):
        if len(self.t) < 5:
            self.logger.debug("Less than 5 TOAs. Skipping filtering. ")
            return 

        # MAD filter
        # if mad:
            # self.mad_filter()

        # Error filter
        if error:
            self.error_filter()

        if dropout:
            # if len(self.t) > 90 and self.m.CHI2R.value < 5:
                # Quantile filter
                # if quantile:
                #     self.quantile_filter()
            if len(self.t) > 90 and len(self.get_unfreezed_params()) >=4:
                self.mad_filter2()
            else:
                # Dropout filter
                self.dropout_chi2r_filter()

    # def mad_filter(self, threshold=7):
    #     # get mad
    #     resids = np.abs(np.array(self.prefit_resids))
    #     mad = median_abs_deviation(resids)

    #     # filter
    #     toas_bad = np.where((resids - np.median(resids)) / mad > threshold)[0]
    #     toas_good = np.where((resids - np.median(resids)) / mad <= threshold)[0]

    #     # get toas and mjds
    #     self.bad_toas += self.t[toas_bad]
    #     self.bad_resids = np.concatenate((self.bad_resids, resids[toas_bad]))
    #     self.t = self.t[toas_good]

    #     return self.t

    def mad_filter2(self, threshold=3): # mad is the robust estimate of std dev. thres of 3 corresponds to 99.7% confidence interval
        # get resids
        prefit_resids = Residuals(self.t, self.m)
        resids = np.array(prefit_resids.time_resids.to(u.s).value)
        resids_errs = self.t.table["error"].to(u.s).value

        # # get mad and median
        # mad = median_abs_deviation(resids)
        # median = np.median(resids)

        # # scale the mad to get the threshold
        # mad_threshold = threshold * mad

        # get median
        median = np.median(resids)

        # get threshold
        mad_threshold = stats_utils.mad_outlier_thresholds(resids, z_score=threshold, return_interval=False)
        
        # filter data
        toas_bad = np.where(np.abs(resids - median) >= mad_threshold)[0]
        toas_good = np.where(np.abs(resids - median) < mad_threshold)[0]

        # sanity check: do not filter out the lastest 3 TOAs
        for i in toas_bad:
            if i >= len(self.t) - 3:
                toas_good = np.append(toas_good, i)
                toas_bad = np.delete(toas_bad, np.where(toas_bad == i))

        # sanity check: do not filter out toas within 2 times median of toa error or 1% of the phase
        P0 = (1 / self.m.F0.value)
        resids_errs_med = np.median(resids_errs) * 2
        for i in toas_bad:
            if np.abs(resids[i]) / P0 < 0.015 or np.abs(resids[i]) < resids_errs_med:
                toas_good = np.append(toas_good, i)
                toas_bad = np.delete(toas_bad, np.where(toas_bad == i))

        # get toas and mjds
        self.logger.debug(f"Bad TOAs (mad): {toas_bad}")
        self.bad_toas += self.t[toas_bad]
        self.bad_resids = np.concatenate((self.bad_resids, prefit_resids.time_resids[toas_bad]))
        self.t = self.t[toas_good]

        return self.t

    def quantile_filter(self, threshold=0.95):
        # get resids
        prefit_resids = Residuals(self.t, self.m)
        resids = np.abs(np.array(prefit_resids.time_resids))

        # filter
        toas_bad = np.where(resids > np.quantile(resids, threshold))[0]
        toas_good = np.where(resids <= np.quantile(resids, threshold))[0]

        # not filter out the lastest 3 TOAs
        i_bad_toas_but_new = np.where(toas_bad >= len(self.t) - 3)[0]
        toas_good = np.append(toas_good, toas_bad[i_bad_toas_but_new])
        toas_bad = np.delete(toas_bad, i_bad_toas_but_new)

        # get toas and mjds
        self.logger.debug(f"Bad TOAs (quantile): {toas_bad}")
        self.bad_toas += self.t[toas_bad]
        self.bad_resids = np.concatenate((self.bad_resids, prefit_resids.time_resids[toas_bad]))
        self.t = self.t[toas_good]

        return self.t

    def error_filter(self, threshold=0.01):
        # Get prefit residuals
        prefit_resids = Residuals(self.t, self.m)

        # Cut errors
        while True:
            # get error_ok
            threshold_phase = threshold * (1 / self.m.F0.value) * u.s
            if threshold_phase < 15000 * u.us:
                threshold_phase = 15000 * u.us
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
        self.logger.debug(f"Bad TOAs (error): {toas_bad}")
        self.bad_toas += self.t[toas_bad]
        self.bad_resids = np.concatenate((self.bad_resids, prefit_resids.time_resids[toas_bad]))
        self.t = self.t[toas_good]

        return self.t

    
    def dropout_chi2r_filter(self, threshold=30):
        utils.print_info("Running dropout_chi2r_filter, the following PINT output is coming from dropout trials. ")

        # Get chi2rs from dropout trials
        with Pool(self.n_pools) as p:
            dropout_chi2rs = list(tqdm.tqdm(p.map(self._dropout_filter_get_chi2rs, range(len(self.t))), total=len(self.t), desc="Dropout trials"))

        # Fit model without dropout (i.e., postfit)
        self_tmp = copy.deepcopy(self)
        try:
            self_tmp.fit(fitter="ls", clustering_fitter=False)
            # no need to filter if chi2r < 1
            if self_tmp.f.get_params_dict("all", "quantity")["CHI2R"].value < 1:
                return self.t
        except Exception as e:
            self.logger.warning(f"Trial fit in dropout filter failed. ", e)

        # If all chi2rs are the same value
        if np.all(np.array(dropout_chi2rs) == dropout_chi2rs[0]):
            self.logger.warning(f"Chi2r for all dropout trials are the same ({dropout_chi2rs[0]}). Not TOA will be removed. ")
            return self.t

        # calculate threshold
        dropout_chi2rs = np.array(dropout_chi2rs)
        # ref_chi2r = self_tmp.f.get_params_dict("all", "quantity")["CHI2R"].value
        ref_chi2r = np.median(dropout_chi2rs)
        # threshold_chi2r = ref_chi2r - median_abs_deviation(np.abs(dropout_chi2rs - ref_chi2r)) * threshold
        threshold_chi2r = ref_chi2r - median_abs_deviation(np.abs(dropout_chi2rs)) * threshold

        # filter
        toas_bad = np.where(dropout_chi2rs < threshold_chi2r)[0]
        toas_good = np.where(dropout_chi2rs >= threshold_chi2r)[0]

        # Sanity check: do not filter out the lastest 3 TOAs
        if max(self.t.get_mjds().value) - min(self.t.get_mjds().value) < 60 or len(self.t.get_mjds()) < 30:
            # do not include the lastest 3 TOAs in bad TOAs only if there's a huge gap in the lastest 3 TOAs
            if self.check_toa_gaps(latest_n_days=3, threshold=30):
                self.logger.warning("Huge gap ( > 30 days) in the lastest 3 TOAs. Do not filter the lastest 3 TOAs out since the model might not be able to fit the gap. ")
                i_del = np.where(toas_bad >= len(self.t) - 3)[0]
                toas_good = np.append(toas_good, toas_bad[i_del])
                toas_bad = np.delete(toas_bad, i_del)
        else:
            # do not include the lastest 3 TOAs in bad TOAs
            i_del = np.where(toas_bad >= len(self.t) - 3)[0]
            toas_good = np.append(toas_good, toas_bad[i_del])
            toas_bad = np.delete(toas_bad, i_del)

        # # do not include the lastest 1 TOAs in bad TOAs
        # i_del = np.where(toas_bad >= len(self.t) - 1)[0]
        # toas_good = np.append(toas_good, toas_bad[i_del])
        # toas_bad = np.delete(toas_bad, i_del)

        # sanity check: if there are too many points get filtered out
        if (len(toas_bad) / len(self.t)) > 0.25:
            self.logger.warning(f"More than 25% of points were filtered out by the dropout filter. Only filter out points with top 25% dropout chi2r. ")
            threshold_chi2r = np.percentile(dropout_chi2rs, 25)
            toas_bad = np.where(dropout_chi2rs < threshold_chi2r)[0]
            toas_good = np.where(dropout_chi2rs >= threshold_chi2r)[0]
        
        # get toas resid, err, and mjds
        self.logger.debug(f"Bad TOAs (dropout): {toas_bad}")
        self.bad_toas += self.t[toas_bad]
        self.bad_resids = np.concatenate((self.bad_resids, Residuals(self.t, self.m).time_resids[toas_bad]))
        self.t = self.t[toas_good]

        return self.t

    def _dropout_filter_get_chi2rs(self, i):
        try:
            # copy self
            self_tmp = copy.deepcopy(self)

            # remove toa
            self_tmp.t = self_tmp.t[:i] + self_tmp.t[i+1:]

            # fit
            f_tmp = WLSFitter(self_tmp.t, self_tmp.m)
            f_tmp.fit_toas()

            return f_tmp.get_params_dict("all", "quantity")["CHI2R"].value
        except Exception as e:
            self.logger.warning(f"Dropout trial failed for TOA {i}. ", e)
            return 1e64
    
    def f_test(self, additional_params, p_value_threshold=0.05, beamsize=0.87376064): # chime beam size
        # Ref: [1] https://sites.duke.edu/bossbackup/files/2013/02/NonLinSummary.pdf
        #      [2] https://online.stat.psu.edu/stat501/lesson/6/6.2

        utils.print_info(f"Running f_test for {additional_params}, the following PINT output is coming from f_test trials. ")

        def get_rss(resids):
            '''
            Get the residual sum of squares (RSS) for the given residuals.
            Parameters
            ----------
            resid : array-like
                The residuals to calculate the RSS for.
            Returns
            -------
            float
                The RSS of the residuals.
            '''
            return np.sum([resid**2 for resid in resids])

        # fit current model
        self_current = copy.deepcopy(self)
        try:
            self_current.filter()
            self_current.fit(fitter="ls")
        except:
            self.logger.warning("Failed to fit for TOAs with current model. ")
            return False, 1.0

        # fit model with additional params
        self_additional = copy.deepcopy(self)
        for param in additional_params:
            self_additional.unfreeze(param)
        try:
            self_additional.dropout_chi2r_filter()
            self_additional.fit(fitter="ls")
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

        return self.m.free_params
    
    def check_toa_gaps(self, latest_n_days=2, threshold=15):
        # Get MJDs and sort them by time
        mjds = self.t.get_mjds().value
        mjds.sort()
        mjds = mjds[-latest_n_days:]

        # Check if the difference between the latest n MJDs is greater than the threshold
        if np.max(np.diff(mjds)) > threshold:
            return True

        return False
    
    def fit_mcmc_report(self, savefig, nwalkers=50, nsteps=1500):
        '''
        Fit the model to the TOAs using MCMC and report the results.

        Parameters
        ----------
        nwalkers : int
            The number of walkers for MCMC fitter. Default is 50.
        nsteps : int
            The number of steps for MCMC fitter. Default is 1000.
        '''

        # Check if the model and TOAs are initialized
        if not self.initialized:
            self.initialize()

        # Run fit
        self.logger.debug("Running MCMC fit... ", layer=1)
        f = MCMCFitter(self.t, self.m, nwalkers=nwalkers, nsteps=nsteps, n_pools=self.n_pools)
        f.fit_toas()

        # Generate report
        self.logger.debug("Generating MCMC report... ", layer=1)
        f.plot(savefig=savefig)
        self.logger.debug(f"MCMC report saved to {savefig}")
        
    def fit(self, raise_exception=True, fitter="ls", maxiter=100, nwalkers=50, nsteps=1500, clustering_fitter=True):
        '''
        Fit the model to the TOAs.

        Parameters
        ----------
        raise_exception : bool
            Whether to raise an exception if the fitting fails. Default is True.
        fitter : str
            The fitter to use ("ls", "mcmc", or "auto"). Default is "ls". "auto" will choose the fitter based on the number of free parameters (> 2 = MCMC, <= 2 = LS).
        maxiter : int
            The maximum number of iterations for LS fitter. Default is 100.
        nwalkers : int
            The number of walkers for MCMC fitter. Default is 250.
        nsteps : int
            The number of steps for MCMC fitter. Default is 2500.
        clustering_fitter : bool
            Whether to use the clustering fitter if the fitting fails. Default is True.
        '''

        # Check if the model and TOAs are initialized
        if not self.initialized:
            self.initialize()

        # Reset parameters to 0 if required
        if self.reset_params:
            self.logger.debug("Resetting unfreezed parameters to 0. ")

            for param in ["F1", "PX", "PMRA", "PMDEC"]:
                if param not in self.m:
                    raise Exception(f"Parameter {param} is not present in the model. This could be due to inccorect input parfile to the pipeline. Please make sure Spindown and Equatorial position components are all present in the parfile. ")

            if "F1" not in self.m.free_params:
                self.logger.debug("F1 -> 0", layer=1)
                self.m["F1"].value = 0

            if "PX" not in self.m.free_params:
                self.logger.debug("PX -> 0", layer=1)
                self.m["PX"].value = 0

            if "PMRA" not in self.m.free_params:
                self.logger.debug("PMRA -> 0", layer=1)
                self.m["PMRA"].value = 0

            if "PMDEC" not in self.m.free_params:
                self.logger.debug("PMDEC -> 0", layer=1)
                self.m["PMDEC"].value = 0

        # Check if there are enough TOAs to fit
        if len(self.t) <= 1:
            self.f_status = False
            self.f = {"error": "Not enough TOAs to fit. "}
            self.logger.warning("TOAs are less than 2. Skipping fitting. ")
            return

        # Automatically choose the fitter
        if fitter == "auto":
            if len(self.get_unfreezed_params()) > 2:
                fitter = "mcmc"
                self.logger.debug("Using MCMC fitter. ", layer=1)
            else:
                fitter = "ls"
                self.logger.debug("Using LS fitter. ", layer=1)

        # Run fit
        try:
            if fitter == "ls": # Least Squares fitting
                self.f = WLSFitter(self.t, self.m)
                self.f.fit_toas(maxiter=maxiter)
                self.f_status = True
            elif fitter == "mcmc": # MCMC fitting
                self.f = MCMCFitter(self.t, self.m, nwalkers=nwalkers, nsteps=nsteps, n_pools=self.n_pools)
                self.f.fit_toas()
                self.f_status = True
            else:
                raise Exception(f"Fitter {fitter} is not supported. Supported fitters: ls, mcmc. ")
        except Exception as e:
            self.f_status = False
            self.f = {"error": e}
            self.logger.warning("Fitting failed. ", e)
            
            if raise_exception: # Raise exception if required
                raise e
            else:
                self.logger.error(traceback.format_exc())

        # Run clustering fitter if required
        if clustering_fitter:
            # If fitting failed, or chi2r > 10, try clustering fitter
            if not self.f_status or self.f.get_params_dict("all", "quantity")["CHI2R"].value > 10:
                self.logger.warning("Fitting failed or chi2r > 10. Try clustering fitter. ")

                # Initialize clustering fitter
                cf_m, cf_f = self.clustering_fitter(self.m, self.t, debug=True)

                # If regular fitter was failed, the accept clustering fitter
                if not self.f_status:
                    self.m = cf_m # update model
                else: # else, check if clustering fitter is better (smaller chi2r)
                    if cf_f.get_params_dict("all", "quantity")["CHI2R"].value < self.f.get_params_dict("all", "quantity")["CHI2R"].value:
                        self.m = cf_m
                        self.f = cf_f
                        self.f_status = True
                        self.logger.success("Clustering fitter resolved the issue. ")
                    else:
                        self.logger.error("Clustering fitter is not better. ")

    def get_typical_observation_interval(self, mjds):
        mjds = sorted(mjds)
        
        # Get difference between each observation
        diffs = np.diff(mjds)

        # Get the median and mad of the differences
        median = np.median(diffs)
        mad = median_abs_deviation(diffs)

        return median, mad

    def clustering_fitter(self, m, t, clustering_threshold=12, debug=False):
        # Get mjds
        mjds = t.get_mjds().value
        mjds.sort()
        mjd_median, mjd_mad = self.get_typical_observation_interval(mjds)

        # Clustering
        clusters = [[0]]
        for i in range(len(mjds) - 1):
            if mjds[i+1] - mjds[i] > clustering_threshold * mjd_mad + mjd_median:
                clusters.append([])
            clusters[-1].append(i+1)

        # Sort by num
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

        toas_idxes = []
        this_model = m
        for cluster in clusters:
            toas_idxes += cluster
            toas = t[toas_idxes]

            # fit 
            try:
                fitter = WLSFitter(toas, this_model)
                fitter.fit_toas()
                this_model = fitter.model
            except Exception as e:
                self.logger.warning("Fitting failed in clustering fitter. ", e)
                self.logger.warning("Returning the last successful model. ")
                return m

            if debug:
                # get residuals
                resids = Residuals(toas, this_model).time_resids.to(u.us).value
                plt.plot(toas.get_mjds(), resids, "x")
                plt.show()
                # print(len(toas))
        
        return this_model, fitter

    def plot(self, savefig=None):
        if not self.initialized:
            self.initialize()

        # Get the savefig path
        if savefig is None:
            savefig = "realtime_diagnostics.pdf"

            # MCMC fitter give a different name
            if isinstance(self.f, MCMCFitter):
                savefig = "mcmc_report.pdf"

            # Check if the model output is set
            if(self.model_output != False):
                savefig = os.path.join(os.path.dirname(self.model_output), savefig)

        # Check if the fitter is MCMC
        if isinstance(self.f, MCMCFitter):
            return self.f.plot(savefig=savefig) # use MCMC plot function

        # Get residuals
        rs = Residuals(self.t, self.m).time_resids
        xt = self.t.get_mjds()

        # Initialize the figure
        plt.figure()

        # Plot pre-fit residuals
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

        # Plot post-fit residuals
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

        # Title, labels, etc. 
        plt.title(f"{self.m.PSR.value} Timing Residuals")
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\\rm \\mu s$)")
        plt.grid()
        plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(savefig, bbox_inches="tight", dpi=300)
        self.logger.debug(f"Realtime diagnostic saved to {savefig}")
    
    def save(self, fmt="tempo2"):
        if not self.initialized:
            self.initialize()
        
        # Check if the model is fitted
        if not self.f:
            raise Exception("TOAs are not fitted. ")
        
        # Check if the output path is valid
        if self.model_output != False:
            # Check if overwritting
            if(self.model == self.model_output):
                self.logger.warning("Overwriting", self.model)
                shutil.copyfile(self.model, self.model + f".bak{int(time.time())}")

            # Handle the case where the fitter failed
            if isinstance(self.f, dict):
                with open(self.model_output, "w") as f:
                    f.write(f"# Fitting failed. \n# Error: {self.f['error']}")
            else:
                self.f.model.write_parfile(self.model_output, format=fmt)
        
        # Handle the case where the fitter failed
        if isinstance(self.f, dict):
            return f"# Fitting failed. \n# Error: {self.f['error']}"
        
        return self.f.model.as_parfile()