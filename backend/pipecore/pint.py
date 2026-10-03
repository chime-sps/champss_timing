# PINT
import pint.logging
from pint.models import get_model_and_toas
from pint.models.timing_model import Component
from pint.residuals import Residuals

# Fitters
from ..fitters.wls import WLSFitter
from ..fitters.downhillwls import LenientDownhillWLSFitter
from ..fitters.mcmc import MCMCFitter
from ..fitters.clustering import ClusteringFitter

# Other packages
from multiprocessing import Pool
from scipy.stats import f as f_stats
from scipy.stats import median_abs_deviation, norm
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

        self.m, self.t = None, None
        self.f = None
        self.f_remarks = []
        self.prefit_resids = None
        self.bad_toas = []
        self.bad_resids_prefit = {"vals": [], "errs": []}
        self.bad_resids_postfit = {"vals": [], "errs": []}

        self.initialized = False
        if initialize:
            self.initialize()

    def initialize(self):
        # Initialize model and toas
        self.m, self.t = get_model_and_toas(self.model, self.toas)

        # Check if required params are present
        if "F0" not in self.m.params:
            raise Exception("Spindown parameter F0 is required. ")
        if "RAJ" not in self.m.params or "DECJ" not in self.m.params:
            raise Exception("Position parameters RAJ and DECJ are required. ")

        # Add F1 if not present
        if "F1" not in self.m.params:
            import io
            from pint.models import get_model
            self.logger.debug("Spindown parameter F1 not present. Adding F1 = 0 to the model. ")
            parfile_text = self.m.as_parfile()
            parfile_text += "\nF1 0 1\n"
            self.m= get_model(io.StringIO(parfile_text))

        # Run prefit
        self.prefit_resids = Residuals(self.t, self.m)
        self.bad_toas = self.t[0:0]
        self.bad_resids_prefit["vals"] = self.prefit_resids.time_resids[self.bad_toas]
        self.bad_resids_prefit["errs"] = self.prefit_resids.get_data_error(scaled=True)[self.bad_toas]
        self.bad_resids_postfit["vals"] = self.prefit_resids.time_resids[self.bad_toas]
        self.bad_resids_postfit["errs"] = self.prefit_resids.get_data_error(scaled=True)[self.bad_toas]

        # Set initialized
        self.initialized = True

    def filter(self):
        # Sanity check for the number of TOAs
        if len(self.t) < 5:
            self.logger.debug("Less than 5 TOAs. Skipping filtering. ")
            return 
        
        # Get prefit residuals and errors
        f0 = self.m.F0.value
        prefit_resids = Residuals(self.t, self.m)
        resids_vals = np.array(prefit_resids.time_resids.to(u.s).value)
        resids_errs = np.array(prefit_resids.get_data_error(scaled=True).to(u.s).value)
        mjds = np.array(self.t.get_mjds().value)

        # Generate initial TOA mask
        mask = np.ones(len(self.t), dtype=bool)

        # EM filter
        mask = self.em_filter(mask, resids_vals, f0)
        
        # Error filter
        mask = self.error_filter(mask, resids_vals, resids_errs)

        # MAD filter
        mask = self.mad_filter2(mask, resids_vals, resids_errs)

        # Sanity check: do not filter out TOAs with larger error but still close enough in terms of residuals:
        #     residuals relative to the spin period (1.5 std) AND errors (1.5 sigma error)
        mask |= (np.abs(resids_vals) < 3 * np.std(resids_vals[mask])) & (np.abs(resids_vals) < 1.5 * resids_errs)
        
        # Sanity check: do not filter out the latest TOAs
        mjds_sorted = np.sort(mjds)
        latest_toa_threshold = np.mean([mjds_sorted[-3], mjds_sorted[-4]])
        mask = mask | (mjds > latest_toa_threshold)
        self.logger.debug(f"Will not filter out TOAs later than MJD {latest_toa_threshold}")

        # Apply mask
        self.bad_toas += self.t[~mask]
        self.bad_resids_prefit["vals"] = np.concatenate((self.bad_resids_prefit["vals"], prefit_resids.time_resids[~mask]))
        self.bad_resids_prefit["errs"] = np.concatenate((self.bad_resids_prefit["errs"], prefit_resids.get_data_error(scaled=True)[~mask]))
        self.t = self.t[mask]

        return

    def em_filter(self, mask, vals, f0, threshold=0.2, n_iter=256):
        '''
        Filtering TOAs using the Expectation-Maximization algorithm. 
        Args:
            mask (np.ndarray): Current mask of TOAs.
            vals (np.ndarray): Residual values of TOAs.
            errs (np.ndarray): Errors of TOAs.
            threshold (float): Probability threshold for considering a TOA as inlier.
            n_iter (int): Number of EM iterations.

        Returns:
            np.ndarray: Updated mask after EM filtering.
        '''

        # Initial guess for uniform distribution
        a, b = -0.5 / f0, 0.5 / f0
        if b == a:
            self.logger.warning("Uniform distribution has zero width. Stopping EM without masking.")
            return mask

        # Initial guess for Gaussian distribution
        mu, sigma = 0, vals[mask].std() / 2

        # Initial guess for the fraction of toas that are gaussian
        f_gau = 0.5

        for _ in range(n_iter):
            # [E-step]
            # Get the probability of, given residual r, it comes from gaussian
            # Using Bayes' theorem: 
            # P(gaussian|r) = P(r|gaussian) * P(gaussian) / P(r), where
            # - P(r|gaussian): given the assumed gaussian distribution [mu, sigma], what is the likelihood of getting r
            # - P(gaussian): the probability that r is from gaussian (i.e., "fraction_gaussian")
            # - P(r): the overall probability of getting r (from both distributions), 
            #         which is P(r|gaussian)*P(gaussian) + P(r|uniform)*(1-P(gaussian)). 

            # P(r|gaussian)
            p_r_gau = norm.pdf(vals[mask], mu, sigma)

            # P(r|uniform)
            p_r_uni = 1 / (b - a)

            # P(gaussian|r)
            p_gau_r = (f_gau * p_r_gau) / (p_r_gau * f_gau + p_r_uni * (1 - f_gau))

            # [M-step] 
            # Given P(gaussian|r): 
            # - P(gaussian): the mean of P(gaussian|r) given many r's
            # - mu and sigma: the weighted mean and standard deviation of vals using P(gaussian|r) as weights

            # P(gaussian)
            f_gau = np.mean(p_gau_r)
            if f_gau >= 1:
                self.logger.debug("Fraction of Gaussian components reached 1. Stopping EM without masking.")
                return mask

            # mu and sigma
            mu = np.sum(p_gau_r * vals[mask]) / np.sum(p_gau_r)
            sigma = np.sqrt(
                np.sum(p_gau_r * (vals[mask] - mu)**2) / np.sum(p_gau_r)
            )

            # sanity check for mu and sigma
            if sigma <= 0:
                self.logger.warning("Estimated Gaussian sigma is non-positive. Stopping EM without masking.")
                return mask
            if mu <= a or mu >= b:
                self.logger.warning("Estimated Gaussian mu is out of bounds. Stopping EM without masking.")
                return mask
            
        # Test mixture implies too few outliers
        if np.sum(mask) * (1 - f_gau) < 1: # less than 1% outliers
            self.logger.debug("Mixture model not significantly better than pure Gaussian. Stopping EM without masking.")
            return mask

        # Get this mask
        this_mask = p_gau_r > threshold

        # Create a new overall mask
        new_mask = mask & this_mask

        self.logger.debug(f"Outlier TOAs (em): {np.where(mask & ~new_mask)[0]}")

        return new_mask
    
    def error_filter(self, mask, vals, errs, z_score=3, max_iters=3):
        '''
        Filtering TOAs based on their errors using the MAD.

        Args:
            mask (np.ndarray): Current mask of TOAs.
            vals (np.ndarray): Residual values of TOAs.
            errs (np.ndarray): Errors of TOAs.
            z_score (float): Z-score threshold for outlier detection.
            max_iters (int): Maximum number of iterations for the filtering process.

        Returns:
            np.ndarray: Updated mask after error filtering.
        '''
        
        # Sanity check on number of remaining TOAs
        if mask.sum() < 5:
            self.logger.debug("Less than 5 TOAs remaining. Skipping error filter.")
            return mask
        
        # Normalize error by subtracting the median
        errs_log = np.log10(errs) # Take the logarithm to make the distribution gaussian-ish
        errs_log = errs_log - np.median(errs_log[mask])

        # Get threshold
        mad_threshold = stats_utils.mad_outlier_thresholds(errs_log[mask], z_score=z_score, return_interval=False)
        if not np.isfinite(mad_threshold) or mad_threshold <= 0:
            self.logger.debug("Invalid MAD threshold. Skipping error filter.")
            return mask

        # Get this mask
        this_mask = (errs_log < mad_threshold)

        # Create a new overall mask
        new_mask = mask & this_mask

        self.logger.debug(f"Outlier TOAs (error): {np.where(mask & ~new_mask)[0]}")
        if max_iters <= 1 or np.array_equal(new_mask, mask):
            return new_mask

        return self.error_filter(new_mask, vals, errs, z_score=z_score, max_iters=max_iters-1)

    def mad_filter2(self, mask, vals, errs, threshold=3, max_iters=3): # mad is the robust estimate of std dev. thres of 3 corresponds to 99.7% confidence interval
        '''
        Filtering TOAs based on their residuals using the MAD.

        Args:
            mask (np.ndarray): Current mask of TOAs.
            vals (np.ndarray): Residual values of TOAs.
            errs (np.ndarray): Errors of TOAs.
            threshold (float): Z-score threshold for outlier detection.
            max_iters (int): Maximum number of iterations for the filtering process.

        Returns:
            np.ndarray: Updated mask after MAD filtering.
        '''

        # Sanity check on number of remaining TOAs
        if mask.sum() < 5:
            self.logger.debug("Less than 5 TOAs remaining. Skipping MAD filter.")
            return mask

        # Get threshold
        mad_threshold = stats_utils.mad_outlier_thresholds(vals[mask], z_score=threshold, return_interval=False)
        if not np.isfinite(mad_threshold) or mad_threshold <= 0:
            self.logger.debug("Invalid MAD threshold. Skipping MAD filter.")
            return mask

        # Get this mask
        this_mask = (np.abs(vals) < mad_threshold)

        # Create a new overall mask
        new_mask = mask & this_mask

        self.logger.debug(f"Outlier TOAs (mad): {np.where(mask & ~new_mask)[0]}")
        if max_iters <= 1 or np.array_equal(new_mask, mask):
            return new_mask

        return self.mad_filter2(new_mask, vals, errs, threshold=threshold, max_iters=max_iters-1)
    
    def f_test(self, additional_params, p_value_threshold=0.05, beamsize=0.87376064): # chime beam size
        # Ref: [1] https://sites.duke.edu/bossbackup/files/2013/02/NonLinSummary.pdf
        #      [2] https://online.stat.psu.edu/stat501/lesson/6/6.2

        self.logger.info(f"Running f_test for {additional_params}, the following PINT output is coming from f_test trials. ")

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

        # make a copy of self
        self_current = copy.deepcopy(self)
        self_current.logger = self.logger.copy()

        # fit current model
        try:
            self_current.filter()
            self_current.fit(fitter="ls", clustering_fitter=False)
        except:
            self.logger.warning("Failed to fit for TOAs with current model. ")
            return False, 1.0

        # fit model with additional params
        self_additional = copy.deepcopy(self)
        for param in additional_params:
            self_additional.unfreeze(param)
        try:
            self_additional.filter()
            self_additional.fit(fitter="ls", clustering_fitter=False)
        except:
            self.logger.warning("F-test failed. ")
            return False, 1.0

        # Sanity check for postfit parameters
        raj_current = self_current.f.model.RAJ.quantity.to(u.deg).value
        raj_additional = self_additional.f.model.RAJ.quantity.to(u.deg).value
        raj_diff = np.min([
            (raj_current - raj_additional) % 360,
            (raj_additional - raj_current) % 360
        ])
        decj_current = self_current.f.model.DECJ.quantity.to(u.deg).value
        decj_additional = self_additional.f.model.DECJ.quantity.to(u.deg).value
        decj_diff = np.min([
            (decj_current - decj_additional),
            (decj_additional - decj_current)
        ])
        if raj_diff > beamsize * 2 or decj_diff > beamsize * 1.5:
            self.logger.warning("F-test failed. Postfit RAJ change is much larger than beam size (i.e., not physical). ")
            return False, 1.0

        # Get residuals
        current_resids = self_current.f.resids.time_resids
        additional_resids = self_additional.f.resids.time_resids

        # get rsses
        rss_current = get_rss(current_resids)
        rss_additional = get_rss(additional_resids)

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
    
    def check_toa_gaps(self, latest_n_days=2, threshold=15):
        # Get MJDs and sort them by time
        mjds = np.sort(self.t.get_mjds().value)
        mjds = mjds[-latest_n_days:]

        # Sanity check if there's more than 1 TOA in the latest n days
        if len(mjds) < 2:
            return False

        # Check if the difference between the latest n MJDs is greater than the threshold
        if np.max(np.diff(mjds)) > threshold:
            return True

        return False

    # def compute_phoff(self, m, t, freeze_phoff=True):
    #     # Make sure the PHOFF component present
    #     if not hasattr(m, 'PHOFF'):
    #         m.add_component(
    #             Component.component_types["PhaseOffset"]()
    #         )

    #     # Freeze all parameters except PHOFF
    #     freezed_params = []
    #     for param in m.params:
    #         if param != 'PHOFF' and not m[param].frozen:
    #             m[param].frozen = True
    #             freezed_params.append(param)
        
    #     # Unfreeze PHOFF
    #     m['PHOFF'].frozen = False

    #     # Fit the model
    #     f = LenientDownhillWLSFitter(t, m)
    #     f.fit_toas()
    #     self.logger.debug(f"Computed PHOFF: {f.model['PHOFF'].value}")

    #     # Restore the frozen parameters
    #     for param in freezed_params:
    #         f.model[param].frozen = False

    #     # Freeze PHOFF again if required
    #     if freeze_phoff:
    #         f.model['PHOFF'].frozen = True

    #     return f.model
    
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
        
    def fit(self, raise_exception=True, fitter="ls", maxiter=10, nwalkers=50, nsteps=1500, clustering_fitter=True):
        '''
        Fit the model to the TOAs.

        Parameters
        ----------
        raise_exception : bool
            Whether to raise an exception if the fitting fails. Default is True.
        fitter : str
            The fitter to use ("ls", "mcmc", or "auto"). Default is "ls". "auto" will choose the fitter based on the number of free parameters (> 2 = MCMC, <= 2 = LS).
        maxiter : int
            The maximum number of iterations for LS fitter. Default is 10.
        nwalkers : int
            The number of walkers for MCMC fitter. Default is 50.
        nsteps : int
            The number of steps for MCMC fitter. Default is 1500.
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
        if len(self.t) <= 1 and maxiter > 1:
            raise Exception("Not enough TOAs to fit (need at least 2). ")

        # Automatically choose the fitter
        if fitter == "auto":
            if len(self.get_unfreezed_params()) > 2:
                fitter = "mcmc"
                self.logger.debug("Using MCMC fitter. ", layer=1)
            else:
                fitter = "ls"
                self.logger.debug("Using LS fitter. ", layer=1)
        

        # Initialize a copy of model and toas
        this_m = copy.deepcopy(self.m)
        this_t = copy.deepcopy(self.t)

        # # Compute PHOFF
        # this_m = self.compute_phoff(this_m, this_t, freeze_phoff=False)

        # Remove PHOFF
        if hasattr(this_m, "PHOFF"):
            this_m.remove_component("PhaseOffset")

        # Compute pulse number
        this_t.compute_pulse_numbers(this_m) # compute pulse number to help fitters converge better

        # Initialize fitter
        f_prefit = None
        if fitter == "ls": # Least Squares fitting
            f_prefit = LenientDownhillWLSFitter(this_t, this_m, track_mode="use_pulse_numbers")
            self.f = copy.deepcopy(f_prefit)
        elif fitter == "mcmc": # MCMC fitting
            f_prefit = MCMCFitter(this_t, this_m, nwalkers=nwalkers, nsteps=nsteps, n_pools=self.n_pools)
            self.f = copy.deepcopy(f_prefit)
        else:
            raise Exception(f"Fitter {fitter} is not supported. Supported fitters: ls, mcmc. ")

        # Run fit
        try:
            self.f.fit_toas(maxiter=maxiter)
        except Exception as e:
            self.logger.warning("Fitting failed, restoring prefit model. Error:", e)

            self.f_remarks.append("FITTING_FAILED")
            self.f = f_prefit # Return the prefit model if fitting fails
            
            if raise_exception: # Raise exception if required
                raise e
            else:
                self.logger.error(traceback.format_exc())

        # Run clustering fitter when there's a large gap in the TOAs recently
        if clustering_fitter and len(this_t) >= 3: # Clustering is not meaningful for n_toas < 3
            # Initialize clustering fitter
            cf = ClusteringFitter(this_t, this_m)

            # If there are clusters
            if cf.clustered_toas.has_valid_clusters():
                # Get size and the number of days since the last gap
                gap_size = cf.clustered_toas.get_gap_size()
                c0_baseline, days_since_gap = cf.clustered_toas.get_cluster_baselines()

                # Determine whether this is right after a large gap
                if gap_size > np.min([60, np.max([c0_baseline * 0.1, 10])]) and days_since_gap < 30: 
                    self.logger.debug(f"Right after a large gap: gap_size={gap_size}, days_since_gap={days_since_gap}")

                    # Get chi2r values for each cluster and combined
                    chi2r_c0, _, chi2r_c0c1 = cf.clustered_toas.get_chi2rs(this_m)

                    # Determine whether the combined chi2r is significantly worse than the first cluster alone
                    if chi2r_c0c1 > chi2r_c0 * 3 and chi2r_c0c1 > 1.5: # If the combined chi2r is significantly worse than the first cluster alone
                        self.logger.debug(f"Combined chi2r ({chi2r_c0c1}) is significantly worse than the first cluster alone ({chi2r_c0}).")
                        self.logger.info("Using clustering fitter to resolve phase wraps due to the gap.")

                        # Fit using the clustering fitter
                        best_fitter_state = cf.fit_toas()

                        # Check if the clustering fitter state is better than the LS fitting results
                        if best_fitter_state.model.CHI2R.value < self.f.model.CHI2R.value:
                            self.logger.success(f"Clustering fitter improved the fit: CHI2R {self.f.model.CHI2R.value} -> {best_fitter_state.model.CHI2R.value}")
                            self.f = best_fitter_state
                            self.f_remarks.append("CLUSTERING_FITTER_IMPROVED")

                            # Remove FITTING_FAILED status
                            if "FITTING_FAILED" in self.f_remarks:
                                self.f_remarks.remove("FITTING_FAILED")
                        else:
                            self.logger.debug(f"Clustering fitter did not improve the fit (CHI2R {self.f.model.CHI2R.value} -> {best_fitter_state.model.CHI2R.value})")
                    else:
                        self.logger.debug(f"Combined chi2r ({chi2r_c0c1}) is not significantly worse than the first cluster alone ({chi2r_c0}). Will not use clustering fitter.")
                else:
                    self.logger.debug(f"Not right after a large gap (gap_size={gap_size}, days_since_gap={days_since_gap}). Will not use clustering fitter.")
            else:
                self.logger.debug("No cluster is present. Will not test for large gaps.")

        # Some final checks and calculations
        if hasattr(self, "f"):
            # Check if chi2r is reliable
            if np.isnan(self.f.model.CHI2R.value) or np.isinf(self.f.model.CHI2R.value):
                self.f.model.CHI2R.value = 0.0
                self.f_remarks.append("CHI2R_UNRELIABLE")
                    
            # Calculate residuals for bad toas
            if len(self.bad_toas) > 0:
                bad_residuals = Residuals(self.bad_toas, self.f.model)
                self.bad_resids_postfit["vals"] = bad_residuals.time_resids
                self.bad_resids_postfit["errs"] = bad_residuals.get_data_error(scaled=True)

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

        # Calculate prefit residuals
        prefit_resids = Residuals(self.t, self.m)
        prefit_resids_vals = prefit_resids.time_resids
        prefit_resids_errs = prefit_resids.get_data_error(scaled=True)
        prefit_mjds = self.t.get_mjds()

        # Calculate post-fit residuals if the fitter exists
        if self.f:
            postfit_resids = self.f.resids.time_resids
            postfit_resids_errs = self.f.resids.get_data_error(scaled=True)
            postfit_mjds = self.t.get_mjds()

        # Initialize the figure
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        # Plot pre-fit residuals
        ax[0].errorbar(
            prefit_mjds,
            prefit_resids_vals.to(u.us).value,
            prefit_resids_errs.to(u.us).value,
            fmt="x",
            c="k", 
            capsize=3
        )
        ax[0].errorbar(
            self.bad_toas.get_mjds(),
            self.bad_resids_prefit["vals"].to(u.us).value,
            self.bad_resids_prefit["errs"].to(u.us).value,
            fmt="x",
            label="Bad TOAs",
            c="r",
            capsize=3
        )
        ax[0].legend()
        ax[0].set_ylabel("Residual ($\\rm \\mu s$)")
        ax[0].set_title("Pre-fit Residuals")
        ax[0].grid(True)

        # Plot post-fit residuals
        if(self.f != False):
            ax[1].errorbar(
                postfit_mjds,
                postfit_resids.to(u.us).value,
                postfit_resids_errs.to(u.us).value,
                fmt="x", 
                c="k", 
                capsize=3
            )
            ax[1].errorbar(
                self.bad_toas.get_mjds(),
                self.bad_resids_postfit["vals"].to(u.us).value,
                self.bad_resids_postfit["errs"].to(u.us).value,
                fmt="x",
                label="Bad TOAs",
                c="r",
                capsize=3
            )
            ax[1].legend()
            ax[1].set_ylabel("Residual ($\\rm \\mu s$)")
            ax[1].set_xlabel("MJD")
            ax[1].set_title("Post-fit Residuals")
            ax[1].grid(True)

            # Set the same scale as the pre-fit residuals
            ylim_lower = min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]])
            ylim_upper = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
            ax[0].set_ylim(ylim_lower, ylim_upper)
            ax[1].set_ylim(ylim_lower, ylim_upper)

        # Save the figure
        plt.tight_layout()
        plt.savefig(savefig, bbox_inches="tight", dpi=300)
        self.logger.debug(f"Realtime diagnostic saved to {savefig}")
    
    def save(self, fmt="tempo2", write_tim=False):
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
        