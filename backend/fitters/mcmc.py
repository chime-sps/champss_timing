###################################################################
# PINT MCMC Fitter                                                #
###################################################################

import pint.fitter
from pint.residuals import Residuals
from pint.sampler import EmceeSampler
from pint.mcmc_fitter import MCMCFitter

import time
import os
import copy
import corner
import emcee
import datetime
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
from multiprocessing import Pool
from loguru import logger as log

from ..utils.utils import utils

class EmceeSampler(EmceeSampler):
    def __init__(self, nwalkers, n_pools):
        # Set the number of pools
        self.n_pools = n_pools

        # Initialize the sampler from PINT
        super().__init__(nwalkers=nwalkers)

    def initialize_sampler(self, lnpostfn, ndim):
        """Initialize the internal sampler data.

        This is usually done after __init__ because ndim and lnpostfn are properties
        of the Fitter that holds this sampler.

        """
        self.ndim = ndim
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnpostfn, pool=Pool(self.n_pools))

class MCMCFitter(MCMCFitter):
    def __init__(self, toas, model, nwalkers=250, nsteps=2500, random_seed=0, n_pools=1):
        log.trace(f"Using MCMCFitter (CHAMPSS ver.)")
        
        # Initialize parameters
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.fitted = False

        # Copy a initial model
        self.initial_model = copy.deepcopy(model)

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize PINT MCMC fitter
        super().__init__(
            toas=toas, 
            model=model, 
            sampler=EmceeSampler(nwalkers=nwalkers, n_pools=n_pools), 
            lnprior=self.lnprior_basic, 
            lnlike=self.lnlikelihood_chi2
        )

        # Set initial state
        self.sampler.random_state = np.random.mtrand.RandomState()

    def fit_toas(self):
        self.fitted = True
        return super().fit_toas(maxiter=self.nsteps)

    def lnprior_basic(self, ftr, theta):
        lnsum = 0.0

        for val, key in zip(theta[:-1], ftr.fitkeys[:-1]):
            lnsum += getattr(ftr.model, key).prior_pdf(val, logpdf=True)

        # Sanity check for RA
        initial_ra = self.initial_model.RAJ.quantity.deg
        current_ra = ftr.model.RAJ.quantity.deg
        if np.abs(initial_ra - current_ra) > 0.5:
            return -np.inf

        # Sanity check for DEC
        initial_dec = self.initial_model.DECJ.quantity.deg
        current_dec = ftr.model.DECJ.quantity.deg
        if np.abs(initial_dec - current_dec) > 0.5:
            return -np.inf

        # Sanity check for P0
        initial_p0 = (1 / self.initial_model.F0.value) * u.s
        current_p0 = (1 / ftr.model.F0.value) * u.s
        rotation_per_day = (u.sday / initial_p0).decompose()
        aliasing_p0_lower = (u.sday / (rotation_per_day + 1)).to(u.s)
        aliasing_p0_upper = (u.sday / (rotation_per_day - 1)).to(u.s)
        if not (aliasing_p0_lower < current_p0 < aliasing_p0_upper):
            print(
                "P0 prior: %.3f, current P0: %.3f"
                % (initial_p0, current_p0)
            )
            return -np.inf

        # Sanity check for F1
        current_f1 = ftr.model.F1.quantity
        if current_f1.value > 0: 
            return -(current_f1.value * 1e32)**2
            
        return lnsum


    def lnlikelihood_chi2(self, ftr, theta):
        ftr.set_parameters(theta)
        return -Residuals(toas=ftr.toas, model=ftr.model).chi2

    def plot(self, savefig=None, burnin=500):
        """
        Plot the corner plot of the MCMC results.
        """

        # Initialize the figure
        fig = plt.figure(figsize=(10, 20))
        subfigs = fig.subfigures(3, 1, height_ratios=[3, 2, 1], wspace=0, hspace=0)

        # Plot corner plot
        corner.corner(
            self.sampler.get_chain()[burnin:, :, :].reshape((-1, self.n_fit_params)),
            labels=list(self.get_fitparams().keys()),
            fig=subfigs[0]
        )

        # Plot MCMC chains
        ax_chain = subfigs[1].subplots(len(self.get_fitparams()), 1, sharex=True)
        for i, param in enumerate(self.get_fitparams()):
            ax_chain[i].plot(
                self.sampler.get_chain()[burnin:, :, i], 
                alpha=0.3, 
                color="k",
                lw=0.5
            )
            ax_chain[i].set_ylabel(param)
        ax_chain[0].set_title("MCMC Chains")
        ax_chain[-1].set_xlabel("MCMC step")

        # Plot residuals
        ax_residuals = subfigs[2].subplots(1, 1)
        ax_residuals.errorbar(
            self.toas.get_mjds().to(u.us).value,
            self.resids.time_resids.to(u.us).value,
            self.toas.get_errors().to(u.us).value,
            fmt="x",
            label="Post-fit", 
            alpha=0.75, 
            c="k", 
            capsize=3
        )
        ax_residuals.axhline(
            0, color="k", linestyle="--", lw=0.5, alpha=0.5
        )
        ax_residuals.set_xlabel("MJD")
        ax_residuals.set_ylabel("Residuals (us)")
        ax_residuals.set_title("Residuals")

        # Plot MCMC results
        text = "Post-MCMC values (50th percentile +/- (16th/84th percentile):\n"
        text += "\n"
        for key, param in self.get_mcmc_results(upper=16, lower=84, burnin=burnin).items():
            text += f"%8s:" % key + "%25.15g (+ %12.5g  / - %12.5g)" % (param["value"], param["upper"], param["lower"]) + "\n"
        text += "\n"
        text += f"Burn-in: {burnin} samples | Final ln-posterior: {self.get_mcmc_final_lnposterior():.6f}\n"
        ax_residuals.text(
            1, 6.75, text, fontsize=10, ha="right", va="top", transform=ax_residuals.transAxes
        )

        # Show version text
        fig.text(0.001, 0, f"CHAMPSS Timing Pipeline MCMCFitter ({utils.get_version_hash()}) | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")
        
        # Save the figure
        if savefig is not None:
            fig.savefig(savefig, bbox_inches="tight", dpi=300)
            print(f"Saved figure to {savefig}")
        else:
            plt.show()

    def get_mcmc_final_lnposterior(self):
        """
        Get the final log posterior value from the MCMC fitter.

        Returns
        -------
        float
            The final log posterior value.
        """

        # Check if the MCMC fitter has been fitted
        if not self.fitted:
            raise ValueError("MCMC fitter has not been fitted yet.")
        
        # Get the final log posterior value
        return self.lnposterior(self.maxpost_fitvals)

    def get_mcmc_results(self, upper=16, lower=84, burnin=500):
        """
        Get the model from the MCMC fitter.

        Parameters
        ----------
        upper : int
            The upper percentile for the model parameters (default is 16=16%).
        lower : int
            The lower percentile for the model parameters (default is 84=84%).

        Returns
        -------
        dict
            The model parameters.
        """

        # Check if the MCMC fitter has been fitted
        if not self.fitted:
            raise ValueError("MCMC fitter has not been fitted yet.")
        
        # Get the samples from the MCMC chain
        samples = np.transpose(self.sampler.sampler.get_chain()[burnin:, :, :], (1, 0, 2)).reshape(
            (-1, self.n_fit_params)
        )

        # Get ranges for each parameter
        ranges2 = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [upper, 50, lower], axis=0)),
        )

        # Create a dictionary to store the model parameters
        model_params = {}
        for name, vals in zip(self.fitkeys, ranges2):
            model_params[name] = {
                "value": vals[0],
                "upper": vals[1],
                "lower": vals[2]
            }

        return model_params
