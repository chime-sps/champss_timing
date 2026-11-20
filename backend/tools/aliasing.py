import os
import json
import glob
import psrchive
import pickle
import copy
import tqdm
import numpy as np
import astropy.units as u
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import datetime
import traceback
from pint import models
from pint.fitter import WLSFitter
from pint.models import get_model, get_model_and_toas
from scipy.ndimage import gaussian_filter
from pint.residuals import Residuals
from io import StringIO

from ..utils.utils import utils
# from ..utils.exec import exec
from ..utils.logger import logger
from ..utils.stats_utils import stats_utils
from ..utils.data_quality import data_quality_utils
from ..tools.template_utils import StackTemplate
from ..tools.shift_finder import ShiftFinder
# from ..pipecore.psrchive import psrchive_handler
from ..datastores.database import database
# from ..datastores.archive_cache import archive_cache

def get_aliasing_finder_from_stackfile(stackfile, n_subs=None, **kwargs):
    from ..tools.stacked_data import StackedData
    
    # read stackfile
    stacked_data = StackedData.load(stackfile)

    # scrunch polarisation, frequencies
    stacked_data.scrunch(axis=1)
    stacked_data.scrunch(axis=2)

    # downsample
    if n_subs is not None:
        stacked_data.downsample_to_length(n_subs, axis=0)

    return AliasingFinder(
        data=stacked_data.get_data(normalize=True)[:, 0, :, :], 
        epochs=stacked_data.epochs, 
        metadata=stacked_data.meta,
        **kwargs
    )

def get_smooth_sigma(template):
    width = data_quality_utils.pulse_width(template)
    return width * 0.2

def generate_template_from_stacked_data(data, smooth_sigma=None, logger=logger()):
    # Sanity check on the size of the data
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("Data must be a 2D array with at least one bin per profile")
    
    # Generate a profile template
    logger.debug("Optimizing stacked profile to create a template...")
    stpl = StackTemplate(data, size=data.shape[1], shift_meth="fourier", logger=logger.copy())
    stpl.optimize()
    template = stpl.get_template()

    # Smooth the std profile
    if smooth_sigma is None:
        smooth_sigma = get_smooth_sigma(template)
    logger.debug(f"Smoothing template with sigma={smooth_sigma}")
    template = gaussian_filter(template, sigma=smooth_sigma, mode="wrap")

    return template

class DealiasedTimingSolution:
    def __init__(self, model, toas, best_af, logger):
        self.aliased = model
        self.prefit = None
        self.postfit = None
        self.fitter = None
        self.status = None
        
        self.toas = toas
        self.best_af = best_af
        self.logger = logger

        self.info = {}
        self.info["alias_factor"] = self.best_af

        self.mask = np.ones(len(toas), dtype=bool)

        if self.best_af != 0:
            dealias_res = self._dealias()

            if dealias_res:
                self.status = "ok"
                self.logger.success("Dealiased successfully")
            else:
                self.status = "error"
                self.logger.error("Dealiasing failed")
        else:
            dealias_res = True
            self.status = "not_needed"
            self.logger.success("No dealias needed, alias_factor=0")
            
        self.info["status"] = self.status
        self.info["timestamp"] = datetime.datetime.now().isoformat()

    def _dealias(self):
        # get p0
        p0 = 1 / self.aliased.F0.value

        # calculate alias solution
        dp_alias = p0 / ((u.sday.to(u.s) * (1/self.best_af)) / p0)
        p_dealiased = p0 + dp_alias
        self.logger.info(f"Dealiased period: {p_dealiased} s, dp_alias: {dp_alias} s")
        self.info["p0_aliased"] = p0
        self.info["p0_unaliased"] = p_dealiased

        # calculate outlier threshold and toa mask
        rs_aliased = Residuals(self.toas, self.aliased).phase_resids
        mad_threshold = stats_utils.mad_outlier_thresholds(rs_aliased, z_score=3, return_interval=False)
        self.mask[np.abs(rs_aliased - np.median(rs_aliased)) > mad_threshold] = False

        # create unaliased prefit model, unfreeze F0
        self.prefit = copy.deepcopy(self.aliased)
        self.prefit.F0.quantity = (1 / p_dealiased)
        self.prefit.F0.frozen = False
        
        # remove PHOFF component if exists
        if hasattr(self.prefit, "PHOFF"):
            self.logger.debug("Removing PHOFF component from model")
            self.prefit.remove_component("PhaseOffset")

        # fit model
        try:
            self.fitter = WLSFitter(self.toas[self.mask], self.prefit)
            self.fitter.fit_toas()
            self.fitter.print_summary()
            self.info["chi2r_unaliased"] = self.fitter.model.CHI2R.value
        except Exception as e:
            self.logger.error("Dealiasing failed, fitting failed. ")
            self.logger.error(traceback.format_exc())
            return False

        # check difference between prefit and postfit
        p0_postfit = 1 / self.fitter.model.F0.value
        if np.abs(p0_postfit - p_dealiased) > np.abs(dp_alias):
            self.logger.error("Dealiasing failed, postfit period is too far from dealiased period", "p0_postfit", p0_postfit, "p_dealiased", p_dealiased, "dp_alias", dp_alias)
            return False
        
        # replace the prefit with the postfit model
        self.postfit = self.fitter.model
        self.logger.success(f"Model dealiased successfully")

        # calculate chi2r for aliased model after masking out outliers
        chi2r_aliased = Residuals(self.toas[self.mask], self.aliased).chi2_reduced
        self.info["chi2r_aliased"] = chi2r_aliased

        # check chi2r
        chi2r_postfit = self.postfit.CHI2R.value
        if chi2r_postfit > chi2r_aliased * 1.1:
            self.logger.error("Dealiasing failed, postfit chi2r is much higher than prefit chi2r. ")
            return False
        elif chi2r_postfit > chi2r_aliased:
            self.logger.warning("Postfit chi2r is higher than prefit chi2r. This alias factor may not be reliable.")

        return True

    def get_info(self):
        return self.info

    def print_info(self):
        self.logger.info("Dealiased timing solution information:")
        for key, value in self.info.items():
            self.logger.info(f"{key}: {value}", layer=1)

    def get_dealiased_parfile(self):
        if self.postfit is not None:
            return self.postfit.as_parfile()
        return None

    def write_dealiased_parfile(self, filename):
        if self.postfit is not None:
            with open(filename, 'w') as f:
                f.write(self.postfit.as_parfile())
        else:
            self.logger.error("No postfit model available to write.")

    def plot(self, savefig=None):
        # Get mjds
        y = self.toas[self.mask].get_mjds().value

        # Get residuals
        r_aliased = Residuals(self.toas[self.mask], self.aliased)
        r_prefit = Residuals(self.toas[self.mask], self.prefit) if self.prefit is not None else None
        r_postfit = Residuals(self.toas[self.mask], self.postfit) if self.postfit is not None else None

        # Get y errors
        yerr_aliased = r_aliased.get_data_error().to("s").value * self.aliased.F0.value
        yerr_prefit = r_prefit.get_data_error().to("s").value * self.prefit.F0.value if r_prefit is not None else None
        yerr_postfit = r_postfit.get_data_error().to("s").value * self.postfit.F0.value if r_postfit is not None else None

        _, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Plot aliased
        ax[0].errorbar(
            y, 
            r_aliased.phase_resids, 
            yerr=yerr_aliased, 
            fmt='x', 
            label='prefit (p_aliased)', 
            color='blue', 
            zorder=0, 
            alpha=0.25, 
            capsize=3
        )
        ax[0].set_title(f'Prefit - aliasing by {self.best_af} phase(s) per day')
        ax[0].set_ylabel('Residual (phase)')

        # Plot prefit
        if r_prefit is not None:
            ax[1].errorbar(
                y, 
                r_prefit.phase_resids, 
                yerr=yerr_prefit, 
                fmt='x', 
                label='prefit (p_unaliased)', 
                color='green', 
                capsize=3
            )
        else:
            ax[1].text(
                0.5, 0.5, 'No prefit residuals available. Is the timing solution aliased?',
                ha='center', va='center', transform=ax[1].transAxes
            )
        ax[1].set_title('Prefit - unaliased')
        ax[1].set_ylabel('Residual (phase)')

        # Plot postfit
        if r_postfit is not None:
            ax[2].errorbar(
                y, 
                r_postfit.phase_resids, 
                yerr=yerr_postfit, 
                fmt='x', 
                label='postfit (p_unaliased)', 
                color='black', 
                capsize=3
            )
        else:
            ax[2].text(
                0.5, 0.5, 'No postfit residuals available. Is the timing solution aliased?',
                ha='center', va='center', transform=ax[2].transAxes
            )
        ax[2].set_title('Postfit - unaliased')
        ax[2].set_xlabel('MJD')

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
            self.logger.success(f"Dealiasing plot saved to {savefig}")

        plt.show()

class AliasingFinderResultsByShifts:
    def __init__(
            self, 
            shifts_expected,  
            shifts_measured_vals, 
            shifts_measured_errs, 
            mask, 
            af_trials, 
            rms, 
            data,
            template, 
            stacked_snr, 
            avg_snr_per_subints, 
            psr_id, 
            n_day_stacked, 
            logger
        ):
        self.shifts_expected = shifts_expected
        self.shifts_measured_vals = shifts_measured_vals
        self.shifts_measured_errs = shifts_measured_errs
        self.mask = mask
        self.af_trials = af_trials
        self.rms = rms
        self.stacked_snr = stacked_snr
        self.data = data
        self.template = template
        self.avg_snr_per_subints = avg_snr_per_subints
        self.psr_id = psr_id
        self.n_day_stacked = n_day_stacked
        self.logger = logger

        # Generate subint indices
        self.subint_idxs = np.arange(self.data.shape[0])

        # Get af_max and af_min
        self.af_max = np.max(self.af_trials)
        self.af_min = np.min(self.af_trials)

        # Find the trial that minimizes the rms
        self.best_i = np.where(self.rms == np.min(self.rms))[0][0]
        self.best_af = self.af_trials[self.best_i]

        # Get expected shifts
        self.best_expected_shifts = self.shifts_expected[self.best_i]

        # Get shifts and nearby the expected shifts
        self.nearby_expected_shifts = []
        if self.best_af > self.af_min:
            self.nearby_expected_shifts.append(self.shifts_expected[self.best_i - 1])
        if self.best_af < self.af_max:
            self.nearby_expected_shifts.append(self.shifts_expected[self.best_i + 1])

    def __normalize_power(self, power):
        power = power - np.mean(power)
        power = power / np.max(power)

        return power

    def dealias(self, par_or_model, tim_or_toas):
        """
        Dealias the given model and toas (or par and tim files) using the best aliasing factor found.

        Parameters
        ----------
        par_or_model : str or pint.models.timing_model.TimingModel
            Either the path to the par file or an already loaded TimingModel instance.
        tim_or_toas : str or pint.toa.TOAs
            Either the path to the tim file or an already loaded TOAs instance.

        Returns
        -------
        DealiasedTimingSolution
            The dealiased timing solution using the best aliasing factor.
        """

        # Determine if input is file paths or already loaded model and toas
        if isinstance(par_or_model, str) and isinstance(tim_or_toas, str):
            model, toas = get_model_and_toas(par_or_model, tim_or_toas)
        else:
            model = par_or_model
            toas = tim_or_toas

        return DealiasedTimingSolution(
            model=model,
            toas=toas,
            best_af=self.best_af,
            logger=self.logger
        )

    def get_summary(self):
        return {
            "psr_id": str(self.psr_id),
            "n_stacked": int(self.n_day_stacked),
            "alias_factor": int(self.best_af),
            "snr_stacked": float(self.stacked_snr),
            "avg_snr_per_subints": float(self.avg_snr_per_subints),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def print_summary(self):
        self.logger.info("Aliasing Finder Summary:")
        for key, value in self.get_summary().items():
            self.logger.info(f"{key}: {value}", layer=1)

    def plot(self, savefig=None):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8),  gridspec_kw={'width_ratios': [3, 1]})

        # plot rms
        axs[0, 0].plot(self.af_trials, self.rms, c="k", lw=1)
        axs[0, 0].axvline(self.best_af, label=f"Best fit: aliasing by {self.best_af} phase(s) per day", c="r", lw=1)
        axs[0, 0].set_title(f"RMS vs Alias Factor")
        axs[0, 0].set_xlabel("Alias Factor")
        axs[0, 0].set_ylabel("RMS")
        axs[0, 0].set_yscale("log")
        axs[0, 0].legend(frameon=False, fontsize=8)

        # plot shifts
        axs[1, 0].errorbar(
            self.subint_idxs[self.mask], 
            self.shifts_measured_vals[self.mask], 
            self.shifts_measured_errs[self.mask], 
            c="k", 
            lw=1, 
            label="Measured TOAs", 
            capsize=2, 
            fmt="x"
        )
        axs[1, 0].errorbar(
            self.subint_idxs, 
            self.best_expected_shifts, 
            c="r", 
            lw=1, 
            linestyle="-", 
            label=f"Best fit: aliasing by {self.best_af} phase(s) per day"
        )
        for i, this_nearby_expected_shifts in enumerate(self.nearby_expected_shifts):
            axs[1, 0].errorbar(
                self.subint_idxs, 
                this_nearby_expected_shifts,  
                c="k", 
                lw=1, 
                linestyle="--", 
                label=f"Best fit +/- 1 phase per day" if i == 0 else None, 
                alpha=0.5
            )
        axs[1, 0].set_title("Shifts")
        axs[1, 0].set_xlabel("Subints")
        axs[1, 0].set_ylabel("Shift (phase)")
        axs[1, 0].legend(frameon=False, fontsize=8)

        # plot powers
        axs[2, 0].plot(self.__normalize_power(self.template), c="k", lw=1, label="Std Profile")
        axs[2, 0].plot(self.__normalize_power(np.sum(self.data, axis=0)), c="k", lw=1, label=f"Sum of subints 0-{int(len(self.data))-1}", alpha=0.75, linestyle=":")
        axs[2, 0].plot(self.__normalize_power(np.sum(self.data[:int(len(self.data)/2)], axis=0)), c="r", lw=1, label=f"Sum of subints 0-{int((len(self.data))/2)-1}", alpha=0.75)
        axs[2, 0].plot(self.__normalize_power(np.sum(self.data[int(len(self.data)/2):], axis=0)), c="b", lw=1, label=f"Sum of subints {int((len(self.data))/2)}-{len(self.data)-1}", alpha=0.75)
        axs[2, 0].set_title(f"Powers")
        axs[2, 0].set_xlabel("Samples")
        axs[2, 0].set_ylabel("Power")
        axs[2, 0].legend(frameon=False, fontsize=8)

        # # normalize stacked data
        # for i, _ in enumerate(self.data):
        #     self.data[i] = np.array(self.data[i]) - min(self.data[i])
        #     self.data[i] = np.array(self.data[i]) / max(self.data[i])

        # find the peack of the stacked profiles
        peak_index = np.argmax(np.sum(self.data, axis=0))

        # calculate the shifts in phase
        shifts_phase = []
        for shift in self.best_expected_shifts:
            this_shift = peak_index + (shift / 1) * len(self.data[0])
            if this_shift > len(self.data[0]):
                this_shift -= len(self.data[0])
            elif this_shift < 0:
                this_shift += len(self.data[0])
            shifts_phase.append(this_shift)

        # plot stacked profile
        axs[0, 1].plot(np.sum(self.data, axis=0), lw=0.5, c="k")
        axs[0, 1].set_title("Stacked Profile")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # plot stacked data
        axs[1, 1].remove()
        axs[2, 1].remove()
        n_params_gs = axs[0, 1].get_gridspec()
        axs_subints = fig.add_subplot(n_params_gs[1:3, 1])
        axs_subints.matshow(self.data, cmap="gray_r", aspect="auto", rasterized=True)
        axs_subints.plot(shifts_phase, self.subint_idxs, color="r", lw=1, ls=":", label="Best fit")
        axs_subints.plot([peak_index - 0.015 * len(self.data[0])] * len(self.subint_idxs), self.subint_idxs, "r-", label="No aliasing", lw=0.5, alpha=0.75)
        axs_subints.set_title("Subint Profiles")
        axs_subints.legend(frameon=False, fontsize=8)
        axs_subints.set_xlabel("Phase")
        axs_subints.set_ylabel("Sample")
        axs_subints.set_xticks(np.arange(0, len(self.data[0]) + 1, len(self.data[0]) / 2))
        axs_subints.set_xticklabels(["0", "0.5", "1"])
        axs_subints2 = axs_subints.twiny()
        axs_subints2.set_xlim(axs_subints.get_xlim())
        
        # plot info
        fig.text(0.001, 0.000, f"CHAMPSS Timing Pipeline alias_utils ({utils.get_version_hash()}) | {self.psr_id} | {self.n_day_stacked} observations stacked | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")

        fig.tight_layout()

        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()

class AliasingFinderResultsBySNR:
    def __init__(
            self, 
            af_vals, 
            snrs, 
            logger
        ):
        self.af_vals = af_vals
        self.snrs = snrs
        self.logger = logger

        # Find the best shift based on SNR
        max_snr_idx = np.argmax(self.snrs)
        self.best_af = self.af_vals[max_snr_idx]
        self.best_snr = self.snrs[max_snr_idx]

    def dealias(self, par_or_model, tim_or_toas):
        """
        Dealias the given model and toas (or par and tim files) using the best aliasing factor found.

        Parameters
        ----------
        par_or_model : str or pint.models.timing_model.TimingModel
            Either the path to the par file or an already loaded TimingModel instance.
        tim_or_toas : str or pint.toa.TOAs
            Either the path to the tim file or an already loaded TOAs instance.

        Returns
        -------
        DealiasedTimingSolution
            The dealiased timing solution using the best aliasing factor.
        """

        # Determine if input is file paths or already loaded model and toas
        if isinstance(par_or_model, str) and isinstance(tim_or_toas, str):
            model, toas = get_model_and_toas(par_or_model, tim_or_toas)
        else:
            model = par_or_model
            toas = tim_or_toas

        return DealiasedTimingSolution(
            model=model,
            toas=toas,
            best_af=self.best_af,
            logger=self.logger
        )

    def get_summary(self):
        summary = {
            "best_af": self.best_af,
            "best_snr": self.best_snr,
            "n_trials": len(self.af_vals)
        }
        return summary

    def print_summary(self):
        summary = self.get_summary()
        self.logger.info("Aliasing Finder Results Summary:")
        self.logger.info(f"Best AF: {summary['best_af']}")
        self.logger.info(f"Best SNR: {summary['best_snr']}")
        self.logger.info(f"Number of Trials: {summary['n_trials']}")

    def plot(self, savefig=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.af_vals, self.snrs, 'kx')
        plt.axvline(self.best_af, color='r', linestyle='--', label=f"Best SNR at {self.best_af} phase(s) per day")
        plt.xlabel("Shift")
        plt.ylabel("SNR")
        plt.title("Aliasing Finder Results by SNRs")
        plt.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()

class AliasingFinder():
    def __init__(self, data, epochs, metadata={}, ncpus=1, logger=logger()):
        """
        Initialize the alias_utils class.

        Parameters:
        data (np.ndarray): Pulsar data array with shape (n_subs, n_freqs, n_bins).
        epochs (np.ndarray): Array of subint epochs in days.
        ncpus (int): Number of CPUs to use.
        logger (logger): Logger instance.
        metadata (dict): Additional metadata (from stack file) for the observation.
        """

        self.data = data
        self.epochs = epochs
        self.ncpus = ncpus
        self.logger = logger
        self.metadata = metadata

        # Get sidereal day constant from astropy
        self.sidereal_day = (1 * u.sday).to(u.day).value

        # Sanity check on the data dimension
        if self.data.ndim != 3:
            raise ValueError("Data array must have three dimensions (n_subs, n_freqs, n_bins).")

        # Srunch frequency: (n_subs, n_freqs, n_bins) -> (n_subs, n_bins)
        self.data = np.nanmean(self.data, axis=1)  # Average over frequency channels

        # Get size of the data
        self.n_subints, self.n_bins = self.data.shape

    @staticmethod
    def _find_shift(payload):

        if len(payload) == 3:
            template, power, n_steps = payload
        else:
            template, power = payload
            n_steps = 2000  # default value

        # Initialize shift finder
        shift_finder = ShiftFinder(template, power)

        # Find shift
        shift_finder.compute(n_steps=n_steps, burn_in=int(n_steps/2), progress=False)

        # Get shift result
        measured_shift = shift_finder.result.shift

        # Measure subint SNR
        subint_snr = data_quality_utils.boxcar_snr(power)

        return {
            "shift": measured_shift.n / len(template), 
            "shift_unc": measured_shift.s / len(template), 
            "subint_snr": subint_snr
        }

    def generate_template(self, smooth_sigma=None):
        return generate_template_from_stacked_data(
            self.data, 
            smooth_sigma=smooth_sigma, 
            logger=self.logger
        )

    def __compute_expected_shifts(self, alias_factor):
        # Compute slope in the unit of phase/day
        slope = 1 / (self.sidereal_day / alias_factor)

        # Compute shifts
        shifts = slope * self.epochs

        return shifts

    def find_aliasing_by_shifts(self, af_min=-30, af_max=30, smooth_sigma=None, ignore_subints=[]):
        
        # Generate template
        template = self.generate_template(smooth_sigma=smooth_sigma)

        # Find shifts for each subintegration
        with multiprocessing.Pool(self.ncpus) as pool:
            self.logger.debug(f"Calculating shifts for {self.n_subints} subintegration...")
                
            # run get shifts
            get_shift_res = list(
                tqdm.tqdm(
                    pool.imap(
                        self._find_shift, 
                        [(template, this_subint_powers) for this_subint_powers in self.data]
                    ),
                    desc="Calculating shifts", 
                    unit="subint", 
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    leave=False
                )
            )

            # unpack results
            measured_shifts = np.array([])
            measured_shifts_unc = np.array([])
            subint_snrs = np.array([])
            for res in get_shift_res:
                measured_shifts = np.append(measured_shifts, res["shift"])
                measured_shifts_unc = np.append(measured_shifts_unc, res["shift_unc"])
                subint_snrs = np.append(subint_snrs, res["subint_snr"])

        # Create a mask for subints
        mask = np.ones(len(measured_shifts), dtype=bool)
        if len(ignore_subints) > 0:
            mask[ignore_subints] = False

        # Find outlier indices based on shifts_unc
        mask = mask & (measured_shifts_unc - np.median(measured_shifts_unc) < 3 * np.std(measured_shifts_unc))

        # calcualte weights
        weight = 1 / measured_shifts_unc**2 # calculate weight

        # search for alias factor
        self.logger.debug(f"Searching for alias factor in range [{af_min}, {af_max})")
        trials = np.arange(af_min, af_max)
        rms = []
        expected_shifts = []
        for trial in trials:
            # compute expected shifts
            this_expected_shifts = self.__compute_expected_shifts(trial)

            # compute residuals
            this_residuals = measured_shifts[mask] - this_expected_shifts[mask]

            # subtract residuals by mean
            # this_residuals -= np.mean(this_residuals)
            this_mean = np.sum(weight[mask] * this_residuals) / np.sum(weight[mask])
            this_residuals -= this_mean

            # get rms
            this_rms = np.sqrt(np.sum(np.array(weight)[mask] * (this_residuals)**2) / np.sum(np.array(weight)[mask]))

            rms.append(this_rms)
            expected_shifts.append(this_expected_shifts + this_mean) # add offset back for diagnostic plot

        # measure a few more metrics on the stacked data
        stacked_snr = data_quality_utils.boxcar_snr(self.data.sum(axis=0))
        avg_snr_per_subints = np.mean(subint_snrs)

        # gather additional metadata
        psr_id = "unknown"
        try:
            model = get_model(StringIO(self.metadata.model))
            if hasattr(model, "PSR"):
                psr_id = model.PSR.value
            elif hasattr(model, "PSRJ"):
                psr_id = model.PSRJ.value
        except Exception as e:
            self.logger.warning(f"Failed to parse model to get pulsar name: {e}")

        n_day_stacked = "unknown"
        n_day_stacked = len(self.metadata.input_files)

        return AliasingFinderResultsByShifts(
            shifts_expected=expected_shifts,  
            shifts_measured_vals=measured_shifts, 
            shifts_measured_errs=measured_shifts_unc, 
            mask=mask, 
            af_trials=trials, 
            rms=rms, 
            data=self.data,
            template=template, 
            stacked_snr=stacked_snr, 
            avg_snr_per_subints=avg_snr_per_subints,
            psr_id=psr_id,
            n_day_stacked=n_day_stacked, 
            logger=self.logger
        )
    
    def find_aliasing_by_snr(self, af_min=-30, af_max=30, smooth_sigma=None, ignore_subints=[]):
        from ..utils.correlation import fourier_shifts
        from ..utils.data_quality import MatchedFilterSNR, data_quality_utils

        # Generate template
        template = self.generate_template(smooth_sigma=smooth_sigma)

        # Loop over trials and measure SNR for each
        snrs = []
        af_vals = np.arange(af_min, af_max)
        for this_af in af_vals:
            # get expected shifts as a function of subint index relative to the start of the observation
            this_slope = (this_af * self.duration) / (self.sidereal_day * self.n_subints)

            # apply mask
            this_corrected_data = copy.deepcopy(self.data)
            for subint_idx in ignore_subints:
                this_corrected_data[subint_idx, :] = 0

            # correct for the shifts
            for subint_idx in range(self.n_subints):
                shift = subint_idx * this_slope
                this_corrected_data[subint_idx, :] = fourier_shifts.roll(this_corrected_data[subint_idx, :], -shift*self.n_bins)

            # create profile
            this_profile = np.sum(this_corrected_data, axis=0)
            # plt.plot(this_profile)

            # calculate matched filter snr
            snrs.append(
                MatchedFilterSNR(this_profile, template, shift_meth="fourier").compute()
            )

        return AliasingFinderResultsBySNR(
            af_vals=af_vals,
            snrs=snrs, 
            logger=self.logger
        )

        # # find where rms is minimum
        # best_i = np.where(rms == np.min(rms))[0][0]
        # best_alias_factor = trials[best_i]
        # best_expected_shifts = shifts_expected[best_i]

        # # get expected shifts for nearby alias factors for diagnostic plot
        # nearby_expected_shifts = []
        # if best_alias_factor > af_min:
        #     nearby_expected_shifts.append(shifts_expected[best_i - 1])
        # if best_alias_factor < af_max:
        #     nearby_expected_shifts.append(shifts_expected[best_i + 1])

        # # plot diagnostic
        # self.cf_plot_diagnostic(
        #     trials = trials,
        #     rms = rms,
        #     good_i = mask,
        #     shifts_x = subint_idxes,
        #     shifts = shifts,
        #     shifts_unc = shifts_unc, 
        #     shifts_expected = best_expected_shifts,
        #     nearby_expected_shifts = nearby_expected_shifts,
        #     best_af = best_alias_factor,
        #     std_profile = std_profile,
        #     data_stacked = data_stacked, 
        #     n_stacked = self.su.n_stacked
        # )

        # # print summary
        # self.logger.success(
        #     pd.DataFrame({
        #         "Alias Factor": [best_alias_factor], 
        #         "Stacked S/N": [stacked_snr],
        #         "Avg S/N Per Obs": [avg_snr],
        #         "Avg S/N Per Subints": [avg_snr_per_subints]
        #     })
        # )

        # self.alias_factor = float(best_alias_factor) # ensure it is stored as a float instead of np.float64
        # self.snr_stacked = float(stacked_snr)
        # self.avg_snr_per_obs = float(avg_snr)
        # self.avg_snr_per_subints = float(avg_snr_per_subints)

        # return best_alias_factor








    
    
    # def calc_dealias_factor(self, obs_phase_shift, obs_length, obs_interval=1):
    #     return (obs_interval / obs_length) * obs_phase_shift

    # def get_stacked_powers_and_duration(self):
    #     # Reducing data dimensions: (time, pol, freq, phase) -> (time, freq, phase) -> (time, phase)
    #     return self.su.get_data(fscrunch=True, keepdims=True)[:, 0, 0, :], np.mean(self.su.durations), np.mean(self.su.snrs)

    # def dealias(self, save_pkl=False):
    #     if self.alias_factor is None:
    #         raise ValueError("Alias factor not available. Run tp_get_alias_factor or cf_get_alias_factor first.")

    #     # Initialize dealias_utils instance
    #     self.logger.debug(f"Initializing dealias_utils...")
    #     du = dealias_utils(
    #         psrdir=self.psrdir,
    #         parfile=self.parfile,
    #         workspace=self.workspace,
    #         outdir=self.outdir,
    #         logger=self.logger.copy()
    #     )

    #     # Try to dealias
    #     self.logger.debug(f"Dealiasing with alias_factor={self.alias_factor}...")
    #     du_res = du.dealias(self.alias_factor)

    #     # # Save diagnostic plots
    #     # if os.path.exists(f"{self.workspace}/diagnostic.pdf"):
    #     #     self.logger.info(f"Find alias diagnostic plot > {du.psrdir_dealias}")
    #     #     shutil.move(self.workspace + "/diagnostic.pdf", f"{du.psrdir_dealias}/diagnostic.pdf")

    #     # Generate summary
    #     self.summary = {
    #         "psr_id": self.psrdir.split("/")[-1],
    #         "n_stacked": self.su.n_stacked, 
    #         "alias_factor": self.alias_factor, 
    #         "snr_stacked": self.snr_stacked,
    #         "avg_snr_per_obs": self.avg_snr_per_obs,
    #         "avg_snr_per_subints": self.avg_snr_per_subints,
    #         "notes": {"remark": "DEALIAS_FITTING_OK" if du_res else "DEALIAS_FITTING_FAILED"}
    #     }

    #     # Include dealias result in the summary
    #     for key, value in du.info.items():
    #         self.summary[key] = value

    #     return self.summary

    # def weighted_mean(self, vals, errs):
    #     weights = 1 / errs**2
    #     return np.sum(weights * vals) / np.sum(weights)

    # def save_outfiles(self, filename="auto", pickle_only=False):
    #     # Get filename
    #     if filename == "auto":
    #         filename = self.psrdir + "/dealias"
            
    #     # Check if outdir exists
    #     if (not os.path.exists(self.outdir) or glob.glob(self.outdir) == []) and not pickle_only:
    #         raise Exception(f"Directory does not exist / empty: {self.outdir}. Was the alias_utils initialized or loaded from a pickle file?")

    #     # Save pickle
    #     with open(self.outdir + "/alias_utils.pkl", "wb") as f:
    #         pickle.dump(self, f)
    #         self.logger.success(f"Pickle saved to {self.outdir}/alias_utils.pkl")
    #         if pickle_only:
    #             return

    #     # Save stacked data
    #     self.su.save(
    #         filename=os.path.join(self.outdir, "stacked_data.npz"), 
    #         format="npz"
    #     )

    #     # Save summary to json 
    #     with open(os.path.join(self.outdir, "summary.json"), "w") as f:
    #         # Make sure everything is serializable
    #         summary_json = {}
    #         for key, value in self.summary.items():
    #             # np float 
    #             if isinstance(value, np.floating):
    #                 value = float(value)
    #             elif isinstance(value, np.integer):
    #                 value = int(value)

    #             summary_json[key] = value
    #         json.dump(summary_json, f, indent=4)
            
    #     # Copy everything in outdir to filename
    #     if not os.path.isdir(filename):
    #         if os.path.exists(filename):
    #             raise Exception(f"Directory already exists and is not writable: {filename}. Is there a file with the same name?")
    #     else:
    #         # Remove the directory
    #         shutil.rmtree(filename)

    #     # Create the directory
    #     os.makedirs(filename)

    #     # Copy the contents of outdir to filename
    #     for f in glob.glob(self.outdir + "/*"):
    #         self.logger.debug(f"Copying {f} to {filename}", layer=1)
    #         if os.path.isdir(f):
    #             shutil.copytree(f, os.path.join(filename, os.path.basename(f)))
    #         else:
    #             shutil.copy2(f, filename)
    #     self.logger.success(f"Save outputs to {filename}")
        
    #     # self.cleanup_workspace = False

    # def write_db(self, db_path="auto", db_hdl=None):
    #     # Check if summary is available
    #     if not self.summary:
    #         raise ValueError("No summary available. Please run dealias() first.")

    #     # Check if a database handle is provided
    #     if db_hdl is None:
    #         # Get db_path
    #         if db_path == "auto":
    #             db_path = f"{self.psrdir}/champss_timing.sqlite3.db"
            
    #         # Create a new database
    #         self.logger.debug(f"Creating new database at {db_path}")
    #         loaded_db_hdl = database(db_path)
    #     else: 
    #         loaded_db_hdl = db_hdl

    #     # Write database
    #     loaded_db_hdl.insert_dealias_history(
    #         n_stacked=float(self.summary["n_stacked"]), # ensure they are not np.float64
    #         alias_factor=float(self.summary["alias_factor"]),
    #         snr_stacked=float(self.summary["snr_stacked"]),
    #         notes=self.summary["notes"]
    #     )

    #     # Close db_hdl
    #     if db_hdl is None:
    #         self.logger.debug(f"Closing database at {db_path}")
    #         loaded_db_hdl.close()

    #     self.logger.success(f"Database updated: {db_path}")

    # def cleanup(self):
    #     if self.cleanup_workspace:
    #         self.logger.debug("Cleaning up workspace")
    #         shutil.rmtree(self.workspace)
    #     else:
    #         self.logger.debug(f"Workspace will NOT be removed at {self.workspace} due to cleanup=False")

    # def commit(self, db_hdl=None):
    #     '''
    #     Commit the changes to the psrdir and database

    #     Parameters:
    #     - db_hdl: Optional database handle. If not provided, a new database connection will be created and closed within the method.
    #     '''

    #     # Commit changes to psrdir
    #     self.logger.debug(f"Committing changes to psrdir: {self.psrdir}")
    #     self.logger.level_up()
    #     self.save_outfiles()
    #     self.logger.level_down()

    #     # Commit changes to database
    #     self.logger.debug(f"Committing changes to database: {self.psrdir}")
    #     self.logger.level_up()
    #     self.write_db(db_hdl=db_hdl)
    #     self.logger.level_down()

    # def __enter__(self):
    #     self.initialize()
    #     return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     if exc_type is not None:
    #         self.logger.error(f"Error while running find_alias. Workspace will NOT be removed at {self.workspace}")
    #         try:
    #             self.save(self.workspace + "/alias_utils.pkl")
    #             self.logger.error(f"Alias_utils object saved to {self.workspace}/alias_utils.pkl")
    #         except:
    #             self.logger.error(f"Error while saving alias_utils object to {self.workspace}/alias_utils.pkl")
    #         raise exc_type(exc_value).with_traceback(traceback)
    #     else:
    #         self.cleanup()