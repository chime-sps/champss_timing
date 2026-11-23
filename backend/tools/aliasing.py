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
import matplotlib.pyplot as plt
import datetime
import traceback
from pint import models
from pint.fitter import WLSFitter
from pint.models import get_model, get_model_and_toas
from scipy.ndimage import gaussian_filter
from pint.residuals import Residuals
from scipy.stats import median_abs_deviation
from io import StringIO

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

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

def get_aliasing_finder_from_stackfile(stackfile, n_subs=None, inj_af=None, **kwargs):
    from ..tools.stacked_data import StackedData
    
    # read stackfile
    stacked_data = StackedData.load(stackfile)

    # inject aliasing if specified
    if inj_af is not None:
        stacked_data.dealiasing(-inj_af)

    # drop partially filled subintegrations
    stacked_data.drop_partial_subints()

    # scrunch polarisation, frequencies
    stacked_data.scrunch(axis=1)
    stacked_data.scrunch(axis=2)

    # downsample
    if n_subs is not None:
        stacked_data.downsample_to_length(n_subs, axis=0)

    return AliasingFinder(
        stacked_data=stacked_data,
        **kwargs
    )

def get_smooth_sigma(template):
    width = data_quality_utils.pulse_width(template)
    return width * 0.2

def generate_template_from_stacked_data(stacked_data, smooth_sigma=None, logger=logger()):
    # Get subint data
    data = stacked_data.get_data(scrunch=(1, 2), normalize=True)[:, 0, 0, :]
    
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
    def __init__(self, stacked_data, trials, subint_powers, template, logger):
        self.epochs = stacked_data.epochs
        self.trials = trials
        self.subint_powers = subint_powers
        self.template = template
        self.logger = logger

        # Get stack and subint snrs
        self.stacked_snr = stacked_data.get_stacked_snr()
        self.subint_snrs = stacked_data.get_subint_snrs()

        # Get pulsar name
        self.psr_id = None
        try:
            model = get_model(StringIO(stacked_data.meta.model))
            if hasattr(model, "PSR"):
                self.psr_id = model.PSR.value
            elif hasattr(model, "PSRJ"):
                self.psr_id = model.PSRJ.value
        except Exception as e:
            self.psr_id = "unknown"

        # Get number of days stacked
        self.n_day_stacked = len(stacked_data.meta.input_files)

        # Find the best trial
        self.best_idx, self.nearby_trials_idx = self.__get_best_trial()

        # Get best and nearby shifts as shortcuts
        self.best_trial = self.trials[self.best_idx]
        self.nearby_trials = [self.trials[i] for i in self.nearby_trials_idx]

        # Get the subint powers after correcting for aliasing at the best trial
        stacked_data_corr = stacked_data.copy()
        stacked_data_corr.dealiasing(alias_factor=self.best_trial.alias_factor)
        self.subint_powers_corr = stacked_data_corr.get_subints(normalize=True)

    ############################################
    # Helper functions
    ############################################

    def __get_best_trial(self):
        # Get rms values
        rms_values = self.get_rms_values()

        # # Get alias factors
        # alias_factors = self.get_alias_factors()

        # Find the best index
        best_idx = np.argmin(rms_values)

        # Get nearby trials
        nearby_trials_idx = []
        if best_idx > 0:
            nearby_trials_idx.append(best_idx - 1)
        if best_idx < len(rms_values) - 1:
            nearby_trials_idx.append(best_idx + 1)

        return best_idx, nearby_trials_idx

    def get_alias_factors(self):
        return [trial.alias_factor for trial in self.trials]

    def get_rms_values(self):
        return [trial.rms for trial in self.trials]

    def validate(self):
        if len(self.trials) == 0:
            raise ValueError("No trials have been added.")

    ############################################
    # Dealiasing
    ############################################

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
            best_af=self.best_trial.alias_factor,
            logger=self.logger
        )

    ############################################
    # Diagnostic functions
    ############################################

    def __normalize_power(self, power):
        power = power - np.mean(power)
        power = power / np.max(power)

        return power
    
    def get_summary(self):
        return {
            "psr_id": str(self.psr_id),
            "n_stacked": int(self.n_day_stacked),
            "alias_factor": int(self.get_alias_factors()[self.best_idx]),
            "snr_stacked": float(self.stacked_snr),
            "avg_snr_per_subints": float(np.mean(self.subint_snrs)),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def print_summary(self):
        self.logger.info("Aliasing Finder Summary:")
        for key, value in self.get_summary().items():
            self.logger.info(f"{key}: {value}", layer=1)

    def plot(self, savefig=None):
        fig, axs = plt.subplots(3, 3, figsize=(16, 8),  gridspec_kw={'width_ratios': [3, 1, 1]}, constrained_layout=True)

        # plot rms
        axs[0, 0].plot(self.get_alias_factors(), self.get_rms_values(), c="k", lw=1)
        axs[0, 0].axvline(
            self.best_trial.alias_factor, 
            label=f"Best fit: aliasing by {self.best_trial.alias_factor} phase(s) per day", 
            c="r", 
            lw=1
        )
        axs[0, 0].set_title(f"RMS vs Alias Factor")
        axs[0, 0].set_xlabel("Alias Factor")
        axs[0, 0].set_ylabel("RMS")
        axs[0, 0].set_yscale("log")
        axs[0, 0].legend(frameon=False, fontsize=8)

        # plot shifts
        axs[1, 0].errorbar(
            self.epochs[self.best_trial.mask] * 86400, 
            self.best_trial.get_values(masked=True), 
            self.best_trial.get_uncertainties(masked=True), 
            c="k", 
            lw=1, 
            label="Measured TOAs", 
            capsize=2, 
            fmt="x"
        )
        axs[1, 0].errorbar(
            self.epochs * 86400, 
            self.best_trial.get_expected_shifts(masked=False), 
            c="r", 
            lw=1, 
            linestyle="-", 
            label=f"Best fit: aliasing by {self.best_trial.alias_factor} phase(s) per day"
        )
        for i, this_nearby_shift in enumerate(self.nearby_trials):
            axs[1, 0].errorbar(
                self.epochs * 86400, 
                this_nearby_shift.get_expected_shifts(masked=False),  
                c="k", 
                lw=1, 
                linestyle="--", 
                label=f"Best fit +/- 1 phase per day" if i == 0 else None, 
                alpha=0.5
            )
        axs[1, 0].set_title("Shifts")
        axs[1, 0].set_xlabel("Time since reference epoch (s)")
        axs[1, 0].set_ylabel("Shift (phase)")
        axs[1, 0].legend(frameon=False, fontsize=8)

        # plot powers
        axs[2, 0].plot(
            self.__normalize_power(self.template), 
            c="k", 
            lw=1, 
            label="Std Profile"
        )
        axs[2, 0].plot(
            self.__normalize_power(
                np.sum(
                    self.subint_powers, 
                    axis=0
                )
            ), 
            c="k", 
            lw=1, 
            label=f"Sum of subints 0-{int(len(self.subint_powers))-1}", 
            alpha=0.75, 
            linestyle=":"
        )
        axs[2, 0].plot(
            self.__normalize_power(
                np.sum(
                    self.subint_powers[:int(self.best_trial.n_subints/2)], 
                    axis=0
                )
            ), 
            c="r", 
            lw=1, 
            label=f"Sum of subints 0-{int(self.best_trial.n_subints/2)-1}", 
            alpha=0.75
        )
        axs[2, 0].plot(
            self.__normalize_power(
                np.sum(self.subint_powers[int(self.best_trial.n_subints/2):], 
                       axis=0
                )
            ), 
            c="b", 
            lw=1, 
            label=f"Sum of subints {int(self.best_trial.n_subints/2)}-{self.best_trial.n_subints-1}", 
            alpha=0.75
        )
        axs[2, 0].set_title(f"Powers")
        axs[2, 0].set_xlabel("Samples")
        axs[2, 0].set_ylabel("Power")
        axs[2, 0].legend(frameon=False, fontsize=8)

        # # normalize stacked data
        # for i, _ in enumerate(self.data):
        #     self.data[i] = np.array(self.data[i]) - min(self.data[i])
        #     self.data[i] = np.array(self.data[i]) / max(self.data[i])

        # find the peack of the stacked profiles
        peak_index = np.argmax(np.sum(self.subint_powers, axis=0))

        # calculate the shifts in phase
        shifts_phase = []
        for shift in self.best_trial.get_expected_shifts(masked=False):
            this_shift = peak_index + (shift / 1) * len(self.subint_powers[0])
            if this_shift > len(self.subint_powers[0]):
                this_shift -= len(self.subint_powers[0])
            elif this_shift < 0:
                this_shift += len(self.subint_powers[0])
            shifts_phase.append(this_shift)

        # generate x-axis for plotting
        subint_power_x = np.arange(len(self.subint_powers[0]))

        # plot stacked profile
        axs[0, 1].plot(np.sum(self.subint_powers, axis=0), lw=0.5, c="k")
        axs[0, 1].text(
            0.95, 
            0.95, 
            f"Observed", 
            ha="right", 
            va="top", 
            fontsize=8,
            transform=axs[0, 1].transAxes
        )
        axs[0, 1].set_title("Stacked Profile")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # plot stacked data
        axs[1, 1].remove()
        axs[2, 1].remove()
        n_params_gs = axs[0, 1].get_gridspec()
        axs_subints = fig.add_subplot(n_params_gs[1:3, 1])
        # axs_subints.matshow(self.subint_powers, cmap="gray_r", aspect="auto", rasterized=True)
        axs_subints.pcolormesh(subint_power_x, self.epochs * 86400, self.subint_powers, cmap="gray_r", rasterized=True)
        axs_subints.plot(shifts_phase, self.epochs * 86400, color="r", lw=1, ls=":", label="Best fit")
        axs_subints.plot(
            [peak_index - 0.015 * len(self.subint_powers[0])] * len(self.epochs), 
            self.epochs * 86400, 
            "r-", 
            label="No aliasing", 
            lw=0.5, 
            alpha=0.75
        )
        axs_subints.set_title("Subint Powers")
        axs_subints.legend(frameon=False, fontsize=8)
        axs_subints.set_xlabel("Phase")
        axs_subints.set_ylabel("Sample")
        axs_subints.set_xticks(np.arange(0, len(self.subint_powers[0]) + 1, len(self.subint_powers[0]) / 2))
        axs_subints.set_xticklabels(["0", "0.5", "1"])
        axs_subints2 = axs_subints.twiny()
        axs_subints2.set_xlim(axs_subints.get_xlim())

        if self.best_trial.alias_factor != 0:
            # plot stacked profile after correction for aliasing
            axs[0, 2].plot(np.sum(self.subint_powers_corr, axis=0), lw=0.5, c="k")
            axs[0, 2].text(
                0.95, 
                0.95, 
                f"De-aliased", 
                ha="right", 
                va="top", 
                fontsize=8,
                transform=axs[0, 2].transAxes
            )
            axs[0, 2].set_title(f"Stacked Profile")
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])

            # plot stacked data after correction for aliasing
            axs[1, 2].remove()
            axs[2, 2].remove()
            n_params_gs = axs[0, 2].get_gridspec()
            axs_subints_corr = fig.add_subplot(n_params_gs[1:3, 2])
            # axs_subints_corr.matshow(self.subint_powers_corr, cmap="gray_r", aspect="auto", rasterized=True)
            axs_subints_corr.pcolormesh(subint_power_x, self.epochs * 86400, self.subint_powers_corr, cmap="gray_r", rasterized=True)
            axs_subints_corr.plot(shifts_phase, self.epochs * 86400, color="r", lw=1, ls=":", label="Best fit")
            axs_subints_corr.plot(
                [peak_index - 0.015 * len(self.subint_powers_corr[0])] * len(self.epochs), 
                self.epochs * 86400, 
                "r-", 
                label="No aliasing", 
                lw=0.5, 
                alpha=0.75
            )
            axs_subints_corr.set_title(f"Subint Powers")
            axs_subints_corr.legend(frameon=False, fontsize=8)
            axs_subints_corr.set_xlabel("Phase")
            axs_subints_corr.set_ylabel("Sample")
            axs_subints_corr.set_xticks(np.arange(0, len(self.subint_powers_corr[0]) + 1, len(self.subint_powers_corr[0]) / 2))
            axs_subints_corr.set_xticklabels(["0", "0.5", "1"])
            axs_subints_corr2 = axs_subints_corr.twiny()
            axs_subints_corr2.set_xlim(axs_subints_corr.get_xlim())
        else:
            for ax in axs[:, 2]:
                ax.remove()

            axs_subints_corr = fig.add_subplot(n_params_gs[:, 2])
            axs_subints_corr.text(
                0.5, 0.5, 
                "No aliasing correction required", 
                ha="center", 
                va="center", 
                transform=axs_subints_corr.transAxes
            )
        
        # plot info
        fig.text(
            0.001, 
            0.000, 
            f"CHAMPSS Timing Pipeline alias_utils ({utils.get_version_hash()}) | {self.psr_id} | {self.n_day_stacked} observations stacked | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            fontsize=9, 
            ha="left", 
            va="bottom", 
            family="monospace"
        )

        fig.tight_layout()

        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()

class AliasingFinderResultsBySNR:
    def __init__(
            self, 
            stacked_data,
            af_vals, 
            metric_name,
            metric_vals, 
            logger
        ):
        self.stacked_data = stacked_data.copy()
        self.af_vals = af_vals
        self.metric_name = metric_name
        self.metric_vals = metric_vals
        self.logger = logger

        # Find the best shift based on SNR
        max_metric_idx = np.argmax(self.metric_vals)
        self.best_af = self.af_vals[max_metric_idx]
        self.best_metric = self.metric_vals[max_metric_idx]

        # Dealias the stacked data
        self.stacked_data.dealiasing(alias_factor=self.best_af)

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
            "metric_name": self.metric_name,
            "best_metric": self.best_metric,
            "n_trials": len(self.af_vals)
        }
        return summary

    def print_summary(self):
        summary = self.get_summary()
        self.logger.info("Aliasing Finder Results Summary:")
        self.logger.info(f"Best AF: {summary['best_af']}")
        self.logger.info(f"Metric Name: {summary['metric_name']}")
        self.logger.info(f"Best Metric: {summary['best_metric']}")
        self.logger.info(f"Number of Trials: {summary['n_trials']}")

    def plot(self, savefig=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.af_vals, self.metric_vals, 'kx')
        plt.axvline(self.best_af, color='r', linestyle='--', label=f"Best {self.metric_name.upper()} at {self.best_af} phase(s) per day")
        plt.xlabel("Shift")
        plt.ylabel(self.metric_name.upper())
        plt.title("Aliasing Finder Results by SNRs")
        plt.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()

class AliasingFinderMeasuredShifts:
    def __init__(self, measured_shifts):
        self.values = np.array([])
        self.uncertainties = np.array([])
        self.snrs = np.array([])
        self.subint_idx = np.array([])

        for measured_shift in list(measured_shifts):
            self.values = np.append(self.values, measured_shift["value"])
            self.uncertainties = np.append(self.uncertainties, measured_shift["uncertainty"])
            self.snrs = np.append(self.snrs, measured_shift["snr"])
            self.subint_idx = np.append(self.subint_idx, measured_shift["subint_idx"])

    def validate(self):
        # Check for same lengths
        if len(self.values) != len(self.uncertainties) or len(self.values) != len(self.snrs) or len(self.values) != len(self.subint_idx):
            raise ValueError("Measured shifts arrays must have the same length.")

        # Check for any non-real values
        if not np.all(np.isreal(self.values)):
            raise ValueError("All measured shift values must be real numbers.")
        if not np.all(np.isreal(self.uncertainties)):
            raise ValueError("All measured shift uncertainties must be real numbers.")
        if not np.all(np.isreal(self.snrs)):
            raise ValueError("All measured shift SNRs must be real numbers.")

class AliasingFinderShiftTrials:
    def __init__(self, alias_factor, measured_shifts, expected_shifts, mask, n_subints):
        self.alias_factor = alias_factor
        self.mask = mask
        self.n_subints = n_subints

        # Initialize shifts
        self.subint_idx = np.asarray(measured_shifts.subint_idx)
        self.values = np.asarray(measured_shifts.values)
        self.uncertainties = np.asarray(measured_shifts.uncertainties)
        self.snrs = np.asarray(measured_shifts.snrs)
        self.expected_shifts = np.asarray(expected_shifts)

        # Initialize rms etc
        self.rms = None
        self.mean = None

    ##############################################
    # Helper functions
    ##############################################

    def get_values(self, masked=True):
        if masked:
            return self.values[self.mask]
        return self.values

    def get_uncertainties(self, masked=True):
        if masked:
            return self.uncertainties[self.mask]
        return self.uncertainties

    def get_expected_shifts(self, masked=True):
        if masked:
            return self.expected_shifts[self.mask]
        return self.expected_shifts

    ##############################################
    # Creating and managing subint shifts
    ##############################################

    def compute_rms(self):
        # Calcualte weights
        weight = 1 / self.uncertainties**2 # calculate weight

        # Get residuals for masked subints
        residuals = self.values[self.mask] - self.expected_shifts[self.mask]

        # Subtract residuals by mean
        # this_residuals -= np.mean(this_residuals)
        self.mean = np.sum(weight[self.mask] * residuals) / np.sum(weight[self.mask])
        residuals -= self.mean

        # Calculate rms
        self.rms = np.sqrt(np.sum(np.array(weight)[self.mask] * (residuals)**2) / np.sum(np.array(weight)[self.mask]))

    def validate(self):
        if self.subint_idx.size == 0:
            raise ValueError("No subints added.")
        
        if not (len(self.subint_idx) == len(self.values) == len(self.uncertainties) == len(self.snrs)):
            raise ValueError("Mismatch in lengths of subint data.")
        
        if len(self.expected_shifts) != 0 and len(self.expected_shifts) != len(self.subint_idx):
            raise ValueError("Mismatch in lengths of expected shifts and subint data.")
        
        if len(self.mask) != 0 and len(self.mask) != len(self.subint_idx):
            raise ValueError("Mismatch in lengths of mask and subint data.")
        
        if len(self.subint_idx) != self.n_subints:
            raise ValueError("Number of subints added does not match n_subints.")

        if self.rms is None or self.mean is None:
            raise ValueError("RMS and mean have not been computed.")

        if np.diff(self.subint_idx).min() <= 0:
            raise ValueError("Subint indices are not strictly increasing.")
        

class AliasingFinder():
    def __init__(self, stacked_data, ncpus=1, logger=logger()):
        """
        Initialize the alias_utils class.

        Parameters:
        data (np.ndarray): Pulsar data array with shape (n_subs, n_freqs, n_bins).
        epochs (np.ndarray): Array of subint epochs in days.
        ncpus (int): Number of CPUs to use.
        logger (logger): Logger instance.
        metadata (dict): Additional metadata (from stack file) for the observation.
        """

        self.stacked_data = stacked_data.copy()
        self.ncpus = ncpus
        self.logger = logger

        # Get sidereal day constant from astropy
        self.sidereal_day = (1 * u.sday).to(u.day).value

    def __find_shift(self, payload):

        # Unpack payload
        n_steps, template, subint, subint_idx = payload

        # Initialize shift finder
        shift_finder = ShiftFinder(template, subint)

        # Find shift
        shift_finder.compute(n_steps=n_steps, burn_in=int(n_steps/2), progress=False)

        # Get shift result
        measured_shift = shift_finder.result.shift

        # Measure subint SNR
        subint_snr = data_quality_utils.boxcar_snr(subint)

        return {
            "value": measured_shift.n / len(template),
            "uncertainty": measured_shift.s / len(template),
            "snr": subint_snr,
            "subint_idx": subint_idx
        }

    def __compute_expected_shifts(self, alias_factor):
        # Compute slope in the unit of phase/day
        slope = 1 / (self.sidereal_day / alias_factor)

        # Compute shifts
        shifts = slope * self.stacked_data.epochs

        return shifts

    def find_aliasing_by_shifts(self, af_min=-30, af_max=30, smooth_sigma=None, n_steps=1000, ignore_subints=[]):

        # Generate template by maximizing SNR
        template = self.find_aliasing_by_snr(
            af_min=af_min,
            af_max=af_max,
            smooth_sigma=smooth_sigma,
            ignore_subints=ignore_subints
        ).stacked_data.get_profile()

        # Smooth the template
        if smooth_sigma is None:
            smooth_sigma = get_smooth_sigma(template)
        self.logger.debug(f"Smoothing template with sigma={smooth_sigma}")
        template = gaussian_filter(template, sigma=smooth_sigma, mode="wrap")

        # # Generate template
        # template = generate_template_from_stacked_data(
        #     stacked_data=self.stacked_data,
        #     smooth_sigma=smooth_sigma, 
        #     logger=self.logger
        # )

        # Get subint powers from stacked data
        subint_powers = self.stacked_data.get_data(scrunch=(1, 2), normalize=True)[:, 0, 0, :]

        # Get subint count
        n_subints = len(subint_powers)

        # Generate aliasing factor trials
        af_trials = np.arange(af_min, af_max)

        # Find shifts for each subintegration
        with Pool(self.ncpus) as pool:
            self.logger.debug(f"Calculating shifts for {self.stacked_data.get_shape()[0]} subintegration...")
                
            # run get shifts
            measured_shifts = AliasingFinderMeasuredShifts(
                tqdm.tqdm(
                    pool.imap(
                        self.__find_shift, 
                        [(n_steps, template, subint, i) for i, subint in enumerate(subint_powers)]
                    ),
                    total=n_subints,
                    desc="Calculating shifts", 
                    unit="subint", 
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    leave=False
                )
            )

            # validate shifts
            measured_shifts.validate()

        # Create a mask for shift measurements
        mask = np.ones(n_subints, dtype=bool)
        if len(ignore_subints) > 0:
            mask[ignore_subints] = False

        # Mask out subints with uncertainties greater than 3 sigma from the median
        mask = mask & (measured_shifts.uncertainties - np.median(measured_shifts.uncertainties) < 3 * (median_abs_deviation(measured_shifts.uncertainties) * 1.4826))

        # Mask out subints with snr below 3 sigma from the median
        mask = mask & (measured_shifts.snrs >= np.median(measured_shifts.snrs) - 3 * (median_abs_deviation(measured_shifts.snrs) * 1.4826))

        # Loop over aliasing factor trials
        trial_instances = []
        for this_af in tqdm.tqdm(af_trials, desc="Taking trials", unit="trial", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", leave=False):

            # Initialize shifts container
            trial = AliasingFinderShiftTrials(
                alias_factor=this_af, 
                measured_shifts=measured_shifts,
                expected_shifts=self.__compute_expected_shifts(this_af), 
                mask=mask,
                n_subints=len(subint_powers)
            )

            # Compute residuals and rms between measured and expected shifts
            trial.compute_rms()

            # Validate shifts instance
            trial.validate()

            trial_instances.append(trial)

            # # initialize shifts container
            # shifts = list(
            #     tqdm.tqdm(
            #         pool.imap(
            #             self.__initialize_shift_instance, 
            #             [(this_af, len(subint_powers), measured_shifts) for this_af in af_trials]
            #         ),
            #         total=len(af_trials),
            #         desc="Taking trials", 
            #         unit="trial", 
            #         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            #         leave=False
            #     )
            # )

        # Loop over measured shifts to compare with the expected shifts
        results = AliasingFinderResultsByShifts(
            stacked_data=self.stacked_data,
            trials=trial_instances, 
            subint_powers=subint_powers,
            template=template,
            logger=self.logger
        )

        # Validate results
        results.validate()

        return results
    
    def find_aliasing_by_snr(self, af_min=-30, af_max=30, smooth_sigma=None, ignore_subints=[], metric="chi2"):
        from ..utils.correlation import fourier_shifts
        from ..utils.data_quality import MatchedFilterSNR, data_quality_utils

        # Generate template
        template = generate_template_from_stacked_data(
            stacked_data=self.stacked_data,
            smooth_sigma=smooth_sigma, 
            logger=self.logger
        )

        # Make a copy of the stacked data
        stacked_data = self.stacked_data.copy()

        # Scrunch polarisation and frequency to save computation and RAM
        stacked_data.scrunch(axis=1)
        stacked_data.scrunch(axis=2)

        # Apply mask
        for subint_idx in ignore_subints:
            stacked_data.data[subint_idx, :, :, :] = 0

        # Loop over trials and measure SNR for each
        metric_vals = []
        af_vals = np.arange(af_min, af_max)
        for this_af in af_vals:
            # Make a copy of the reduced stacked data
            this_data = copy.deepcopy(stacked_data)

            # Apply dealiasing shifts
            this_data.dealiasing(alias_factor=this_af)

            # Get SNR
            if metric == "snr":
                metric_vals.append(
                    MatchedFilterSNR(this_data.get_profile(), template, shift_meth="fourier").compute()
                )
            elif metric == "chi2":
                metric_vals.append(this_data.get_stacked_chi2())

        return AliasingFinderResultsBySNR(
            stacked_data=self.stacked_data,
            af_vals=af_vals,
            metric_name=metric,
            metric_vals=metric_vals, 
            logger=self.logger
        )
