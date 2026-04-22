import os
import glob
import psrchive
import pickle
import shutil
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
from scipy.ndimage import gaussian_filter
from pint.residuals import Residuals
from astropy.io import ascii
from astropy.table import Table

from ..utils.utils import utils
from ..utils.exec import exec
from ..utils.logger import logger
from ..tools.template_utils import StackTemplate
from ..tools.shift_finder import ShiftFinder
from ..pipecore.psrchive import psrchive_handler
from ..tools.stack_utils import stack_utils
from ..datastores.database import database
from ..datastores.archive_cache import archive_cache


class dealias_utils():
    def __init__(self, psrdir, parfile, workspace, outdir, logger=logger()):
        self.logger = logger
        self.psrdir = psrdir
        self.psrdir_dealias = psrdir + "/dealias"
        self.parfile = parfile
        self.workspace = workspace
        self.outdir = outdir

        self.info = {}

        if not os.path.exists(self.psrdir):
            raise FileNotFoundError(f"Directory not found: {self.psrdir}")

        if not os.path.exists(self.workspace):
            raise FileNotFoundError(f"Directory not found: {self.workspace}")

        if not os.path.exists(self.psrdir_dealias):
            os.makedirs(self.psrdir_dealias)

    def write_dealias_info(self, filename):
        if self.info == {}:
            raise ValueError("No dealias information available. Please run dealias_utils.dealias() first.")
        
        # add timestamp
        self.info["timestamp"] = utils.get_time_string()

        # write to file
        ascii.write(Table([self.info]), format='ecsv', output=filename, overwrite=True)

    def dealias(self, alias_factor):
        self.info["alias_factor"] = alias_factor

        # read the mjd_start and mjd_end from the parfile
        mjd_start, mjd_end = utils.read_start_end_from_parfile(self.parfile, raise_exception=False)

        # copy the parfile to the workspace
        shutil.copy(self.parfile, self.workspace + "/pulsar.dealias.par")

        # create timfile from database
        with database(f"{self.psrdir}/champss_timing.sqlite3.db", readonly=True) as db_hdl:
            open(self.workspace + "/pulsar.dealias.tim", "w").write(db_hdl.create_timfile(mjd_range=[mjd_start, mjd_end])) # create only mjds between the mjd range specified in the parfile

        # dealias
        if alias_factor != 0:
            dealias_res = self._dealias(
                alias_factor=alias_factor,
                parfile=self.workspace + "/pulsar.dealias.par",
                timfile=self.workspace + "/pulsar.dealias.tim",
                parfile_out=self.outdir + "/pulsar.dealiased.par"
            )

            # move parfile to psrdir
            if dealias_res:
                # shutil.copy(self.workspace + "/pulsar.dealiased.par", f"{self.psrdir_dealias}/pulsar.dealiased.par")
                # shutil.copy(self.workspace + "/pulsar.dealias.pdf", f"{self.psrdir_dealias}/pulsar.dealias.pdf")
                self.info["status"] = "ok"
                self.logger.success("Dealiased successfully")
            else:
                self.info["status"] = "error"
                self.logger.error("Dealiasing failed")
        else:
            dealias_res = True
            self.info["status"] = "not_needed"
            self.logger.success("No dealias needed, alias_factor=0")

            # if os.path.exists(f"{self.psrdir_dealias}/pulsar.dealias.pdf"):
            #     self.logger.debug(f"Removing {self.psrdir_dealias}/pulsar.dealias.pdf") # remove diagnostic plot to avoid confusion
            #     os.remove(f"{self.psrdir_dealias}/pulsar.dealias.pdf")

            # if os.path.exists(f"{self.psrdir_dealias}/pulsar.dealiased.par"):
            #     self.logger.debug(f"Removing {self.psrdir_dealias}/pulsar.dealias.pdf") # remove diagnostic plot to avoid confusion
            #     os.remove(f"{self.psrdir_dealias}/pulsar.dealiased.par")

        self.write_dealias_info(f"{self.outdir}/dealias_info.ecsv")
        self.logger.success(f"Dealiased information saved to outdir. ")

        return dealias_res

    def _dealias(self, alias_factor, parfile, timfile, parfile_out=None):
        # get model and p0
        m, t = models.get_model_and_toas(parfile, timfile)
        p0 = 1 / m.F0.value
        self.info["chi2r_aliased"] = m.CHI2R.value

        # calculate alias solution
        dp_alias = p0 / ((u.sday.to(u.s) * (1/alias_factor)) / p0)
        p_dealiased = p0 + dp_alias
        self.logger.info(f"Dealiased period: {p_dealiased} s, dp_alias: {dp_alias} s")
        self.info["p0_aliased"] = p0
        self.info["p0_unaliased"] = p_dealiased

        # update model
        m.F0.quantity = (1 / p_dealiased)
        m.F0.frozen = False

        # remove PHOFF component if exists
        if hasattr(m, "PHOFF"):
            self.logger.debug("Removing PHOFF component from model")
            m.remove_component("PhaseOffset")

        # filterout bad toas
        rs_aliased = Residuals(t, m).phase_resids
        i_good = np.abs(rs_aliased) < np.quantile(np.abs(rs_aliased), 0.95)
        t = t[i_good]
        rs_aliased = rs_aliased[i_good]

        # fit model
        try:
            f = WLSFitter(t, m)
            f.fit_toas()
            f.print_summary()
            self.info["chi2r_unaliased"] = f.model.CHI2R.value
        except Exception as e:
            self.logger.error("Dealiasing failed, fitting failed. ")
            self.logger.error(traceback.format_exc())
            return False

        # plot residuals
        plt.figure(figsize=(12, 8))
        rs = Residuals(t, m).phase_resids
        plt.errorbar(t.get_mjds(), rs_aliased, yerr=t.get_errors().to(u.s).value * f.model.F0.value, fmt='x', label='prefit (p_aliased)', color='black', zorder=0, alpha=0.25)
        plt.plot(t.get_mjds(), rs, "x", label='prefit (p_unaliased)', color='blue')
        plt.plot(t.get_mjds(), f.resids.phase_resids * f.model.F0.value, "x", label='postfit (p_unaliased)', color='green')
        plt.xlabel('MJD')
        plt.ylabel('Residual (phase)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.outdir + "/pulsar.dealias.pdf")
        self.logger.success(f"Dealiasing plot saved to {self.outdir}/pulsar.dealias.pdf")

        # check difference between prefit and postfit
        p0_postfit = 1 / f.model.F0.value
        if np.abs(p0_postfit - p_dealiased) > np.abs(dp_alias):
            self.logger.error("Dealiasing failed, postfit period is too far from dealiased period", "p0_postfit", p0_postfit, "p_dealiased", p_dealiased, "dp_alias", dp_alias)
            return False

        # check chi2r
        chi2r_prefit = m.CHI2R.value
        chi2r_postfit = f.model.CHI2R.value
        if chi2r_postfit > chi2r_prefit * 1.1:
            self.logger.error("Dealiasing failed, postfit chi2r is much higher than prefit chi2r. ")
            return False
        elif chi2r_postfit > chi2r_prefit:
            self.logger.warning("Dealiasing failed, postfit chi2r is higher than prefit chi2r. This alias factor may not be correct.")
        
        # write out new parfile
        if parfile_out is None:
            parfile_out = parfile + ".dealiased"
        f.model.write_parfile(parfile_out)
        self.logger.success(f"Dealiased parfile written to {parfile_out}")

        return True

class alias_utils():
    def __init__(self, psrdir, ar_list, parfile, jumps={}, n_subints=32, n_bins=128, workspace="/tmp", cleanup=True, mode="auto", n_pools="auto", logger=logger()):
        # self.ars = ars
        # self.ar_list = [ar["path"] for ar in ars]

        # set parameters
        self.psrdir = psrdir
        self.ar_list = ar_list
        self.parfile = parfile
        self.mode = mode
        self.fs = []
        self.fs_prepared = []
        self.jumps = jumps
        self.n_subints = n_subints
        self.n_bins = n_bins
        self.workspace = workspace + f"/{utils.get_time_string()}__{utils.get_rand_string()}"
        self.outdir = self.workspace + "/outfiles"
        self.sidereal_day = 0.99727 # day
        self.cleanup_workspace = cleanup
        self.alias_factor = None
        self.avg_snr = 0
        self.data_stacked = []
        self.su = None
        self.summary = None

        # set methods
        self.logger = logger
        self.exec_handler = exec
        self.n_pools = n_pools
        self.psrchive = psrchive_handler(self)

        # # get update eph methods from archive_cache
        # self.update_eph = archive_cache("").exec_update_model
        # self.get_md5 = archive_cache("").get_md5

    def _initialize__copy_ar(self, ar):
        # check file size
        if os.path.getsize(ar) == 0:
            self.logger.warning(f"File size is 0: {ar}, skipping", layer=1)
            return None
        self.logger.debug(f"Copying {ar} to workspace", layer=1)
        shutil.copy(ar, self.workspace)
        return f"{self.workspace}/{os.path.basename(ar)}"

    def initialize(self):
        # create workspace
        self.logger.debug(f"Creating workspace: {self.workspace}")
        os.makedirs(self.workspace)
        os.makedirs(self.outdir)
        
        # Initialize stack_utils
        self.su = stack_utils(
            files=self.ar_list,
            parfile=self.parfile,
            workspace=self.workspace, 
            n_subs=self.n_subints, n_pols=1, n_freqs=1, n_bins=self.n_bins, 
            n_pools=self.n_pools, 
            jumps=self.jumps, 
            logger=self.logger.copy()
        )

        # Stack
        self.su.stack(normalize=True)

        # Check if stacking was successful
        if self.su.n_stacked == 0:
            raise Exception("Stacking failed, no files stacked. This may be due to the input files being empty or all files being skipped due to failure in processing.")

    def get_shift(self, std_profile, power, n_steps=2000):
        # Initialize shift finder
        shift_finder = ShiftFinder(std_profile, power)

        # Find shift
        shift_finder.compute(n_steps=n_steps, burn_in=int(n_steps/2), progress=False)

        # Get shift result
        measured_shift = shift_finder.result.shift

        return {"shift": measured_shift.n, "shift_unc": measured_shift.s}
    
    def calc_dealias_factor(self, obs_phase_shift, obs_length, obs_interval=1):
        return (obs_interval / obs_length) * obs_phase_shift

    def get_stacked_powers_and_duration(self):
        return self.su.get_data()[:, 0, 0, :], np.mean(self.su.durations), np.mean(self.su.snrs)

    def cf_get_alias_factor(self, af_min=-30, af_max=30, smooth_sigma=5, subint_range=[]):
        # get stacked powers and duration
        data_stacked, duration_, avg_snr = self.get_stacked_powers_and_duration()
        n_subints = len(data_stacked[:, 0])

        # correct for duration
        duration = duration_ * (n_subints - 1) / n_subints
        # duration = duration_
        self.logger.info(f"Duration: {duration_} days, Corrected duration: {duration} days, Number of subints: {n_subints}, Number of subints: {n_subints}")
        
        # cross-correlating subints to minimize the possible shifts (from unknown aliasing)
        self.logger.debug("Generating standard profile by stacking subints...")
        stpl = StackTemplate(data_stacked, size=self.n_bins, shift_meth="fourier", logger=self.logger.copy())
        stpl.optimize()

        # sum subints to a std profile
        std_profile = stpl.get_template()

        # smooth the std profile if required
        std_profile = gaussian_filter(np.array(data_stacked).sum(axis=0), sigma=smooth_sigma)

        # get shifts
        with multiprocessing.Pool(self.n_pools) as pool:
            # pack arguments
            get_shift_args = []
            for this_subint_powers in data_stacked:
                # add to args
                get_shift_args.append((std_profile, this_subint_powers))
                
            # run get shifts
            get_shift_res = list(tqdm.tqdm(pool.starmap(self.get_shift, get_shift_args), total=len(get_shift_args), desc="Calculating shifts", unit="subint"))

            # unpack results
            shifts = []
            shifts_unc = []
            for i, res in enumerate(get_shift_res):
                shifts.append(res["shift"])
                shifts_unc.append(res["shift_unc"])

        # apply subint_range
        if len(subint_range) == 2:
            # shifts = shifts[subint_range[0]:subint_range[1]]
            # shifts_unc = shifts_unc[subint_range[0]:subint_range[1]]
            shifts_i_range = [int(np.floor(subint_range[0])), int(np.ceil(subint_range[1]))]
            shifts = shifts[shifts_i_range[0]:shifts_i_range[1]]
            shifts_unc = shifts_unc[shifts_i_range[0]:shifts_i_range[1]]

        # find outlier indices based on shifts_unc
        good_i = np.where(np.array(shifts_unc) < np.quantile(shifts_unc, 0.997))[0]
        # shifts = np.array(shifts)[good_i]
        # shifts_unc = np.array(shifts_unc)[good_i]

        # get shift in phase
        shifts = np.array(shifts) / len(std_profile)
        shifts_unc = np.array(shifts_unc) / len(std_profile)

        # prepare data for alias factor searching
        shifts_actual, shifts_actual_unc = np.array(shifts)[good_i], np.array(shifts_unc)[good_i] # apply masks
        weight = 1 / np.array(shifts_actual_unc)**2 # calculate weight

        # search for alias factor
        trials = np.arange(af_min, af_max)
        rms = []
        shifts_expected = []
        # shifts_actual = np.array(shifts[good_i]) - self.weighted_mean(shifts[good_i], shifts_unc[good_i]) # normalized phase shift
        for trial in tqdm.tqdm(trials):
            # get expected shifts
            this_slope = (trial * duration) / (self.sidereal_day * len(shifts))
            this_shifts_expected = np.arange(len(shifts)) * this_slope
            this_shifts_expected = this_shifts_expected[good_i] # apply mask

            # get residuals
            this_residuals = shifts_actual - this_shifts_expected
            this_offset = np.mean(this_residuals)
            this_residuals = this_residuals - this_offset # remove mean to avoid bias

            # get rms
            # this_rms = np.sum(weight * (shifts_actual - this_shifts_expected)**2) / np.sum(weight)
            this_rms = np.sqrt(np.sum(weight * (this_residuals)**2) / np.sum(weight))

            rms.append(this_rms)
            shifts_expected.append(this_shifts_expected + this_offset) # add offset back for diagnostic plot

        # find where rms is minimum
        best_i = np.where(rms == np.min(rms))[0][0]
        best_alias_factor = trials[best_i]
        best_expected_shifts = shifts_expected[best_i]

        # get expected shifts for nearby alias factors for diagnostic plot
        nearby_expected_shifts = []
        if best_alias_factor > af_min:
            nearby_expected_shifts.append(shifts_expected[best_i - 1])
        if best_alias_factor < af_max:
            nearby_expected_shifts.append(shifts_expected[best_i + 1])

        # plot diagnostic
        self.cf_plot_diagnostic(
            trials = trials,
            rms = rms,
            # shifts_x = np.arange(0, n_subints, bin_size),
            shifts_x = good_i,
            shifts_actual = shifts_actual,
            shifts_actual_unc = shifts_actual_unc,
            shifts_expected = best_expected_shifts,
            nearby_expected_shifts = nearby_expected_shifts,
            best_af = best_alias_factor,
            std_profile = std_profile,
            power1 = data_stacked[0],
            power2 = data_stacked[-1],
            data_stacked = data_stacked, 
            n_stacked = self.su.n_stacked
        )

        # print summary
        self.logger.success(
            pd.DataFrame({
                "Stacked S/N": [avg_snr],
                "Alias Factor": [best_alias_factor]
            })
        )

        self.alias_factor = best_alias_factor
        self.avg_snr = avg_snr

        return best_alias_factor

    def cf_plot_diagnostic(self, trials, rms, shifts_x, shifts_actual, shifts_actual_unc, shifts_expected, nearby_expected_shifts, best_af, std_profile, power1, power2, data_stacked, n_stacked):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8),  gridspec_kw={'width_ratios': [3, 1]})

        # plot rms
        axs[0, 0].plot(trials, rms, c="k", lw=1)
        axs[0, 0].axvline(best_af, label=f"Best AF = {best_af}", c="r", lw=1)
        axs[0, 0].set_title(f"RMS vs Alias Factor")
        axs[0, 0].set_xlabel("Alias Factor")
        axs[0, 0].set_ylabel("RMS")
        axs[0, 0].set_yscale("log")
        axs[0, 0].legend()

        # plot shifts
        axs[1, 0].errorbar(shifts_x, np.array(shifts_actual), shifts_actual_unc, c="k", lw=1, label="Actual", marker="x", capsize=2, fmt="x")
        axs[1, 0].errorbar(shifts_x, np.array(shifts_expected), c="r", lw=1, linestyle="-", label=f"Fitted Drift (AF = {best_af})")
        for i, nearby_shifts in enumerate(nearby_expected_shifts):
            axs[1, 0].errorbar(shifts_x, nearby_shifts, shifts_actual_unc, c="k", lw=1, linestyle="--", label=f"Fitted Drift +/- 1 AF" if i == 0 else None, alpha=0.5)
        axs[1, 0].set_title("Shifts")
        axs[1, 0].set_xlabel("Subints")
        axs[1, 0].set_ylabel("Shift (phase)")
        axs[1, 0].legend()

        # plot powers
        axs[2, 0].plot(self.normalize_power(std_profile), c="k", lw=1, label="Std Profile")
        axs[2, 0].plot(self.normalize_power(power1), c="r", lw=1, label="First Subint", alpha=0.75)
        axs[2, 0].plot(self.normalize_power(power2), c="b", lw=1, label="Last Subint", alpha=0.75)
        axs[2, 0].set_title(f"Powers")
        axs[2, 0].set_xlabel("Samples")
        axs[2, 0].set_ylabel("Power")
        axs[2, 0].legend()

        # normalize stacked data
        for i, _ in enumerate(data_stacked):
            data_stacked[i] = np.array(data_stacked[i]) - min(data_stacked[i])
            data_stacked[i] = np.array(data_stacked[i]) / max(data_stacked[i])

        # plot stacked profile
        axs[0, 1].plot(np.sum(data_stacked, axis=0), lw=0.5, c="k")
        axs[0, 1].set_title("Stacked Profile")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # plot stacked data
        axs[1, 1].remove()
        axs[2, 1].remove()
        n_params_gs = axs[0, 1].get_gridspec()
        axs_subints = fig.add_subplot(n_params_gs[1:3, 1])
        axs_subints.matshow(data_stacked, cmap="gray_r", aspect="auto")
        axs_subints.set_title("Subint Profiles")
        axs_subints.set_xlabel("Phase")
        axs_subints.set_ylabel("Sample")
        axs_subints.set_xticks(np.arange(0, len(data_stacked[0]) + 1, len(data_stacked[0]) / 2))
        axs_subints.set_xticklabels(["0", "0.5", "1"])
        axs_subints2 = axs_subints.twiny()
        axs_subints2.set_xlim(axs_subints.get_xlim())
        
        # plot info
        fig.text(0.001, 0.000, f"CHAMPSS Timing Pipeline alias_utils ({utils.get_version_hash()}) | {self.psrdir.split('/')[-1]} | {n_stacked} files | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")

        fig.tight_layout()
        plt.savefig(self.outdir + "/diagnostic.pdf")

    def dealias(self, save_pkl=False):
        if self.alias_factor is None:
            raise ValueError("Alias factor not available. Run tp_get_alias_factor or cf_get_alias_factor first.")

        self.logger.debug(f"Initializing dealias_utils...")
        du = dealias_utils(
            psrdir=self.psrdir,
            parfile=self.parfile,
            workspace=self.workspace,
            outdir=self.outdir,
            logger=self.logger.copy()
        )

        self.logger.debug(f"Dealiasing with alias_factor={self.alias_factor}...")
        du_res = du.dealias(self.alias_factor)

        if os.path.exists(f"{self.workspace}/diagnostic.pdf"):
            self.logger.info(f"Find alias diagnostic plot > {du.psrdir_dealias}")
            shutil.move(self.workspace + "/diagnostic.pdf", f"{du.psrdir_dealias}/diagnostic.pdf")

        # Generate summary
        if du_res:
            self.summary = {
                "psr_id": self.psrdir.split("/")[-1],
                "n_stacked": self.su.n_stacked, 
                "alias_factor": float(self.alias_factor), 
                "snr_stacked": float(self.avg_snr),
                "notes": {"remark": "DEALIAS_FITTING_OK"}
            }
        else:
            self.summary = {
                "psr_id": self.psrdir.split("/")[-1],
                "n_stacked": self.su.n_stacked, 
                "alias_factor": float(self.alias_factor), 
                "snr_stacked": float(self.avg_snr),
                "notes": {"remark": "DEALIAS_FITTING_FAILED"}
            }

        return self.summary

    def normalize_power(self, power):
        power -= np.mean(power)
        power /= np.max(power)

        return power

    def weighted_mean(self, vals, errs):
        weights = 1 / errs**2
        return np.sum(weights * vals) / np.sum(weights)

    def save_outfiles(self, filename="auto", pickle_only=False):
        # Get filename
        if filename == "auto":
            filename = self.psrdir + "/dealias"
            if not os.path.isdir(filename):
                if not os.path.exists(filename):
                    os.makedir(filename)
                    self.logger.success(f"Directory created: {filename}")
                else:
                    raise Exception(f"Directory already exists and is not writable: {filename}. Is there a file with the same name?")

        # Check if outdir exists
        if (not os.path.exists(self.outdir) or glob.glob(self.outdir) == []) and not pickle_only:
            raise Exception(f"Directory does not exist / empty: {self.outdir}. Was the alias_utils initialized or loaded from a pickle file?")

        # Save pickle
        with open(self.outdir + "/alias_utils.pkl", "wb") as f:
            pickle.dump(self, f)
            self.logger.success(f"Pickle saved to {self.outdir}/alias_utils.pkl")
            if pickle_only:
                return

        # Copy everything in outdir to filename
        if not os.path.isdir(filename):
            if os.path.exists(filename):
                raise Exception(f"Directory already exists and is not writable: {filename}. Is there a file with the same name?")
        else:
            # Remove the directory
            shutil.rmtree(filename)

        # Create the directory
        os.makedirs(filename)

        # Copy the contents of outdir to filename
        for f in glob.glob(self.outdir + "/*"):
            self.logger.debug(f"Copying {f} to {filename}", layer=1)
            if os.path.isdir(f):
                shutil.copytree(f, os.path.join(filename, os.path.basename(f)))
            else:
                shutil.copy2(f, filename)
        self.logger.success(f"Save outputs to {filename}")
        
        # self.cleanup_workspace = False

    def write_db(self, db_path="auto"):
        # Check if summary is available
        if not self.summary:
            raise ValueError("No summary available. Please run dealias() first.")

        # Get db_path
        if db_path == "auto":
            db_path = f"{self.psrdir}/champss_timing.sqlite3.db"

        # Write database
        with database(db_path) as db_hdl:
            db_hdl.insert_dealias_history(
                n_stacked=self.summary["n_stacked"], 
                alias_factor=self.summary["alias_factor"],
                snr_stacked=self.summary["snr_stacked"],
                notes=self.summary["notes"]
            )

        self.logger.success(f"Database updated: {db_path}")

    def cleanup(self):
        if self.cleanup_workspace:
            self.logger.debug("Cleaning up workspace")
            shutil.rmtree(self.workspace)
        else:
            self.logger.debug(f"Workspace will NOT be removed at {self.workspace} due to cleanup=False")

    def commit(self):
        '''
        Commit the changes to the psrdir and database
        '''

        # Commit changes to psrdir
        self.logger.debug(f"Committing changes to psrdir: {self.psrdir}")
        self.logger.level_up()
        self.save_outfiles()
        self.logger.level_down()

        # Commit changes to database
        self.logger.debug(f"Committing changes to database: {self.psrdir}")
        self.logger.level_up()
        self.write_db()
        self.logger.level_down()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.logger.error(f"Error while running find_alias. Workspace will NOT be removed at {self.workspace}")
            try:
                self.save(self.workspace + "/alias_utils.pkl")
                self.logger.error(f"Alias_utils object saved to {self.workspace}/alias_utils.pkl")
            except:
                self.logger.error(f"Error while saving alias_utils object to {self.workspace}/alias_utils.pkl")
            raise exc_type(exc_value).with_traceback(traceback)
        else:
            self.cleanup()