import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from ..utils.data_quality import data_quality_utils
from ..utils.correlation import fourier_shifts
from ..utils.logger import logger

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

def snr(profile, widths=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
        off_pulse=None):
    """
    Boxcar matched-filter S/N (circular).

    off_pulse : slice, index array, or boolean mask selecting off-pulse
        bins for baseline/noise estimation. If None, uses a robust
        median/MAD estimate over the whole profile (fine if duty cycle
        is small).

    Returns (best_snr, best_width, best_start_bin).
    """
    p = np.asarray(profile, dtype=np.float64)
    if off_pulse is not None:
        ref = p[off_pulse]
        mu, sig = ref.mean(), ref.std(ddof=1)
    else:
        mu = np.median(p)
        sig = 1.4826 * np.median(np.abs(p - mu))
    if not np.isfinite(sig) or sig == 0:
        return 0.0, 0, 0
    z = (p - mu) / sig

    n = z.size
    cs = np.concatenate(([0.0], np.cumsum(np.concatenate((z, z)))))
    best = (-np.inf, 0, 0)
    for w in widths:
        if w >= n:
            continue
        sums = cs[w:w + n] - cs[:n]          # circular boxcar sums
        i = int(np.argmax(sums))
        s = sums[i] / np.sqrt(w)
        if s > best[0]:
            best = (s, w, i)
    return best

class DMOptimizerState:
    def __init__(self, data, folding_period, freqs, ref_freq, folding_dm, k_dm, ddm=0):
        """
        Initialize a DMOptimizerState instance.

        Parameters
        ----------
        data : np.ndarray
            Data array with shape (n_subs, n_freqs, n_bins).
        folding_period : float
            Folding period of the pulsar.
        freqs : np.ndarray
            Array of frequency channels.
        ref_freq : float
            Reference frequency for DM correction.
        folding_dm : float
            Folding DM value for the pulsar.
        ddm : float, optional
            Initial DM step to apply, by default 0.
        """
        
        self.data = data
        self.folding_period = folding_period
        self.freqs = freqs
        self.ref_freq = ref_freq
        self.folding_dm = folding_dm
        self.k_dm = k_dm

        if self.data.ndim != 3:
            raise ValueError("Data array must have three dimensions (n_subs, n_freqs, n_bins).")

        self.ddm = 0
        self.snr = None
        self.chi2 = None
        self.nfreqs = self.data.shape[1]
        self.nbins = self.data.shape[2]

        if self.nfreqs != len(self.freqs):
            raise ValueError("Number of frequency channels must match the second dimension of the waterfall data.")

        # Take step
        self.__take_step(ddm)

    def __calc_dt(self, freq, ddm): 
        """
        Calculate the time delay for a given frequency and DM step.

        Parameters
        ----------
        freq : float
            Frequency of the channel.
        ddm : float
            DM step.

        Returns
        -------
        float
            Time delay for the given frequency and DM step.

        Returns
        -------
        float
            Time delay for the given frequency and DM step.
        """
        
        return self.k_dm * ddm * (freq ** -2 - self.ref_freq ** -2)

    def __take_step(self, ddm):
        """
        Apply a DM step to the data array in the current step.

        Parameters
        ----------
        ddm : float
            DM step to apply.
        """

        # Apply the dDM steps
        if ddm != 0:
            for i, freq in enumerate(self.freqs):
                self.data[:, i, :] = fourier_shifts.roll(
                    self.data[:, i, :], 
                    -self.__calc_dt(freq, ddm) / self.folding_period * self.nbins
                )
                
            self.ddm += ddm

        # Compute SNR and chi2
        profile = self.get_pulse_profile()
        self.snr = data_quality_utils.boxcar_snr(profile)
        # self.snr = snr(profile)[0]
        self.chi2 = data_quality_utils.chi2(profile)

    def take_step(self, ddm):
        """
        Apply a DM step to the data array and return an updated step. 

        Parameters
        ----------
        ddm : float
            DM step to apply.

        Returns
        -------
        DMOptimizerState
            Updated DMOptimizerState instance after applying the DM step.
        """

        new_step = copy.deepcopy(self)
        new_step.__take_step(ddm)

        return new_step

    def get_waterfall(self):
        """
        Get the current waterfall data.

        Returns
        -------
        np.ndarray
            Waterfall data array with shape (n_freqs, n_bins).
        """
        return self.data.sum(axis=0)

    def get_pulse_profile(self):
        """
        Get the current pulse profile by averaging the waterfall data over frequency.

        Returns
        -------
        np.ndarray
            Pulse profile array with shape (n_bins,).
        """
        return self.data.sum(axis=0).sum(axis=0)

    def get_ddm(self):
        """
        Get the current DM step.

        Returns
        -------
        float
            Current DM step.
        """
        return self.ddm

    def get_dm(self):
        """
        Get the current DM value.

        Returns
        -------
        float
            Current DM value (folding DM + dDM).
        """
        return self.folding_dm + self.ddm

    def get_snr(self):
        """
        Get the signal-to-noise ratio (SNR) of the current waterfall data.

        Returns
        -------
        float
            SNR of the current waterfall data.
        """
        return self.snr

    def plot_waterfall(self, ax=None):
        """
        Plot the waterfall data.

        Returns
        -------
        None
        """

        this_ax = ax
        if this_ax is None:
            this_ax = plt.gca()
        
        this_ax.pcolormesh(np.linspace(0, 1, self.nbins), self.freqs, self.get_waterfall(), shading='auto', cmap='gray_r')
        this_ax.text(0.05, 0.95, f"SNR = {self.get_snr():.2f}", transform=this_ax.transAxes, verticalalignment='top', c="k")
        this_ax.set_ylabel("Frequency (MHz)")
        this_ax.set_xlabel("Phase")
        this_ax.set_title(f"dDM = {round(self.ddm, 2)}")

        if ax is None:
            plt.show()

class DMOptimizerResults:
    def __init__(self, steps, initial_state, folding_dm, metric):
        self.steps = steps
        self.initial_state = initial_state
        self.folding_dm = folding_dm
        self.metric = metric

        # Sanity check on steps
        if not self.steps or len(self.steps) == 0:
            raise ValueError("No steps provided to DMOptimizerResults.")

        # Get the best state
        self.best_state = self.__get_best_state()

    def __add__(self, other):
        """Combine two DMOptimizerResults objects."""
        
        if not isinstance(other, DMOptimizerResults):
            raise ValueError("Can only combine with another DMOptimizerResult object.")

        return DMOptimizerResults(
            steps=self.steps + other.steps,
            initial_state=self.initial_state,
            folding_dm=self.folding_dm,
            metric=self.metric
        )

    def __get_best_state(self):
        """Helper function to get the best state based on the metric."""
        
        # Get metric values
        metric_vals = self.get_metric_vals()

        # Reveal the best idnex
        best_idx = np.argmax(metric_vals)

        return self.steps[best_idx]

    def get_dm_vals(self):
        """
        Get the list of DM steps corresponding to the stored steps.

        Returns
        -------
        list of float
            List of DM steps.
        """
        return [self.folding_dm + step.get_ddm() for step in self.steps]

    def get_snr_vals(self):
        """
        Get the list of SNR values corresponding to the stored steps.

        Returns
        -------
        list of float
            List of SNR values.
        """
        return [step.get_snr() for step in self.steps]

    def get_chi2_vals(self):
        """
        Get the list of chi-squared values corresponding to the stored steps.

        Returns
        -------
        list of float
            List of chi-squared values.
        """
        return [step.chi2 for step in self.steps]

    def get_metric_vals(self):
        """
        Get the list of metric values corresponding to the stored steps.

        Returns
        -------
        list of float
            List of metric values.
        """
        if self.metric == 'snr':
            return self.get_snr_vals()
        elif self.metric == 'chi2':
            return self.get_chi2_vals()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def plot(self):
        """
        Plot the SNR values as a function of DM steps.

        Returns
        -------
        None
        """


        def get_phase_of_peak_amplitude(profile):
            return np.argmax(profile) / len(profile)

        fig = plt.figure(figsize=(6, 8))
        outer = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.19, wspace=0.05)

        # Left: initial state
        left = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

        init_profile = fig.add_subplot(left[0])
        init_profile.plot(
            np.linspace(0, 1, self.initial_state.nbins),
            self.initial_state.get_pulse_profile(), 
            c='k', 
            lw=1
        )
        init_profile.text(0.98, 0.98, f"DM: {self.folding_dm:.2f}\nSNR: {self.initial_state.get_snr():.2f}", ha='right', va='top', transform=init_profile.transAxes)
        init_profile.set_xlim(0, 1)
        init_profile.set_xticks([])
        init_profile.set_yticks([])
        for spine in init_profile.spines.values():
            spine.set_visible(False)
        init_profile.set_title("Initial State")

        init_waterfall = fig.add_subplot(left[1])
        init_waterfall.pcolormesh(
            np.linspace(0, 1, self.initial_state.nbins), 
            self.initial_state.freqs, 
            self.initial_state.get_waterfall(),
            cmap='gray_r'
        )
        init_waterfall.axvline(
            get_phase_of_peak_amplitude(self.initial_state.get_pulse_profile()), 
            color='r', 
            linestyle=':', 
            lw=1
        )
        init_waterfall.set_xlabel("Phase")
        init_waterfall.set_ylabel("Frequency (MHz)")
        init_waterfall.set_xticks([0, 0.25, 0.5, 0.75])

        # Right: final state
        right = outer[0, 1].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

        best_profile = fig.add_subplot(right[0])
        best_profile.plot(
            np.linspace(0, 1, self.best_state.nbins),
            self.best_state.get_pulse_profile(), 
            c='k', 
            lw=1
        )
        best_profile.set_xlim(0, 1)
        best_profile.set_xticks([])
        best_profile.set_yticks([])
        for spine in best_profile.spines.values():
            spine.set_visible(False)
        best_profile.text(0.98, 0.98, f"DM: {self.folding_dm + self.best_state.get_ddm():.2f}\nSNR: {self.best_state.get_snr():.2f}", ha='right', va='top', transform=best_profile.transAxes)
        best_profile.set_title("Best State")

        best_waterfall = fig.add_subplot(right[1])
        best_waterfall.pcolormesh(
            np.linspace(0, 1, self.best_state.nbins), 
            self.best_state.freqs, 
            self.best_state.get_waterfall(),
            cmap='gray_r'
        )
        best_waterfall.axvline(
            get_phase_of_peak_amplitude(self.best_state.get_pulse_profile()), 
            color='r', 
            linestyle=':', 
            lw=1
        )
        best_waterfall.set_xlabel("Phase")
        best_waterfall.set_xticks([0, 0.25, 0.5, 0.75, 1])
        best_waterfall.set_yticks([])

        # Bottom: SNR vs DMs
        snr_dms = fig.add_subplot(outer[1, :])
        snr_dms.plot(
            self.get_dm_vals(),
            self.get_metric_vals(),
            color='k',
            linestyle='-',
            marker=',',
            lw=0.1
        )
        snr_dms.axvline(self.folding_dm, color='k', linestyle=':', lw=1)
        snr_dms.axvline(self.best_state.get_dm(), color='r', linestyle='-', lw=1)
        snr_dms.set_xlabel("DM")
        snr_dms.set_ylabel(self.metric.upper())

class DMOptimizer:
    def __init__(
            self, 
            data, 
            freq_lo, 
            freq_hi, 
            folding_period, 
            folding_dm,
            ref_freq=None, 
            metric='snr', 
            k_dm=1.0 / 2.41e-4, 
            ncpus=1, 
            logger=logger()
        ):

        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.data = data
        self.folding_period = folding_period
        self.folding_dm = folding_dm
        self.k_dm = k_dm
        self.metric = metric
        self.logger = logger
        self.ncpus = ncpus

        # Generate linearly spaced frequency array for the waterfall data
        self.freqs = np.linspace(freq_lo, freq_hi, data.shape[1])

        # Default reference frequency is the midpoint
        self.ref_freq = ref_freq
        if self.ref_freq is None:
            self.ref_freq = (freq_lo + freq_hi) / 2

        # Number of bins and frequencies in data array
        self.nbins = data.shape[2]
        self.nfreqs = len(self.freqs)
        self.ddm = 0

        # Initialize DM optimizer step
        self.initial_state = DMOptimizerState(
            data=self.data, 
            folding_period=self.folding_period,
            freqs=self.freqs,
            ref_freq=self.ref_freq,
            folding_dm=self.folding_dm, 
            k_dm=self.k_dm
        )

    def __calc_step_size(self, tol):
        """
        Calculate the DM step size based on the tolerance of error in phase across the entire band.

        Parameters
        ----------
        tol : float, optional
            Tolerance  in phase across the entire band.

        Returns
        -------
        float
            Calculated DM step size.
        """
        return np.abs(tol * self.folding_period / self.k_dm / (
            self.freq_lo ** -2 - self.freq_hi ** -2
        ))

    def __take_trials(self, ddm_values):
        """
        Take multiple DM trials.

        Parameters
        ----------
        ddm_values : list of float
            List of DM steps to apply.

        Returns
        -------
        list of DMOptimizerState
            List of updated DMOptimizerState instances after applying each DM step.
        """

        with Pool(self.ncpus) as pool:
            trial_steps = list(
                tqdm.tqdm(
                    pool.imap(
                        self.initial_state.take_step, 
                        ddm_values
                    ), 
                    total=len(ddm_values), 
                    bar_format="Taking {n}/{total} DM trials: {percentage:.1f}% ({elapsed}<{remaining})", 
                    leave=False
                )
            )

        return trial_steps

    def optimize(self, dm_range=[-2, 2], tol=1e-3):
        """
        Optimize the DM by trying multiple DM steps within the specified range.

        Parameters
        ----------
        dm_range : list of float, optional
            Range of DM steps to try.
        tol : float, optional
            Tolerance in phase across the entire band.

        Returns
        -------
        DMOptimizerResults
            The DMOptimizerResults instance containing all trial steps and the best DM step applied.
        """

        # Compute step size
        step_size = self.__calc_step_size(tol)
        self.logger.debug(f"Step size: {step_size}")

        # Generate DM trial values
        ddm_values = np.arange(dm_range[0], dm_range[1], step_size)
        self.logger.debug(f"Taking {len(ddm_values)} DM trials")

        # take steps
        trial_steps = self.__take_trials(ddm_values)

        return DMOptimizerResults(trial_steps, self.initial_state, self.folding_dm, metric=self.metric)