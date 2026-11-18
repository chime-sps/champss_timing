import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from astropy.time import Time
from ..tools.template_utils import StackTemplate
from ..tools.aliasing import generate_template_from_stacked_data
from ..utils.data_quality import data_quality_utils
from ..utils.correlation import fourier_shifts
from ..utils.logger import logger

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

K_DM = 1.0 / 2.41e-4 # use the DM constant consistent with PINT

def get_dm_optimizer_from_stackfile(stackfile, n_subs=None, n_freqs=None, **kwargs):
    from ..tools.stacked_data import StackedData
    
    # read stackfile
    stacked_data = StackedData.load(stackfile)

    # scrunch polarisation
    stacked_data.scrunch(axis=1)

    # downsample
    if n_subs is not None:
        stacked_data.downsample_to_length(n_subs, axis=0)
    if n_freqs is not None: 
        stacked_data.downsample_to_length(n_freqs, axis=2)

    return DMOptimizer(
        data=stacked_data.get_data(normalize=True)[:, 0, :, :], 
        freq_lo=stacked_data.meta.freq_lo, 
        freq_hi=stacked_data.meta.freq_hi, 
        folding_period=stacked_data.meta.period, 
        folding_dm=stacked_data.meta.dm, 
        **kwargs
    )

def compute_dispersive_delay(freq, ddm, ref_freq, k_dm):
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
    
    return k_dm * ddm * (freq ** -2 - ref_freq ** -2)

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
        
        self.data = np.asarray(data, dtype=np.float64) # ensure data is float64 for precision
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
        self.nsubs = self.data.shape[0]
        self.nfreqs = self.data.shape[1]
        self.nbins = self.data.shape[2]

        if self.nfreqs != len(self.freqs):
            raise ValueError("Number of frequency channels must match the second dimension of the waterfall data.")

        # Take step
        self.__take_step(ddm)

    def __calc_dt(self, freq, ddm): 
        return compute_dispersive_delay(freq, ddm, self.ref_freq, self.k_dm)

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

    def bootstrap(self, indices):
        """
        Bootstrap the data array by selecting a subset of indices along the first axis (subints).

        Parameters
        ----------
        indices : array-like
            Indices of the subints to select.

        Returns
        -------
        DMOptimizerState
            New DMOptimizerState instance with the bootstrapped data.
        """

        new_step = copy.deepcopy(self)
        new_step.data = new_step.data[indices, :, :] # select the specified subints
        new_step.__take_step(ddm=0) # take a fake step to update SNR and chi2

        return new_step

    def get_data(self):
        """
        Get the current data array.

        Returns
        -------
        np.ndarray
            Current data array.
        """
        return self.data

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
    def __init__(self, steps, initial_state, folding_dm, metric, logger):
        self.steps = steps
        self.initial_state = initial_state
        self.folding_dm = folding_dm
        self.metric = metric
        self.logger = logger

        # Sanity check on steps
        if not self.steps or len(self.steps) == 0:
            raise ValueError("No steps provided to DMOptimizerResults.")

        # Get the best state
        self.best_state = self.__get_best_state()

        # Get the best DM value
        self.value = self.best_state.get_dm()

        # Bootstrap to estimate DM uncertainty
        self.uncertainty = self.__bootstrap()

    def __add__(self, other):
        """Combine two DMOptimizerResults objects."""
        
        if not isinstance(other, DMOptimizerResults):
            raise ValueError("Can only combine with another DMOptimizerResult object.")

        return DMOptimizerResults(
            steps=self.steps + other.steps,
            initial_state=self.initial_state,
            folding_dm=self.folding_dm,
            metric=self.metric,
            logger=self.logger
        )

    def __get_best_state(self):
        """Helper function to get the best state based on the metric."""
        
        # Get metric values
        metric_vals = self.get_metric_vals()

        # Reveal the best index
        best_idx = np.argmax(metric_vals)

        # Refine around the best index
        best_dm, correction = self.__refine_best_dm(self.get_dm_vals(), metric_vals, best_idx)
        self.logger.debug(f"Refined best DM: {best_dm}, correction: {correction}")

        # Create a new state at the refined DM
        best_state = self.steps[best_idx].take_step(ddm=correction)
        self.logger.debug(f"Created best state at DM: {best_state.get_dm()}")

        # Check if the metric value improved after refinement
        metric_val_before = getattr(self.steps[best_idx], self.metric)
        metric_val_after = getattr(best_state, self.metric)
        if metric_val_after < metric_val_before:
            self.logger.warning(f"Metric ({self.metric}) decreased from {metric_val_before} to {metric_val_after} after peak refinement.")

        return best_state

    def __refine_best_dm(self, dm_vals, metric_vals, i_best):
        """Helper function to define the best DM based on the metric."""

        correction = 0
        if i_best != 0 and i_best != len(dm_vals) - 1:
            # Find nearby points
            i_nearby = slice(i_best - 1, i_best + 2)

            # Fit a parabola to the nearby points (i.e., interpolate)
            c2, c1, _ = np.polyfit(dm_vals[i_nearby] - dm_vals[i_best], metric_vals[i_nearby], 2)
            
            # Apply the correction to dm_best
            correction = - c1 / (2 * c2)

        return dm_vals[i_best] + correction, correction

    def __bootstrap(self, n_boot=64, seed=0):
        """DM uncertainty from bootstrapping over subintegrations."""

        # Initialize random seed
        rng = np.random.default_rng(seed)

        if self.steps[0].nsubs < 8:
            self.logger.warning("Number of subintegrations is less than 8, bootstrapping may be unreliable.")

        best_dm_vals = []
        for _ in tqdm.tqdm(
            range(n_boot), 
            bar_format="Bootstrapping: {percentage:.1f}% ({elapsed}<{remaining})",
            leave=False
        ):
            metric_vals = []
            dm_vals = np.array([], dtype=float)

            # Generate random indices for subints
            idx = rng.integers(0, self.steps[0].nsubs, self.steps[0].nsubs)

            for step in self.steps:
                # Bootstrap
                step_bs = step.bootstrap(indices=idx)

                if self.metric == "snr":
                    metric_vals.append(step_bs.snr)
                else:
                    metric_vals.append(step_bs.chi2)

                dm_vals = np.append(dm_vals, step_bs.get_dm())

            # Find the best dm index
            i_best = np.argmax(metric_vals)

            # Refine around i_best
            dm_best, _ = self.__refine_best_dm(dm_vals, metric_vals, i_best)

            best_dm_vals.append(dm_best)

        return np.std(best_dm_vals)

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

    def get_data(self):
        """
        Get the list of data arrays corresponding to the stored steps.

        Returns
        -------
        list of ndarray
            List of data arrays.
        """
        return self.best_state.get_data()

    def as_ufloat(self):
        """
        Get the best DM value along with its uncertainty.

        Returns
        -------
        uncertainties.ufloat
            Best DM value with its uncertainty.
        """
        return ufloat(self.value, self.uncertainty)

    def plot(self, savefig=None):
        """
        Plot the SNR values as a function of DM steps.

        Returns
        -------
        None
        """


        def get_phase_of_peak_amplitude(profile):
            return np.argmax(profile) / len(profile)

        fig = plt.figure(figsize=(8, 10))
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
            cmap='gray_r', 
            rasterized=True
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
            cmap='gray_r', 
            rasterized=True
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
        snr_dms.axvline(self.folding_dm, color='k', linestyle=':', lw=1, label="Folding DM")
        snr_dms.axvline(self.best_state.get_dm(), color='r', linestyle='-', lw=1, label="Best DM")
        snr_dms.axvspan(
            self.value - self.uncertainty, 
            self.value + self.uncertainty, 
            color='r', 
            alpha=0.3
        )
        snr_dms.legend()
        snr_dms.set_xlabel("DM")
        snr_dms.set_ylabel(self.metric.upper())

        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)


class DMOptimizer:
    def __init__(
            self, 
            data, 
            freq_lo, 
            freq_hi, 
            folding_period, 
            folding_dm,
            ref_freq=None, 
            metric='chi2',  
            ncpus=1, 
            logger=logger()
        ):

        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.data = data
        self.folding_period = folding_period
        self.folding_dm = folding_dm
        self.metric = metric
        self.logger = logger
        self.ncpus = ncpus

        # Set the DM constant
        self.k_dm = K_DM

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

    def optimize(self, dm_range=[-1, 1], tol=1e-3):
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
        ddm_values = np.arange(np.ceil(dm_range[0] / step_size), np.floor(dm_range[1] / step_size) + 1) * step_size
        self.logger.debug(f"Taking {len(ddm_values)} DM trials")

        # take steps
        trial_steps = self.__take_trials(ddm_values)

        return DMOptimizerResults(trial_steps, self.initial_state, self.folding_dm, metric=self.metric, logger=self.logger)

