import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from astropy.time import Time
from ..tools.template_utils import StackTemplate
from ..tools.aliasing import generate_template_from_stacked_data
from ..tools.stacked_data import K_DM
from ..utils.data_quality import data_quality_utils
from ..utils.correlation import fourier_shifts
from ..utils.logger import logger

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

def get_dm_optimizer_from_stackfile(stackfile, n_subs=None, n_freqs=None, inj_ddm=None, **kwargs):
    """
    Get a DMOptimizer instance from a stackfile.

    Parameters
    ----------
    stackfile : str
        Path to the stackfile.
    n_subs : int, optional
        Number of subintegrations to downsample to, by default None.
    n_freqs : int, optional
        Number of frequency channels to downsample to, by default None.
    inj_ddm : float, optional
        Injected DM offset to apply (for testing purposes), by default None.
    **kwargs : dict
        Additional keyword arguments to pass to the DMOptimizer.

    Returns
    -------
    DMOptimizer
        DMOptimizer instance initialized with the stacked data.
    """
    
    from ..tools.stacked_data import StackedData
    
    # read stackfile
    stacked_data = StackedData.load(stackfile)

    # scrunch polarisation
    stacked_data.scrunch(axis=1)

    # Dedisperse if an injected DM is provided
    if inj_ddm is not None:
        print(f"!!!Folding DM before injection: {stacked_data.meta.dm}!!!")
        print(f"!!!Injecting DM offset: {inj_ddm}!!!")
        stacked_data.refine_dm(ddm=inj_ddm)
        print(f"!!!Folding DM after injection: {stacked_data.meta.dm}!!!")

    # downsample
    if n_subs is not None:
        stacked_data.downsample_to_length(n_subs, axis=0)
    if n_freqs is not None: 
        stacked_data.downsample_to_length(n_freqs, axis=2)

    return DMOptimizer(
        stacked_data=stacked_data, 
        **kwargs
    )

class DMOptimizerResults:
    def __init__(self, steps, stacked_data, metric, logger):
        self.steps = steps
        self.stacked_data = stacked_data
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
        self.uncertainty = self.__get_uncertainty()

    #########################################
    # Measuring DM and its uncertainty
    #########################################

    def get_dm_vals(self):
        """
        Get the list of DM steps corresponding to the stored steps.

        Returns
        -------
        list of float
            List of DM steps.
        """
        return [step.dm for step in self.steps]

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
        # best_state = self.steps[best_idx].take_step(ddm=correction)
        best_state = self.stacked_data.copy()
        best_state.dedisperse(dm=best_dm)
        self.logger.debug(f"Created best state at DM: {best_state.get_dm()}")

        # Create a state with old DM for comparison
        prev_state = self.stacked_data.copy()
        prev_state.dedisperse(dm=self.get_dm_vals()[best_idx])

        # Get metric values before and after refinement
        metric_val_before = prev_state.get_stacked_metric(metric=self.metric)
        metric_val_after = best_state.get_stacked_metric(metric=self.metric)

        # Check if the metric value improved after refinement
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

    def __get_uncertainty(self):
        """Measure uncertainty from bootstraping"""

        # Unpack metric values from the steps
        bs_metric_vals = np.zeros((len(self.steps), self.steps[0].n_bootstrap))
        for i, step in enumerate(self.steps):
            bs_metric_vals[i, :] = getattr(step, f"bootstrap_{self.metric}")

        # Measure DM values for each bootstrap trial
        best_dm_vals = []
        trial_dm_vals = self.get_dm_vals()
        for j in range(self.steps[0].n_bootstrap):
            # Get metric values
            this_metric_vals = bs_metric_vals[:, j]

            # Find the best index
            i_best = np.argmax(this_metric_vals)

            # Refine the best DM for this bootstrap trial
            best_dm, _ = self.__refine_best_dm(trial_dm_vals, this_metric_vals, i_best)

            best_dm_vals.append(best_dm)

        return np.std(best_dm_vals) # measure uncertainty from the scatter of the best DM values across bootstrap trials


    #########################################
    # Helper functions for metric values
    #########################################

    def get_snr_vals(self):
        """
        Get the list of SNR values corresponding to the stored steps.

        Returns
        -------
        list of float
            List of SNR values.
        """
        return [step.snr for step in self.steps]

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

    #########################################
    # Other helper functions
    #########################################

    def get_data(self, *args, **kwargs):
        """
        Get the list of data arrays corresponding to the stored steps.

        Returns
        -------
        list of ndarray
            List of data arrays.
        """
        return self.best_state.get_data(*args, **kwargs)

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
            np.linspace(0, 1, self.stacked_data.data.shape[3]),
            self.stacked_data.get_profile(), 
            c='k', 
            lw=1
        )
        init_profile.text(0.98, 0.98, f"DM: {self.stacked_data.meta.dm:.2f}\nSNR: {self.stacked_data.get_stacked_snr():.2f}", ha='right', va='top', transform=init_profile.transAxes)
        init_profile.set_xlim(0, 1)
        init_profile.set_xticks([])
        init_profile.set_yticks([])
        for spine in init_profile.spines.values():
            spine.set_visible(False)
        init_profile.set_title("Initial State")

        init_waterfall = fig.add_subplot(left[1])
        init_waterfall.pcolormesh(
            np.linspace(0, 1, self.stacked_data.data.shape[3]), 
            self.stacked_data.freqs, 
            self.stacked_data.get_waterfall(normalize=True),
            cmap='gray_r', 
            rasterized=True
        )
        init_waterfall.axvline(
            get_phase_of_peak_amplitude(self.stacked_data.get_profile()), 
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
            np.linspace(0, 1, self.best_state.data.shape[3]),
            self.best_state.get_profile(), 
            c='k', 
            lw=1
        )
        best_profile.set_xlim(0, 1)
        best_profile.set_xticks([])
        best_profile.set_yticks([])
        for spine in best_profile.spines.values():
            spine.set_visible(False)
        best_profile.text(0.98, 0.98, f"DM: {self.best_state.meta.dm:.2f}\nSNR: {self.best_state.get_stacked_snr():.2f}", ha='right', va='top', transform=best_profile.transAxes)
        best_profile.set_title("Best State")

        best_waterfall = fig.add_subplot(right[1])
        best_waterfall.pcolormesh(
            np.linspace(0, 1, self.best_state.data.shape[3]), 
            self.best_state.freqs, 
            self.best_state.get_waterfall(normalize=True),
            cmap='gray_r', 
            rasterized=True
        )
        best_waterfall.axvline(
            get_phase_of_peak_amplitude(self.best_state.get_profile()), 
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
        snr_dms.axvline(self.stacked_data.get_dm(), color='k', linestyle=':', lw=1, label="Folding DM")
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

class DMOptimizerStep:
    def __init__(self, dm, n_bootstrap):
        self.dm = dm
        self.n_bootstrap = n_bootstrap

        # Initialize data
        self.chi2 = None
        self.snr = None

        # Initialize bootstrap data
        self.bootstrap_chi2 = []
        self.bootstrap_snr = []
        self.bootstrap_indices = []

    def add_trial(self, chi2, snr, subint_indices, is_bootstrap):
        if is_bootstrap:
            self.bootstrap_chi2.append(chi2)
            self.bootstrap_snr.append(snr)
            self.bootstrap_indices.append(subint_indices)
        else:
            self.chi2 = chi2
            self.snr = snr

    def validate(self):
        if self.chi2 is None or self.snr is None:
            raise ValueError("Non-bootstrap trial data is not set.")
        
        if not self.bootstrap_chi2 or not self.bootstrap_snr or not self.bootstrap_indices:
            raise ValueError("Bootstrap trial data is not set.")
        
        if len(self.bootstrap_chi2) != len(self.bootstrap_snr) or len(self.bootstrap_chi2) != len(self.bootstrap_indices):
            raise ValueError("Bootstrap trial data lengths are inconsistent.")

        if len(self.bootstrap_chi2) != self.n_bootstrap:
            raise ValueError("Number of bootstrap trials does not match n_bootstrap.")

class DMOptimizer:
    def __init__(
            self, 
            stacked_data, 
            metric='chi2',  
            ncpus=1, 
            logger=logger()
        ):
        
        self.metric = metric
        self.logger = logger
        self.ncpus = ncpus
        self.stacked_data = stacked_data

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
        return np.abs(tol * self.stacked_data.meta.period / K_DM / (
            self.stacked_data.meta.freq_lo ** -2 - self.stacked_data.meta.freq_hi ** -2
        ))

    def __take_step(self, ddm):
        # Unpack payload
        ddm, bootstrap_idx = ddm

        # Make a copy of the stacked data for this trial
        state = self.stacked_data.copy()

        # Apply the DM step
        state.refine_dm(ddm=ddm)

        # Scrunch frequency and pol axes to reduce data size
        state.scrunch(axis=1)
        state.scrunch(axis=2) # we don't need frequency for bootstrapping
        
        # Initialize step instance
        step = DMOptimizerStep(state.get_dm(), n_bootstrap=len(bootstrap_idx)-1)

        # Bootstrap over the selected indices
        for i, bs_idx in enumerate(bootstrap_idx):
            # Make a copy of the state
            this_state = state.copy()

            # Select the desired subints
            this_state.data = this_state.data[bs_idx]

            step.add_trial(
                chi2=this_state.get_stacked_chi2(), 
                snr=this_state.get_stacked_snr(), 
                subint_indices=bs_idx, 
                is_bootstrap=(i != 0) # the first trial is not a bootstrap (see __take_trials)
            )

        # Validate the step
        step.validate()

        return step

    def __take_trials(self, ddm_values, n_bootstrap=64, seed=0):
        """
        Take multiple DM trials.

        Parameters
        ----------
        ddm_values : list of float
            List of DM steps to apply.

        Returns
        -------
        list of DMOptimizerStep
            List of updated DMOptimizerStep instances after applying each DM step.
        """

        # Generate bootstrap indices for the DM trials
        bootstrap_indices = np.random.default_rng(seed).integers(
            0, 
            self.stacked_data.data.shape[0], # number of subints
            self.stacked_data.data.shape[0] * n_bootstrap
        )

        # Add the original indices at the front, so the first trial always uses the original data
        bootstrap_indices = np.concatenate([
            np.arange(self.stacked_data.data.shape[0]), 
            bootstrap_indices
        ])

        # Reshape the bootstrap indices into a 2D array where each row corresponds to a trial
        bootstrap_indices = bootstrap_indices.reshape(n_bootstrap + 1, self.stacked_data.data.shape[0])

        # Prepare payload for multiprocessing
        payload = [(this_ddm, bootstrap_indices) for this_ddm in ddm_values]
        
        # Run trials
        with Pool(self.ncpus) as pool:
            steps = list(
                tqdm.tqdm(
                    pool.imap(
                        self.__take_step, 
                        payload
                    ), 
                    total=len(ddm_values), 
                    bar_format="Taking {n}/{total} DM trials: {percentage:.1f}% ({elapsed}<{remaining})", 
                    leave=False
                )
            )

        return steps

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
        steps = self.__take_trials(ddm_values)

        return DMOptimizerResults(steps, self.stacked_data, metric=self.metric, logger=self.logger)

