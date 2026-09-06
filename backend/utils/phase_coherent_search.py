from .correlation import fourier_shifts

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy import stats, special
from scipy.ndimage import uniform_filter1d
from multiprocessing import Pool

class PulseProfilesState:
    def __init__(self, shifted_profiles, epochs, df0, df1, center_epoch):
        self.shifted_profiles = shifted_profiles
        self.epochs = epochs
        self.df0 = df0
        self.df1 = df1
        self.center_epoch = center_epoch
        self.grid_search_results = None

    def __normalize(self, profile):
        """Helper function to normalize to zero mean, unit variance"""
        return (profile - np.median(profile)) / np.std(profile)

    def __boxcar_snr(self, profile, max_width=None):
        """Helper function to compute the boxcar SNR of a profile"""
        x = np.asarray(profile, dtype=float)
        n = x.size
        baseline = np.median(x)
        sigma = np.std(np.sort(x)[: 3 * n // 4])
        widths = 2 ** np.arange(int(np.log2((max_width or n // 4))) + 1)
        return max(
            (uniform_filter1d(x, w, mode="wrap").max() - baseline) / (sigma / np.sqrt(w))
            for w in widths
        )

    def get_profiles(self):
        return self.shifted_profiles

    def get_stacked_profile(self, normalize=False):
        """
        Get the stacked profile after applying phase shifts.

        Returns
        -------
        stacked_profile : ndarray
            The stacked (and optionally normalized) profile.
        """

        stacked_profile = np.sum(self.shifted_profiles, axis=0)

        if normalize:
            stacked_profile = self.__normalize(stacked_profile)

        return stacked_profile

    def get_stacked_chisquare(self):
        """
        Get the chi-square of the stacked profile after applying phase shifts.

        Returns
        -------
        chisquare : float
            The chi-square of the stacked profile.
        """

        # Get stacked profile
        stacked_profile = self.get_stacked_profile()

        return stats.chisquare(stacked_profile)[0]

    def get_stacked_snr(self):
        """
        Get the signal-to-noise ratio of the stacked profile after applying phase shifts.

        Returns
        -------
        snr : float
            The signal-to-noise ratio of the stacked profile.
        """

        # Get stacked profile
        stacked_profile = self.get_stacked_profile()

        return self.__boxcar_snr(stacked_profile)

    def plot(self, savefig=None):
        if self.grid_search_results is None:
            _, ax = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={'height_ratios': [1, 2.5], "hspace": 0}, squeeze=False)
        else:
            _, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 2.5], "hspace": 0}, squeeze=False)

        # Stacked Profiles
        ax[0, 0].plot(np.tile(np.sum(self.shifted_profiles, axis=0), 2), lw=0.5, c="k")
        ax[0, 0].set_xlim(0, len(self.shifted_profiles[0]) * 2)
        ax[0, 0].set_title("Stacked Profile")
        ax[0, 0].set_ylabel("Amplitude")
        ax[0, 0].axis('off')

        # Individual Shifted Profiles
        ax[1, 0].pcolormesh(
            np.arange(len(self.shifted_profiles[0]) * 2), 
            self.epochs, 
            np.tile(self.shifted_profiles, (1, 2)), 
            shading='auto', 
            cmap='gray_r'
        )
        ax[1, 0].set_xlabel("Bin")
        ax[1, 0].set_ylabel("Epoch")

        if self.grid_search_results is not None:
            # Plot search results]
            ax[0, 1].text(
                0, 1, 
                f"Best dF0: \n {self.df0}\nBest dF1: \n {self.df1}\nBest SNR: \n {self.get_stacked_snr()}\nPEPOCH: \n {self.center_epoch}\nNumber of Observations: \n {len(self.shifted_profiles)}", 
                ha='left', 
                va='top', 
                transform=ax[0, 1].transAxes
            )
            ax[0, 1].axis('off')
            
            # SNR grid values
            ax[1, 1].pcolormesh(
                self.grid_search_results["df1_vals"],
                self.grid_search_results["df0_vals"],
                self.grid_search_results["snrs"].reshape(
                    len(self.grid_search_results["df0_vals"]), 
                    len(self.grid_search_results["df1_vals"])
                ),
                shading='auto',
                cmap='gray_r'
            )
            ax[1, 1].plot(self.df1, self.df0, 'rx')
            ax[1, 1].axhline(self.df0, color='r', linestyle='--', lw=0.5, alpha=0.25)
            ax[1, 1].axvline(self.df1, color='r', linestyle='--', lw=0.5, alpha=0.25)
            ax[1, 1].set_xlabel("df1 trials")
            ax[1, 1].set_ylabel("df0 trials")

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()

class PulseProfiles:
    def __init__(self, profiles, epochs):
        self.profiles = profiles
        self.epochs = epochs

        if len(self.profiles) == 0 or len(self.epochs) == 0:
            raise ValueError("Profiles and epochs must not be empty.")

        if len(self.profiles) != len(self.epochs):
            raise ValueError("The number of profiles must match the number of epochs.")

        # Sort archives by epoch
        sorted_indices = np.argsort(self.epochs)
        self.profiles = [self.profiles[i] for i in sorted_indices]
        self.epochs = [self.epochs[i] for i in sorted_indices]

        # Interpolate all profiles to have the same number of bins
        max_nbins = np.max([len(profile) for profile in self.profiles])
        for i in range(len(self.profiles)):
            if len(self.profiles[i]) != max_nbins:
                self.profiles[i] = self.__interpolate(self.profiles[i], max_nbins)
    
    def __interpolate(self, data, size):
        if len(data) == size:
            return data

        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, size, endpoint=False)
        data_interp = np.interp(x_new, x_old, data)
        return data_interp

    def get_state(self, df0=0, df1=0, center_epoch=0):
        """
        Apply phase shifts to profiles

        Parameters
        ----------
        df0 : float
            Frequency offset to apply.
        df1 : float
            Frequency derivative offset to apply.
        Returns
        -------
        shifted_profiles : list of ndarray
            List of phase-aligned profiles.
        """

        shifted_profiles = []
        for ar in zip(self.profiles, self.epochs):
            # Calculate phase shift
            dt = (ar[1] - center_epoch) * 86400
            dphase = df0 * dt + 0.5 * df1 * dt**2
            
            # Roll profile
            nbin = len(ar[0])
            shift = dphase * nbin
            aligned_profile = fourier_shifts.roll(ar[0], -shift)

            shifted_profiles.append(aligned_profile)

        return PulseProfilesState(
            shifted_profiles=shifted_profiles, 
            epochs=self.epochs, 
            df0=df0, 
            df1=df1, 
            center_epoch=center_epoch
        )

class PhaseCoherentSearch:
    def __init__(self, profiles, epochs, center_epoch=None):
        # Initialize parameters
        self.profiles = PulseProfiles(profiles, epochs)

        # Calculate the center epoch
        self.center_epoch = np.median(epochs) if center_epoch is None else center_epoch

    def _search_get_snr(self, df0_df1_pair):
        return self.profiles.get_state(df0=df0_df1_pair[0], df1=df0_df1_pair[1], center_epoch=self.center_epoch).get_stacked_snr()

    def search(self, df0_vals, df1_vals, ncpus=None):
        # Initialize search grid
        grid_df0_df1 = [(df0, df1) for df0 in df0_vals for df1 in df1_vals]
        grid_snrs = np.zeros(len(grid_df0_df1))

        # Iterate over the search grid
        # for i, (df0, df1) in enumerate(tqdm.tqdm(grid_df0_df1)):
        #         snr = self.profiles.get_state(df0=df0, df1=df1, center_epoch=self.center_epoch).get_stacked_snr()
        #         grid_snrs[i] = snr
        with Pool(processes=ncpus) as pool:
            grid_snrs = np.array(pool.map(self._search_get_snr, tqdm.tqdm(grid_df0_df1)))

        # Find the best df0 and df1 based on the grid search
        best_index = np.argmax(grid_snrs)
        best_df0, best_df1 = grid_df0_df1[best_index]

        # Get the best state
        best_state = self.profiles.get_state(df0=best_df0, df1=best_df1, center_epoch=self.center_epoch)
        best_state.grid_search_results = {
            "df0_vals": df0_vals,
            "df1_vals": df1_vals,
            "snrs": grid_snrs
        } # Insert grid search results into the best state for plotting

        # Return the best df0 and df1 found in the grid search
        return best_state

        # Estimate the amplitude
        # amplitude_numerator = np.sum(stacked_profile * self.templ)
        # amplitude_denominator = np.sum(self.templ ** 2)
        # # best_amplitude = amplitude_numerator / amplitude_denominator
        # best_amplitude = amplitude_denominator / amplitude_numerator
        # best_amplitude = np.max(self.templ) / (np.max(stacked_profile))
        # stacked_profile = self.get_stacked_profile(df0=best_df0, df1=best_df1, normalize=True)
        # best_amplitude = (np.max(stacked_profile)) / np.max(self.templ)

        # Find the phase offset between the stacked profile and the template
        # best_phase_offset = fourier_shifts.find_shift(stacked_profile, self.get_template(amplitude=best_amplitude)) / len(stacked_profile)
        # best_phase_offset = fourier_shifts.find_shift(stacked_profile, self.templ) / len(stacked_profile)

        # return {
        #     'df0': {
        #         "lower": best_df0 - df0_step_size * 3,
        #         "upper": best_df0 + df0_step_size * 3, 
        #         "initial_guess": best_df0
        #     },
        #     'df1': {
        #         "lower": best_df1 - df1_step_size * 5,
        #         "upper": best_df1 + df1_step_size * 5,
        #         "initial_guess": best_df1
        #     }, 
        #     # 'phase_offset': {
        #     #     "lower": best_phase_offset - 0.1,
        #     #     "upper": best_phase_offset + 0.1,
        #     #     "initial_guess": best_phase_offset
        #     # },
        #     # 'amplitude': {
        #     #     "lower": best_amplitude * 0.25,
        #     #     "upper": best_amplitude * 2,
        #     #     "initial_guess": best_amplitude
        #     # }
        # }

    # def plot(self):
    #     _, ax = plt.subplots(2, 1, figsize=(5, 10), height_ratios=[1, 2])

    #     # Prepare data
    #     if self.plotter is not None:
    #         params = self.get_postfit_params()
    #     else:
    #         print("Warning: No MCMC results found. Using initial guess parameters for plotting.")
    #         params = self.get_initial_guess()

    #     stacked_profile = self.get_stacked_profile(params["df0"][0], params["df1"][0], normalize=True)
    #     # stacked_profile = fourier_shifts.roll(stacked_profile, - params["phase_offset"][0] * len(stacked_profile))
    #     template = self.get_template(params["amplitude"][0], params["phase_offset"][0])
    #     aligned_profiles = self.get_profiles(params["df0"][0], params["df1"][0])
        
    #     # Stacked profile
    #     ax[0].plot(stacked_profile, color='k', label='Stacked Profile', lw=1)
    #     ax[0].plot(template, color='r', label='Template', lw=1)
    #     ax[0].set_xlim(0, len(stacked_profile))

    #     # Profile aligning
    #     ax[1].matshow(aligned_profiles, aspect='auto', cmap='gray_r')








    
    
    # def get_template(self, amplitude=1, phase_offset=0):
    #     return fourier_shifts.roll(self.templ * amplitude, - phase_offset * len(self.templ))

    # def log_prior(self, params):
    #     """
    #     Prior -- assuming uniform within reasonable bounds
    #     """

    #     # Get parameters
    #     df0, df1, phase_offset, amplitude = params

    #     # df0 bounds
    #     if not (self.prior_bounds['df0']['lower'] < df0 < self.prior_bounds['df0']['upper']):
    #         return -np.inf  # Outside bounds
        
    #     # df1 bounds
    #     if not (self.prior_bounds['df1']['lower'] < df1 < self.prior_bounds['df1']['upper']):
    #         return -np.inf  # Outside bounds
        
    #     # Phase offset bounds
    #     if not (self.prior_bounds['phase_offset']['lower'] < phase_offset < self.prior_bounds['phase_offset']['upper']):
    #         return -np.inf  # Outside bounds
        
    #     # Amplitude bounds
    #     if not (self.prior_bounds['amplitude']['lower'] < amplitude < self.prior_bounds['amplitude']['upper']):
    #         return -np.inf  # Outside bounds
        
    #     return 0.0  # Uniform prior within bounds

    # def log_likelihood(self, params):
    #     """
    #     Likelihood assuming Gaussian noise
    #     """

    #     # Get parameters
    #     df0, df1, phase_offset, amplitude = params

    #     # Get stacked profile
    #     stacked_profile = self.get_stacked_profile(df0, df1, normalize=True)

    #     # Phase shift the profile by the phase offset
    #     # stacked_profile = fourier_shifts.roll(stacked_profile, - phase_offset * len(stacked_profile))

    #     # Get scaled template
    #     templ_profile = self.get_template(amplitude, phase_offset=phase_offset)

    #     # Estimate noise from off-pulse region
    #     nbin = len(stacked_profile)
    #     off_pulse = np.concatenate([stacked_profile[:nbin//4], stacked_profile[-nbin//4:]])
    #     sigma = np.std(off_pulse)

    #     # Calculate Chi-squared likelihood
    #     residuals = stacked_profile - templ_profile
    #     chi2 = np.sum((residuals / sigma) ** 2)

    #     return -0.5 * chi2
    
    # def log_probability(self, params):
    #     """
    #     Posterior probability
    #     """
        
    #     lp = self.log_prior(params)
    #     if not np.isfinite(lp):
    #         return -np.inf
        
    #     prob = lp + self.log_likelihood(params)
    #     if not np.isfinite(prob):
    #         return -np.inf
        
    #     return prob
    
    # def optimize(self, nwalkers=16, nsteps=5000, initial_perturbation=0.1):
    #     # Setup random seed for reproducibility
    #     np.random.seed(42)

    #     # # Calculate initial guess
    #     # self.prior_bounds['df0']['initial_guess'], self.prior_bounds['df1']['initial_guess'] = self.calc_initial_guess()
        
    #     # Initial position for walkers
    #     pos = []
    #     for _ in range(nwalkers):
    #         walker_pos = []
    #         for param in self.prior_bounds:
    #             # perturbation = initial_perturbation * max(abs(initial_guess[j]), 1e-6) * np.random.randn()
    #             perturbation = initial_perturbation * np.random.randn() * (self.prior_bounds[param]['upper'] - self.prior_bounds[param]['lower']) / 2
    #             walker_pos.append(self.prior_bounds[param]['initial_guess'] + perturbation)
    #         pos.append(walker_pos)
    #     pos = np.array(pos)

    #     self.sampler = emcee.EnsembleSampler(nwalkers, len(self.prior_bounds), self.log_probability)
    #     self.sampler.run_mcmc(pos, nsteps, progress=True)

    #     self.plotter = MCMCPlotter(self.sampler, ['df0', 'df1', 'phase_offset', 'amplitude'], burnin_fraction=self.burnin_fraction)

    #     return self.plotter
    
    # def get_initial_guess(self):
    #     initial_guess = {}
    #     for param in self.prior_bounds:
    #         initial_guess[param] = (
    #             self.prior_bounds[param]['initial_guess'], 
    #             self.prior_bounds[param]['lower'], 
    #             self.prior_bounds[param]['upper']
    #         )

    #     return initial_guess
    
    # def get_postfit_params(self):
    #     return self.plotter.get_median_params()
    