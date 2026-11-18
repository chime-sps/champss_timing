import numpy as np
from scipy.stats import normaltest, sigmaclip
from scipy.signal import peak_widths
from scipy.ndimage import uniform_filter1d
from .correlation import fourier_shifts, discrete_shifts, subsample_shifts

class data_quality_utils:
    @staticmethod
    def boxcar(profile, max_width=None):
        """
        Helper function to compute the boxcar SNR of a profile (returns best SNR and corresponding width)
        """

        # Ensure profile is a numpy array
        profile = np.asarray(profile, dtype=float)

        # Estimate baseline and noise level
        n = profile.size
        baseline = np.median(profile)
        sigma = np.std(np.sort(profile)[: 3 * n // 4])

        # Generate widths
        widths = 2 ** np.arange(int(np.log2((max_width or n // 4))) + 1)

        # Try different boxcar widths and compute snr
        snrs = [(uniform_filter1d(profile, w, mode="wrap").max() - baseline) / (sigma / np.sqrt(w)) for w in widths]

        # Find the width that maximizes the SNR
        best_i = np.argmax(snrs)
        best_width = widths[best_i]
        best_snr = snrs[best_i]

        return best_snr, best_width
    
    @staticmethod
    def boxcar_snr(profile, max_width=None):
        """Helper function to compute the boxcar SNR of a profile (returns the best SNR)"""
        return data_quality_utils.boxcar(profile, max_width)[0]

    @staticmethod
    def chi2(profile):
        """Compute the chi-squared of a profile against pure noise"""
        x = np.asarray(profile, dtype=float)
        return ((x - x.mean())**2).sum()

    @staticmethod
    def remove_baseline(profile):
        """Remove the baseline from a profile"""

        # Ensure profile is a numpy array
        profile = np.asarray(profile, dtype=float)

        # Find noise baseline
        noise = sigmaclip(profile, low=3, high=3)[0]
        
        return profile - np.mean(noise)

    @staticmethod
    def pulse_width(profile, frac=0.5):
        """
        Measure pulse width of a profile (return width in *bins*). 
        """

        # Remove baseline
        profile = data_quality_utils.remove_baseline(profile)

        # Center profile to make sure it's not wrapping around the edges
        # (note: this may be still problematic for highly scatter pulsars?)
        n = profile.size
        profile = np.roll(profile, n // 2 - np.argmax(profile))

        # Find width above frac * peak
        i_peak = n // 2
        width = peak_widths(
            profile,
            peaks=[i_peak],
            rel_height=1 - frac,
            prominence_data=(
                np.array([profile[i_peak]]), 
                np.array([0]), 
                np.array([n - 1])
            ), # make sure height relative to 0 instead of local minima
        )[0][0]

        return width

class MatchedFilterSNR:
    def __init__(self, profile, template, shift_meth="discrete"):
        self.profile = np.array(profile)
        self.template = np.array(template)

        # Get shift method
        if shift_meth == "discrete":
            self.shift_method = discrete_shifts
        elif shift_meth == "subsample":
            self.shift_method = subsample_shifts
        elif shift_meth == "fourier":
            self.shift_method = fourier_shifts
        else:
            raise ValueError("Invalid shift method. Choose from 'discrete', 'subsample', or 'fourier'.")
        
        # Normalize the profile and template
        self.profile = self.normalize(self.profile)
        self.template = self.normalize(self.template)

        # Scale the template to match the profile
        if len(self.profile) != len(self.template):
            self.template = np.interp(
                np.linspace(0, len(self.template) - 1, len(self.profile)),
                np.arange(len(self.template)),
                self.template
            )

        # Cross-correlate template to the profile
        self.template = self.shift_method.roll(
            self.template, 
            -self.shift_method.find_shift(self.profile, self.template)
        )

        # Match the amplitudes
        self.template = self.template * (
            np.quantile(self.profile, 0.99) / np.quantile(self.template, 0.99)
        )

    def compute(self, noise_std=None):
        """
        Compute the matched filter SNR.
        noise_std: externally provided noise estimate (recommended).
                If None, falls back to std of residual (less accurate).
        """

        # Estimate noise level if not provided
        if noise_std is None:
            noise_std = np.std(self.profile - self.template)
        
        # Calculate SNR
        snr = np.sum(self.profile * self.template) / np.sqrt(np.sum(self.template**2)) / noise_std # Can handle faint signals better. 
        
        # Sanity checks
        if np.isnan(snr) or np.isinf(snr) or snr < 0:
            return 0.0

        return snr

    def normaltest(self):
        """
        Compute the probability that the profile is consistent with noise using a normality test on the residual.
        """

        return normaltest(self.profile)
    
    def plot(self, ax=None):
        """
        Plot the profile and template.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.profile, label='Profile', color='blue')
        ax.plot(self.template, label='Template', color='orange')
        ax.plot(self.profile - self.template, label='Residual', color='green', linestyle='--')
        ax.set_title('Profile and Template (SNR: {:.2f})'.format(self.compute()))
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.legend()
        
        return ax
    
    def normalize(self, data):
        """
        Normalize the data to have zero mean and unit variance.
        """
        data = np.array(data)
        data = data - np.mean(data)

        if np.std(data) == 0:
            return data
        
        return data / np.std(data)