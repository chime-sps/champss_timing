import numpy as np
from scipy.stats import normaltest
from .correlation import fourier_shifts, discrete_shifts, subsample_shifts

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