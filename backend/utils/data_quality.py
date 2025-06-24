import numpy as np
from .correlation import fourier_shifts, discrete_shifts, subsample_shifts

class MatchedFilterSNR:
    def __init__(self, profile, template, shift_meth="discrete"):
        self.profile = profile
        self.template = template

        # Get shift method
        if shift_meth == "discrete":
            self.shift_method = discrete_shifts
        elif shift_meth == "subsample":
            self.shift_method = subsample_shifts
        elif shift_meth == "fourier":
            self.shift_method = fourier_shifts
        else:
            raise ValueError("Invalid shift method. Choose from 'discrete', 'subsample', or 'fourier'.")

        # Cross-correlate template to the profile
        self.template = self.shift_method.roll(
            self.template, 
            -self.shift_method.find_shift(self.profile, self.template)
        )

        # Match the amplitudes
        self.template = self.template * (
            np.mean(self.profile) / np.mean(self.template)
        )

    def compute(self):
        """
        Compute the matched filter SNR.
        """
        
        # Calculate signal and noise levels
        chisq_signal = np.sum((self.profile)**2)
        chisq_noise = np.sum((self.profile - self.template)**2)

        # Calculate SNR
        if chisq_signal - chisq_noise < 0:
            return 0.0
        snr = np.sqrt(chisq_signal - chisq_noise)

        return snr
    
    def plot(self, ax=None):
        """
        Plot the profile and template.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.profile, label='Profile', color='blue')
        ax.plot(self.template, label='Template', color='orange')
        ax.set_title('Profile and Template (SNR: {:.2f})'.format(self.compute()))
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.legend()
        
        return ax