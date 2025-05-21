import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from ..utils.correlation import discrete_shifts, fourier_shifts
from ..utils.logger import logger
from ..io.template_writer import TemplateWriter

class StackTemplateState:
    def __init__(self, profiles, shift_meth):

        # normalize the profiles
        for i, profile in enumerate(profiles):
            if np.std(profile) == 0:
                continue

            # normalize the profile
            profiles[i] = (profile - np.mean(profile)) / np.std(profile)

        self.profiles = profiles
        self.template = np.median(np.array(profiles), axis=0)

        if shift_meth == "discrete":
            self.shifts_utils = discrete_shifts
        elif shift_meth == "fourier":
            self.shifts_utils = fourier_shifts
        else:
            raise ValueError(f"Unknown shift type: {shift_meth}. Use 'discrete' or 'fourier'.")

    def stack_template(self):
        """
        Stack the data from multiple files to create a template.
        """
        # align the data
        aligned_profiles = []
        for profile in self.profiles:
            if np.std(profile) == 0:
                aligned_profiles.append(profile) # still add the profile to keep the same number of profiles
                continue

            # find shift
            lag = self.shifts_utils.find_shift(profile, self.template)

            # shift the profile
            aligned_profiles.append(self.shifts_utils.roll(profile, -lag))

        # stack the data
        self.template = np.median(np.array(aligned_profiles), axis=0)
        self.profiles = aligned_profiles

    def get_snr(self):
        """
        Calculate the signal-to-noise ratio (SNR) of the template.
        """
        # calculate the mean and standard deviation
        template_max = np.max(self.template)
        template_std = np.std(self.template)

        # sanity check
        if template_std == 0:
            return 0

        # calculate the SNR
        template_snr = template_max / template_std

        return template_snr
    
    def plot(self):
        """
        Plot the template.
        """
        
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 10), sharex=True, height_ratios=[1, 4])
        ax[0].set_title('Template')
        ax[0].plot(self.template, label='Template')
        ax[0].set_ylabel('Amplitude')
        ax[0].text(0.975, 0.95, f'SNR = {self.get_snr():.2f}', transform=ax[0].transAxes, va='top', ha='right')
        ax[1].set_title('Daily Profiles')
        ax[1].matshow(self.profiles, aspect='auto')
        ax[1].set_ylabel('Profiles')
        ax[1].set_xlabel('Samples in Phase')
        plt.show()

class StackTemplate:
    """
    This class creates a template.
    """
    def __init__(self, profiles, size=1024, shift_meth="discrete", verbose=False, logger=logger()):
        # make sure the profiles are the same size
        for i, profile in enumerate(profiles):
            if len(profile) != size:
                profiles[i] = np.interp(np.linspace(0, len(profile), size), np.arange(len(profile)), profile)

        self.state = StackTemplateState(profiles, shift_meth=shift_meth)
        self.verbose = verbose
        self.logger = logger

    def optimize(self, tol=1e-8, max_iter=100):
        """
        Optimize the template by iteratively updating it.
        """

        if self.verbose:
            self.logger.debug(f"Initial SNR = {self.state.get_snr():.4f}")
            self.state.plot()

        for i_iter in range(max_iter):
            # copy the state
            this_state = copy.deepcopy(self.state)

            # stack the template
            this_state.stack_template()

            # compare the snr
            d_snr = this_state.get_snr() - self.state.get_snr()
            if self.verbose:
                self.logger.debug(f"Iteration {i_iter}: ")
                self.logger.debug(f"  SNR_prev = {self.state.get_snr():.4f}")
                self.logger.debug(f"  SNR_this = {this_state.get_snr():.4f}")
                self.logger.debug(f"  dSNR = {d_snr:.4f} (tol = {tol})")
                this_state.plot()

            if d_snr < tol:
                if self.verbose:
                    self.logger.success(f"Converged after {i_iter} iterations, dSNR = {d_snr}")
                    self.state.plot()
                break

            # update the state
            self.state = this_state

    def get_residuals(self):
        """
        Calculate the residuals of the profiles.
        """
        # Get the template and aligned profiles
        template, aligned_profiles = self.get_template_and_aligned_profiles()

        # Calculate the residuals
        residuals = np.array([profile - template for profile in aligned_profiles])

        return residuals

    def get_template(self):
        """
        Get the template.
        """
        return self.state.template

    def get_aligned_profiles(self):
        """
        Get the aligned profiles.
        """
        return self.state.profiles

    def get_template_and_aligned_profiles(self):
        """
        Get the template and aligned profiles.
        """
        return self.get_template(), self.get_aligned_profiles()

    def to_archive(self, filename, psr="Template", dm=0, overwrite=True):
        """
        Write the template to a PSRCHIVE compatible format.
        """
        
        with TemplateWriter(filename, overwrite=overwrite) as writer:
            # Write the data to the archive
            writer.write(self.get_template(), interpolate=True)

            # Set the metadata
            writer.set_source(psr)
            writer.set_dm(dm)
            
            # Unload the archive
            writer.unload()