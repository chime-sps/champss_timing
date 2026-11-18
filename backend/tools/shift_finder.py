import numpy as np
import emcee
import matplotlib.pyplot as plt

from ..utils.correlation import fourier_shifts, discrete_shifts
from ..utils.mcmc import PosteriorSamples

class ShiftFinder:
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2

        # Resample arr2 onto arr1's grid if necessary
        n1, n2 = len(arr1), len(arr2)
        if n2 != n1:
            k = np.arange(n2 // 2 + 1)
            spec = np.fft.rfft(arr2) * (n1 / n2) * np.exp(2j * np.pi * k * (0.5 / n1 - 0.5 / n2))
            spec[min(n1, n2) // 2:] = 0.0 # drop Nyquist / harmonics arr1 can't hold
            self.arr2 = np.fft.irfft(spec, n1)

        # Compute the initial guess for shift and amplitude
        self.shift, self.amplitude, self.sigma = self.compute_initial_guess()

        # Initialize the sampler
        self.sampler = None
        self.samples = None
        self.result = None

    def compute_initial_guess(self):
        # Estimate shift
        shift = fourier_shifts.find_shift(self.arr1, self.arr2)
        # shift = discrete_shifts.find_shift(self.arr1, self.arr2)

        # Estimate amplitude: least-squares value at the guessed shift
        # (std(arr1)/std(arr2) is inflated by noise and ignores the alignment)
        t = self.shift_and_resize(shift, 1.0)
        t = t - np.mean(t)
        amplitude = np.dot(self.arr1 - np.mean(self.arr1), t) / np.dot(t, t)

        # Estimate noise level (diagnostic only; sigma is marginalised in the likelihood)
        # dof: n points minus shift, amplitude and the offset removed by the mean
        residuals = self.compute_residuals(shift, amplitude)
        sigma = np.sqrt(np.sum(residuals**2) / (len(residuals) - 3))

        return shift, amplitude, sigma
    
    def compute_initial_walker_positions(self, n_walkers=50):
        spread = np.array([0.5, 0.01 * abs(self.amplitude)])   # amplitude spread relative to its scale
        initial_pos = np.array([self.shift, self.amplitude]) + spread * np.random.randn(n_walkers, 2)
        initial_pos[:, 1] = np.abs(initial_pos[:, 1])

        return initial_pos
    
    def shift_and_resize(self, shift, amplitude):
        shifted_arr2 = fourier_shifts.roll(self.arr2, -shift)
        return amplitude * shifted_arr2
    
    def compute_residuals(self, shift, amplitude):
        model = self.shift_and_resize(shift, amplitude)
        residuals = self.arr1 - model
        residuals = residuals - np.mean(residuals)
        return residuals
    
    def log_likelihood(self, params):
        shift, amplitude = params
        residuals = self.compute_residuals(shift, amplitude)
        # The noise level is unknown: marginalise it analytically (Jeffreys prior) instead
        # of freezing it at the initial guess. Equivalent to fitting sigma at the best fit.
        dof = len(residuals) - 3
        return -0.5 * dof * np.log(np.sum(residuals**2))
    
    def log_prior(self, params):
        shift, amplitude = params
        N = len(self.arr1)

        if shift < (self.shift - 0.1 * N) or shift > (self.shift + 0.1 * N):
            return -np.inf
        
        if amplitude <= 0:
            return -np.inf
        
        return 0.0
    
    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def compute(self, n_walkers=50, n_steps=2000, burn_in=1000, progress=True):
        initial_pos = self.compute_initial_walker_positions(n_walkers)
        self.sampler = emcee.EnsembleSampler(n_walkers, 2, self.log_posterior)
        self.sampler.run_mcmc(initial_pos, n_steps, progress=progress)
        
        self.result = PosteriorSamples(
            self.sampler,
            labels=["shift", "amplitude"],
            burn_in=burn_in,
            thin=15
        )

        return self.result
    
    def plot(self):
        _, ax = plt.subplots(3, 1, figsize=(10, 6))
        ax[0].plot(self.arr1, label='Array 1', alpha=0.7, color='k')

        # Initial guess
        arr2_initial = self.shift_and_resize(self.shift, self.amplitude)
        residual_initial = self.compute_residuals(self.shift, self.amplitude)
        ax[0].plot(arr2_initial, label='Initial Guess', alpha=0.7, color='b')
        ax[1].plot(residual_initial, label='Initial Residual', alpha=0.7, color='b')
        initial_param_str = f'Initial Shift: {self.shift:.2f}\nInitial Amplitude: {self.amplitude:.2f}'

        if self.result is not None: 
            mcmc_shift = self.result.shift
            mcmc_amplitude = self.result.amplitude
            arr2_mcmc = self.shift_and_resize(mcmc_shift.n, mcmc_amplitude.n)
            residual_mcmc = self.compute_residuals(mcmc_shift.n, mcmc_amplitude.n)
            ax[0].plot(arr2_mcmc, label='MCMC Result', alpha=0.7, color='r')
            ax[1].plot(residual_mcmc, label='MCMC Residual', alpha=0.7, color='r')
            mcmc_param_str = f'\nMCMC Shift: {mcmc_shift.n:.2f} +/- {mcmc_shift.s:.2f}\nMCMC Amplitude: {mcmc_amplitude.n:.2f} +/- {mcmc_amplitude.s:.2f}'
        else:
            mcmc_param_str = '\nMCMC Result: Not computed yet'

        ax[0].legend()
        ax[0].set_title('Shift Finder')
        ax[0].set_xlabel('Index')
        ax[0].set_ylabel('Value')
        ax[0].grid()


        ax[2].text(0.01, 1, initial_param_str, fontsize=10, va='top', ha='left', transform=ax[2].transAxes)
        ax[2].text(0.5, 1, mcmc_param_str, fontsize=10, va='top', ha='left', transform=ax[2].transAxes)
        ax[2].axis('off')

        plt.show()