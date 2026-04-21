import numpy as np
import emcee
import matplotlib.pyplot as plt

from ..utils.correlation import fourier_shifts
from ..utils.mcmc import PosteriorSamples

class ShiftFinder:
    def __init__(self, arr1, arr2):
        self.arr1 = self.normalize(arr1)
        self.arr2 = self.normalize(arr2)

        # Compute the initial guess for shift and amplitude
        self.shift, self.amplitude = self.compute_initial_guess()

        # Initialize the sampler
        self.sampler = None
        self.samples = None
        self.result = None

    def normalize(self, arr):
        # Estimate baseline level
        noise_i = arr[arr < np.quantile(arr, 0.2)]
        baseline_level = np.median(noise_i)

        return (arr - baseline_level)

    def compute_initial_guess(self):
        # Shift
        shift = fourier_shifts.find_shift(self.arr1, self.arr2)

        # Amplitude
        amplitude = np.sum(self.arr1) / np.sum(self.arr2)

        return shift, amplitude
    
    def shift_and_resize(self, shift, amplitude):
        shifted_arr2 = fourier_shifts.roll(self.arr2, -shift)
        return amplitude * shifted_arr2
    
    def log_likelihood(self, params):
        shift, amplitude = params
        model = self.shift_and_resize(shift, amplitude)
        residuals = self.arr1 - model
        return -0.5 * np.sum(residuals**2)
    
    def log_prior(self, params):
        shift, amplitude = params
        if -len(self.arr1) < shift < len(self.arr1) and 0 < amplitude < 10:
            return 0.0
        return -np.inf
    
    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def compute(self, n_walkers=50, n_steps=2000, burn_in=1000):
        initial_pos = [self.shift, self.amplitude] + 1e-4 * np.random.randn(n_walkers, 2)
        self.sampler = emcee.EnsembleSampler(n_walkers, 2, self.log_posterior)
        self.sampler.run_mcmc(initial_pos, n_steps, progress=True)
        
        self.result = PosteriorSamples(
            self.sampler,
            labels=["shift", "amplitude"],
            burn_in=burn_in,
            thin=15
        )

        return self.result
    
    def plot(self):
        if self.result is not None: 
            shift, shift_err = self.result.shift.n, self.result.shift.s
            amplitude, amplitude_err = self.result.amplitude.n, self.result.amplitude.s
            source = "MCMC"
        else: 
            shift, amplitude = self.shift, self.amplitude
            shift_err, amplitude_err = 0, 0
            source = "Initial Guess"

        arr_shifted = self.shift_and_resize(shift, amplitude)
        plt.figure(figsize=(10, 5))
        plt.plot(self.arr1, label='Array 1', alpha=0.7, color='k')
        plt.plot(self.arr2, label='Array 2', alpha=0.7, color='b')
        plt.plot(arr_shifted, label='Shifted & Resized Array 2', alpha=0.7, color='r')
        plt.legend()
        plt.title(f"Shift={shift:.2f} +/- {shift_err:.2f}, Amplitude={amplitude:.2f} +/- {amplitude_err:.2f} from {source}")
        plt.show()
    