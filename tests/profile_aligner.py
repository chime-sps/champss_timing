import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import shapiro, kstest, norm

class ProfileAlignerState:
    def __init__(self, template, data, lag, scaling_factor, zoom_in=True):
        self.template = template
        self.data = data
        self.lag = None
        self.scaling_factor = None
        self.bad_bins = []

        # Zoom in on the data if needed
        # if zoom_in:
        #     self.data = self.zoom_in(self.data)
        #     self.template = self.zoom_in(self.template)
        
        # Set the lag and scaling factor
        # self.set_lag_and_scaling(lag, scaling_factor)
        self.lag = lag
        self.scaling_factor = scaling_factor

    def zoom_in(self, data, percent=0.25):
        """
        Zoom in on the fraction of the data of interest.
        """
        # Find the peak
        peak = np.argmax(data)

        # Find the width (half_width) of the interested region
        half_width = int(len(data) * percent / 2)

        # Repeat the data to make sure it is long enough, and cut the data to the zoomed in region
        data = np.tile(data, 3)[len(data) + peak - half_width:len(data) + peak + half_width]

        return data

    def set_lag_and_scaling(self, lag, scaling_factor):
        """
        Set the lag and scaling factor.
        """
        self.lag = lag
        self.scaling_factor = scaling_factor

        # Get bad bins
        self.bad_bins = self.bad_bins + self.get_bad_bins()

    def get_aligned_data(self):
        """
        Get the aligned data.
        """

        # Shift the data by the lag
        shifted_data = np.roll(self.data, -int(self.lag))

        # Scale the data by the scaling factor
        scaled_data = shifted_data * self.scaling_factor

        return scaled_data

    def get_residuals(self, remove_bad_bins=True, bad_bins_val=0):
        """
        Get the residuals.
        """

        # Get the aligned data
        aligned_data = self.get_aligned_data()

        # Calculate the residuals
        residuals = aligned_data - self.template

        # Remove bad bins
        if remove_bad_bins and len(self.bad_bins) > 0:
            residuals[self.bad_bins] = bad_bins_val

        return residuals

    def get_bad_bins(self):
        '''
        Get the bad bins (2-sigma away from delta).
        '''

        # Get residuals
        residuals = self.get_residuals(bad_bins_val=np.nan)

        # Get bad bins
        bad_bins = np.where(np.abs(residuals) > 2 * np.std(residuals) + self.calc_delta())[0]
        # bad_bins = np.where((residuals) > 2 * np.std(residuals) + self.calc_delta())[0]
        
        return self.bad_bins + bad_bins.tolist()

    def calc_delta(self):
        """
        Calculate the delta value for the given lag and scaling factor.
        """

        residuals = self.get_residuals()

        # Calculate the delta value
        # delta = np.sum(np.abs(residuals)) / len(residuals)
        delta = np.sum(residuals**2) / len(residuals)
        
        return delta

    def calc_delta_nf(self):
        bad_bins = self.get_bad_bins()

        if (len(self.data) - len(bad_bins)) <= 0:
            return np.inf

        # Get n_duty (number of bins > std in template)
        n_duty = np.max([np.sum(self.template > np.std(self.template)), np.sum(self.data > np.std(self.data))])

        # Calculate n_f
        # n_f = (len(self.data) - len(self.bad_bins))
        n_f = (n_duty - len(bad_bins))

        # Panelty for rejecting bins near the maximum
        # ## find where the maximum is in data
        # max_bin = np.argmax(self.data)
        # ## find the distance to the maximum
        # bad_bins_dist

        # Sanity check
        if n_f <= 0:
            return np.inf

        # return self.calc_delta() / (len(self.data) - len(self.bad_bins))
        return self.calc_delta() / n_f

    def plot(self, title=None):
        """
        Plot the current state of the profile aligner.
        """

        # Get the aligned data
        aligned_data = self.get_aligned_data()

        # Get the residuals
        # residuals = self.get_residuals(remove_bad_bins=False)
        residuals = self.get_residuals(remove_bad_bins=True, bad_bins_val=np.nan)

        # Plot the aligned data and the residuals
        plt.plot(aligned_data, label='Aligned Data', c='g')
        plt.plot(self.template, label=' Template', c='b')
        plt.plot(residuals, label='Residuals', c='k')

        # Show information about the lag and scaling factor
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'Lag: {self.lag}, Scaling Factor: {self.scaling_factor}')
        plt.text(0.025, 0.975, f'$\delta$ = {self.calc_delta()}, \n$\delta/n_' + '{\\rm f}' + f'$ = {self.calc_delta_nf()}', fontsize=9, 
                 transform=plt.gca().transAxes, ha='left', va='top')
        plt.xlabel('Bin Number')
        plt.ylabel('Amplitude')
        plt.axhline(0, color='k', linestyle='--')

        # Plot the bad bins
        if len(self.bad_bins) > 0:
            # plt.scatter(self.bad_bins, [0] * len(self.bad_bins), c='r', label='Bad Bins', marker=".")
            for bad_bin in self.bad_bins:
                plt.axvline(bad_bin, color='r', linestyle='-', alpha=0.15, lw=0.5)
            plt.axvline(bad_bin, color='r', linestyle='-', label='Bad Bins', alpha=0.15, lw=0.5)

        plt.legend(loc='upper right')
        plt.show()

class ProfileAligner:
    def __init__(self, template, data, initial_guess=None, verbose=False):
        # Preprocess the data
        # template = np.array(template) / np.std(template)
        # template = template - np.median(template)
        # data = np.array(data) / np.std(data)
        # data = data - np.median(data)

        self.template = template
        self.data = data
        self.verbose = verbose

        # Resize the template into the same size as the data
        if len(self.template) != len(self.data):
            self.template = np.interp(
                np.linspace(0, len(self.template), len(self.data)),
                np.arange(len(self.template)),
                self.template,
            )
        
        # Create the initial state
        if initial_guess is None:
            initial_guess = self.guess()
        self.state = ProfileAlignerState(self.template, self.data, *initial_guess)

        # Set search steps
        # self.lag_range = (self.state.lag - int(len(self.data) * 0.05), self.state.lag + int(len(self.data) * 0.05))
        # self.scaling_factor_range = (np.max(self.state.scaling_factor) * 0.50, np.max(self.state.scaling_factor) * 1.5)
        self.d_lag = int(len(self.data) * 0.05)
        self.d_scaling_factor = self.state.scaling_factor * 0.05

        if self.verbose:
            print(f"Initial Lag: {self.state.lag}, Scaling Factor: {self.state.scaling_factor}")
            print(f"Initial Delta: {self.state.calc_delta()}, Delta/nf: {self.state.calc_delta_nf()}")
            print(f"Search Steps: d_lag = {self.d_lag}, d_scaling_factor = {self.d_scaling_factor}")

    def guess(self):
        """
        Calculate initial guess for the lag and scaling factor.
        """

        # Calculate initial guess for the lag
        cross_correlation = np.correlate(self.data, self.template, mode='full')
        lag = np.argmax(cross_correlation) - (len(self.template) - 1)

        # Calculate initial guess for the scaling factor
        # scaling_factor = np.sum(self.template ** 2) / np.sum(self.data * np.roll(self.template, lag))
        # scaling_factor = np.sum(self.template) / np.sum(self.data)
        scaling_factor = np.max(self.template) / np.max(self.data)

        return lag, scaling_factor

    def take_step(self, new_lag, new_scaling_factor):
        """
        Take a step in the optimization process.
        """

        # Get a new state
        # new_state = ProfileAlignerState(self.template, self.data, new_lag, new_scaling_factor)
        new_state = copy.deepcopy(self.state)
        new_state.set_lag_and_scaling(new_lag, new_scaling_factor)

        return new_state
        # return new_state.calc_delta_nf()

    def take_step_and_calc_delta_nf(self, new_lag, new_scaling_factor):
        """
        Take a step in the optimization process and calculate the delta_nf.
        """

        return self.take_step(new_lag, new_scaling_factor).calc_delta_nf()

    def take_trials(self, lag, d_lag, scaling_factor, d_scaling_factor, n_trials=[10, 10], n_iters=10):
        """
        Take a series of trials to find the best lag and scaling factor.
        """

        if n_iters == 0:
            return lag, scaling_factor

        # Generate a grid of lag and scaling factor values
        lag_values = np.linspace(lag - d_lag, lag + d_lag, n_trials[0])
        scaling_factor_values = np.linspace(scaling_factor - d_scaling_factor, scaling_factor + d_scaling_factor, n_trials[1])

        # Make sure lag_values are only integers
        lag_values = np.round(lag_values).astype(int)
        lag_values = np.unique(lag_values) # Remove duplicates to save time
        n_trials[0] = len(lag_values)

        if self.verbose:
            print("", f"Taking trials with lag = {lag}, d_lag = {d_lag}, scaling_factor = {scaling_factor}, d_scaling_factor = {d_scaling_factor}")
            print("", f"Number of trials: {n_trials[0]} x {n_trials[1]}")
        
        # Run trials
        results = np.zeros((n_trials[0], n_trials[1]))
        for i, lag in enumerate(lag_values):
            for j, scaling_factor in enumerate(scaling_factor_values):
                # Get the new state
                new_state = self.take_step(lag, scaling_factor)

                # Calculate the delta_nf
                results[i, j] = new_state.calc_delta_nf()

        # Get the best lag and scaling factor
        best_lag, best_scaling_factor = np.unravel_index(np.argmin(results), results.shape)
        best_lag = lag_values[best_lag]
        best_scaling_factor = scaling_factor_values[best_scaling_factor]
        
        if self.verbose:
            print("", "", f"Best Lag: {best_lag}, Best Scaling Factor: {best_scaling_factor}")

        return self.take_trials(
            lag=best_lag,
            d_lag=d_lag - 1,
            scaling_factor=best_scaling_factor,
            d_scaling_factor=d_scaling_factor * 0.9,
            n_trials=n_trials, 
            n_iters=n_iters - 1
        )

    def optimize(self, max_iters=100, delta_nf_tol=1e-16, d_delta_nf_tol=0.0001):
        """
        Optimize the lag and scaling factor.
        """
        for i in range(max_iters):
            # # Minimize the d_delta_nf
            # result = minimize(
            #     lambda x: self.take_step_and_calc_delta_nf(x[0], x[1]),
            #     [self.state.lag, self.state.scaling_factor],
            #     bounds=[
            #         (self.state.lag - int(len(self.data) * 0.05), self.state.lag + int(len(self.data) * 0.05)), 
            #         (np.max(self.state.scaling_factor) * 0.99, np.max(self.state.scaling_factor) * 1.01)
            #     ], 
            #     options={"maxiter": 1, "disp": False}
            # )

            # # Get the new lag and scaling_factor
            # new_lag, new_scaling_factor = result.x

            # Take trials
            new_lag, new_scaling_factor = self.take_trials(
                lag=self.state.lag,
                d_lag=self.d_lag,
                scaling_factor=self.state.scaling_factor,
                d_scaling_factor=self.d_scaling_factor,
                n_trials=[2 * self.d_lag, 150]
            )

            # Get the new state
            new_state = self.take_step(new_lag, new_scaling_factor)

            # Calculate the d_delta_nf
            d_delta_nf = (self.state.calc_delta_nf() - new_state.calc_delta_nf()) / self.state.calc_delta_nf()

            # Print the new state
            if self.verbose:
                print(f"Iteration {i}: d_delta_nf = {d_delta_nf}, lag = {new_lag}, scaling_factor = {new_scaling_factor}")
                new_state.plot()

            # Check for convergence
            if d_delta_nf < d_delta_nf_tol or np.abs(new_state.calc_delta_nf()) < delta_nf_tol:
                if self.verbose:
                    self.state.plot(title="Best State")
                    print(f"Converged after {i} iterations (d_delta_nf = {d_delta_nf})")
                break

            # Update the state
            self.state = new_state


        return self.state

class ProfileStacker:
    def __init__(self, profiles, meth="sum", test="ks", threshold=1e-16, reverse=False, verbose=False):
        """
        Initialize the ProfileStacker.
        :param profiles: List of profiles to stack (from latest to oldest).
        :param meth: Method to stack the profiles. Allowed methods are: "sum", "mean", "median".
        :param test: Test to use for normality. Allowed tests are: "shapiro", "ks".
        :param threshold: Threshold for the test. Default is 1e-16.
        :param reverse: If True, reverse the order of the profiles (i.e. input data should from oldest to latest if is True).
        :param verbose: If True, print the test results.
        """

        self.profiles = profiles
        self.threshold = threshold
        self.verbose = verbose
        self.allowed_tests = ["shapiro", "ks", "snr"]
        self.allowed_methods = ["sum", "mean", "median"]

        if reverse:
            self.profiles = list(reversed(self.profiles))

        self.stack_meth = None
        if meth == "sum":
            self.stack_meth = np.sum
        elif meth == "mean":
            self.stack_meth = np.mean
        elif meth == "median":
            self.stack_meth = np.median
        else:
            raise ValueError(f"Method {meth} not allowed. Allowed methods are: {self.allowed_methods}")

        self.test = None
        if test == "shapiro":
            self.test = self.shapiro_test
        elif test == "ks":
            self.test = self.ks_test
        elif test == "snr":
            self.test = self.snr
        else:
            raise ValueError(f"Test {test} not allowed. Allowed tests are: {self.allowed_tests}")

    def shapiro_test(self, data):
        """
        Perform the Shapiro-Wilk test for normality.
        """
        stat, p = shapiro(data)

        if p < self.threshold:
            if self.verbose:
                print(f"Test passed. p-value: {p}")
            return True
        else:
            return False

    def ks_test(self, data):
        """
        Perform the Kolmogorov-Smirnov test for normality.
        """
        stat, p = kstest(data, 'norm')
        
        if p < self.threshold:
            if self.verbose:
                print(f"Test passed. p-value: {p}")
            return True
        else:
            return False

    def snr(self, data):
        """
        Calculate the signal-to-noise ratio (SNR).
        """
        # Calculate the SNR
        snr = np.max(data) / np.std(data)
        
        if snr > self.threshold:
            if self.verbose:
                print(f"Threshold passed. snr: {snr}")
            return True
        else:
            return False
        
    def get_latest_profile(self):
        """
        Get the latest profile.
        """
        profiles = []
        for i, profile in enumerate(self.profiles):
            # append profile
            profiles.append(profile)

            # run test
            if self.test(self.stack_meth(profiles, axis=0)):
                break

        if len(profiles) == len(self.profiles):
            raise ValueError(f"Test failed even after stacking all {len(profiles)} profiles. ")

        stacked_profile = self.stack_meth(profiles, axis=0)
        return stacked_profile

    def get_profiles_in_time(self, same_binsize=True):
        """
        Get the profiles in time.
        """
        all_profiles = copy.deepcopy(self.profiles)
        grouped_profiles = []

        # Calculate bins
        bins = []
        this_profile = []
        while len(all_profiles) > 1:
            # pop the first profile
            this_profile.append(all_profiles.pop(0))

            # run test
            if self.test(self.stack_meth(this_profile, axis=0)):
                bins.append(len(this_profile))
                this_profile = []

        # Average bin size
        if same_binsize:
            avg_bin_size = int(np.mean(bins))
            bins = [avg_bin_size] * (len(self.profiles) // avg_bin_size)

        # Bin data
        for i in range(len(bins)):
            # Get the profiles in the bin
            bin_profiles = self.profiles[i * bins[i]:(i + 1) * bins[i]]

            # Stack the profiles in the bin
            stacked_profile = self.stack_meth(bin_profiles, axis=0)

            # Append the stacked profile to the grouped profiles
            grouped_profiles.append(stacked_profile)

        return grouped_profiles