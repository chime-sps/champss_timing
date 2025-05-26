import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm, chisquare, chi2

from .template_utils import StackTemplate
from ..utils.stats_utils import stats_utils
from ..utils.logger import logger
from ..utils.utils import utils

class ProfileStacker:
    """
    Stack multiple profiles to reach a given threshold.
    """
    def __init__(self, profiles, mjds=None, meth="sum", test="ks", threshold=1e-16, reverse=False, aligned=False, shift_meth="discrete", verbose=False, logger=logger()):
        """
        Initialize the ProfileStacker.
        :param profiles: List of profiles to stack (from latest to oldest by default if no MJDs are provided).
        :param meth: Method to stack the profiles. Allowed methods are: "sum", "mean", "median".
        :param test: Test to use for normality. Allowed tests are: "shapiro", "ks".
        :param threshold: Threshold for the test. Default is 1e-16.
        :param reverse: If True, reverse the order of the profiles (i.e. input data should from oldest to latest if is True).
        :param aligned: If True, the profiles are aligned. Default is False.
        :param shift_meth: Method to use for shifting the profiles. Allowed methods are: "discrete", "fourier". The parameter is passed to the StackTemplate class.
        :param verbose: If True, print the test results.
        """
        
        self.profiles = profiles
        self.mjds = mjds
        self.threshold = threshold
        self.verbose = verbose
        self.logger = logger
        self.allowed_tests = ["shapiro", "ks", "snr"]
        self.allowed_methods = ["sum", "mean", "median"]

        if self.mjds is None:
            self.mjds = np.arange(len(profiles))
            self.logger.warning("No mjds provided. Using default mjds from 0 to len(profiles) - 1.")
        else:
            # check if mjds and profiles have the same length
            if len(self.mjds) != len(self.profiles):
                raise ValueError(f"Length of mjds ({len(self.mjds)}) and profiles ({len(self.profiles)}) do not match.")

            # sort the profiles by mjds
            sorted_idxes = np.argsort(self.mjds)
            self.profiles = [self.profiles[i] for i in sorted_idxes]
            self.mjds = [self.mjds[i] for i in sorted_idxes]

        if reverse:
            self.profiles = list(reversed(self.profiles))

        if not aligned:
            stpl = StackTemplate(self.profiles, shift_meth=shift_meth)
            stpl.optimize()
            self.profiles = stpl.get_aligned_profiles()

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

    def get_profiles_in_time(self, same_binsize=True, max_binsize=30, min_binsize=1):
        """
        Get the profiles in time.
        """
        all_profiles = copy.deepcopy(self.profiles)
        grouped_profiles = []
        grouped_mjds = []

        # Calculate bins
        bins = []
        this_profile = []
        while len(all_profiles) > 1:
            # pop the first profile
            this_profile.append(all_profiles.pop(0))

            # min bin size
            if len(this_profile) >= min_binsize:
                # run test
                if self.test(self.stack_meth(this_profile, axis=0)) or len(this_profile) >= max_binsize:
                    bins.append(len(this_profile))
                    this_profile = []

        # If no bins
        if len(bins) == 0:
            bins = [len(self.profiles)]

        # Average bin size
        if same_binsize:
            avg_bin_size = int(np.mean(bins))
            bins = [avg_bin_size] * (len(self.profiles) // avg_bin_size)

        # # Apply max/min bin size
        # if max_binsize is not None:
        #     bins = [min(b, max_binsize) for b in bins]
        # if min_binsize is not None:
        #     bins = [max(b, min_binsize) for b in bins]

        # Bin data
        for i in range(len(bins)):
            # Get the profiles in the bin
            bin_profiles = self.profiles[i * bins[i]:(i + 1) * bins[i]]

            # Get the mjds in the bin
            bin_mjds = self.mjds[i * bins[i]:(i + 1) * bins[i]]

            # Stack the profiles in the bin
            stacked_profile = self.stack_meth(bin_profiles, axis=0)

            # Append the stacked profile and mjds to the grouped profiles
            grouped_profiles.append(stacked_profile)
            grouped_mjds.append(np.mean(bin_mjds))

        return grouped_profiles, grouped_mjds, bins

class ProfileAnalyzer(StackTemplate):
    """
    Analysis pulse profile changes
    """

    def __init__(self, profiles, mjds=None, size=1024, shift_meth="fourier", zoom_in=False, verbose=False, logger=logger()):
        """
        Initialize the ProfileAnalyzer class.

        Parameters
        ----------
        profiles : list
            List of profiles to analyze.
        size : int, optional
            Size of the profiles, by default 1024
        shift_meth : str, optional
            Method to use for shifting the profiles, "discrete" (default) or "fourier". This is passed to the StackTemplate class.
        zoom_in : bool, optional
            If True, zoom in on the profiles, by default False
        verbose : bool, optional
            If True, print verbose output, by default False
        """

        # Save mjds
        self.mjds = mjds

        # Zoom in on the profiles
        if zoom_in:
            for i, profile in enumerate(profiles):
                profiles[i] = self.zoom_in(profile)

        # Initialize the StackTemplate class
        super().__init__(profiles, size=size, shift_meth=shift_meth, verbose=verbose, logger=logger)

        # Optimize the template
        super().optimize(tol=1e-8, max_iter=100)

        # Get binned profiles
        self.binned_profiles, self.binned_mjds, self.profile_bins = ProfileStacker(
            self.get_aligned_profiles(),
            mjds=self.mjds, 
            meth="median", 
            aligned=True,
            # test="snr", 
            # threshold=3,
            # verbose=verbose,
        ).get_profiles_in_time(
            same_binsize=True, 
            max_binsize=30,
            min_binsize=3,
        )

        # Injections
        self.injected_idxes = []

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

    def inject_fake_profile_changes(self, n=1):
        """
        Inject fake profile changes into the profiles.

        Parameters
        ----------
        n : int, optional
            Number of fake profile changes to inject, by default 1
        """

        # Inject fake profile changes
        for i in range(n):
            # Random offset and scale
            injection_i = np.random.randint(0, len(self.binned_profiles))
            offset = np.random.uniform(0, int(len(self.binned_profiles) * 0.1))
            scale = np.random.uniform(0, 0.75)

            # Inject the fake profile change
            self.binned_profiles[injection_i] += scale * np.roll(self.binned_profiles[injection_i], int(offset))
            self.binned_profiles[injection_i] -= np.mean(self.binned_profiles[injection_i])
            self.binned_profiles[injection_i] /= np.std(self.binned_profiles[injection_i])

            self.injected_idxes.append(injection_i)

        return self.injected_idxes

    def get_binned_residuals(self):
        """
        Get the residuals of the profiles.

        Returns
        -------
        list
            List of residuals.
        """
        # Get the template
        template = self.get_template()

        # Get residuals
        residuals = np.array(self.binned_profiles) - template

        return residuals

    def get_binned_residual_chisquares(self):
        """
        Get the residuals of the profiles.

        Returns
        -------
        list
            List of chi-squares.
        """

        # Get the residuals
        residuals = self.get_binned_residuals()**2

        # Get the chi-squares
        # chisquares = np.array([chisquare(residual)[0] for residual in residuals])
        chisquares = np.array([chisquare(residual, f_exp=np.full(len(residual), np.mean(residual)))[0] for residual in residuals])

        return chisquares

    def get_binned_residual_rms(self):
        """
        Get the residuals of the profiles.

        Returns
        -------
        list
            List of RMS.
        """

        # Get the residuals
        residuals = self.get_binned_residuals()

        # Get the RMS
        rms = []
        for residual in residuals:
            # Get the RMS
            rms.append(np.sqrt(np.mean(residual**2)))

        return np.array(rms)

    def get_binned_residual_kstest(self):
        """
        Get the residuals of the profiles.

        Returns
        -------
        list
            List of KS test results.
        """

        # Get the residuals
        residuals = self.get_binned_residuals()

        # Get the KS test results
        ks_results = []
        for residual in residuals:
            # Get the KS test result
            ks_results.append(kstest(residual, 'norm')[1])
            # ks_results.append(kstest(residual, 'poisson', args=(np.mean(residual),))[1])

        return np.array(ks_results)

    def get_threshold_and_outliers(self, statistic="rms", threshold=3):
        if statistic == "rms":
            # Get the RMS
            rms = self.get_binned_residual_rms()

            # Get thresholds
            threshold = stats_utils.mad_outlier_thresholds(rms, z_score=threshold, return_interval=True)[1]
            
            # Get the outliers
            outliers = np.where(rms > threshold)[0]
        elif statistic == "chisquare":
            # Get the chi-squares
            chisquares = self.get_binned_residual_chisquares()

            # # Get thresholds
            # threshold = stats_utils.mad_outlier_thresholds(chisquares, z_score=threshold, return_interval=True)[1]

            # # Get the outliers
            # outliers = np.where(chisquares > threshold)[0]

            # Get thresholds (note: the distribution is no longer normal, so we use the chi-square distribution!!)
            threshold = chi2.ppf(0.95, len(self.get_template())-1)

            # Get the outliers
            outliers = np.where(chisquares > threshold)[0]
        elif statistic == "kstest":
            # Get the KS test results
            ks_results = self.get_binned_residual_kstest()

            # Get thresholds
            threshold = 0.05

            # Get the outliers
            outliers = np.where(ks_results < threshold)[0]
        else:
            raise ValueError(f"Statistic {statistic} not allowed. Allowed statistics are: rms, chisquare")

        return threshold, outliers

    def plot(self, savefig=None):
        """
        Plot the profiles.
        """

        # # Get the mjds
        # mjds = self.mjds # this mjds may not be uniform as the binned mjds and not uniform profile sampling. 
        # mjds_uniform = np.linspace(np.min(mjds), np.max(mjds), int(len(self.get_aligned_profiles()) / np.mean(self.profile_bins))) # get mjds in uniform bins

        # Get the binned profiles
        binned_profiles = self.binned_profiles
        # binned_profiles_uniform = np.zeros((len(mjds_uniform), len(binned_profiles[0])))
        # mjds_y = []
        # for i, mjd in enumerate(self.binned_mjds):
        #     # Get the index of the closest mjd
        #     idx = np.argmin(np.abs(mjds_uniform - mjd))

        #     # Get the binned profile
        #     binned_profiles_uniform[idx] = binned_profiles[i]

        #     # Append the mjd to the list
        #     mjds_y.append(mjd)
        
        # Get the template
        template = self.get_template()

        # Get the residuals
        residuals = self.get_binned_residuals()
        # residuals_uniform = np.zeros((len(mjds_uniform), len(residuals[0])))
        # for i, mjd in enumerate(self.binned_mjds):
        #     # Get the index of the closest mjd
        #     idx = np.argmin(np.abs(mjds_uniform - mjd))

        #     # Get the binned profile
        #     residuals_uniform[idx] = residuals[i]

        # Get statistics
        chisquares = self.get_binned_residual_chisquares()
        rms = self.get_binned_residual_rms()

        # Get the threshold and outliers
        chisquares_thres, chisquares_outliers = self.get_threshold_and_outliers(statistic="chisquare", threshold=3)
        rms_thres, rms_outliers = self.get_threshold_and_outliers(statistic="rms", threshold=3)
        kstest_thres, kstest_outliers = self.get_threshold_and_outliers(statistic="kstest", threshold=0.05)

        # Plot
        fig, ax = plt.subplots(2, 4, figsize=(15, 12), height_ratios=[1, 4.5], gridspec_kw={'hspace': 0, 'wspace': 0})
        ## get the vmin and vmax
        vmin = np.min(self.binned_profiles)
        vmax = np.max(self.binned_profiles)
        ## Profiles
        ax[0, 0].plot(np.linspace(0, len(template), len(template)), template, color='k')
        ax[0, 0].set_title('Profiles')
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[1, 0].matshow(self.binned_profiles, cmap="gray_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax[1, 0].tick_params(axis='x', labeltop=False, labelbottom=True)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_ylabel('MJDs')
        ax[1, 0].set_xlabel('Phase')
        ## Residuals
        ax[0, 1].plot(np.linspace(0, len(residuals[0]), len(residuals[0])), np.mean(residuals, axis=0), color='k')
        ax[0, 1].set_title('Residuals')
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[1, 1].matshow(residuals, cmap="gray_r", aspect="auto")#, vmin=vmin, vmax=vmax)
        ax[1, 1].invert_yaxis()
        ax[1, 1].tick_params(axis='x', labeltop=False, labelbottom=True)
        ax[1, 1].set_xlabel('Phase')
        ax[1, 1].set_yticks([])
        ## RMS
        ax[0, 2].hist(rms, facecolor="none", edgecolor="k", bins=50, histtype='step')
        ax[0, 2].set_title('RMS')
        ax[0, 2].set_xticks([])
        ax[0, 2].set_yticks([])
        ax[0, 2].axvline(rms_thres, color='r', linestyle='--', label='Threshold')
        ax[1, 2].plot(rms, np.linspace(0, len(chisquares), len(chisquares)), "kx")
        ax[1, 2].set_xlabel('RMS')
        ax[1, 2].set_yticks([])
        ax[1, 2].axvline(rms_thres, color='r', linestyle='--', label='Threshold')
        for i in rms_outliers:
            ax[1, 2].axhline(i, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        for i in self.injected_idxes:
            ax[1, 2].axhline(i, color='g', linestyle='-', alpha=0.5, linewidth=0.5)
        ## Chi-squares
        log_bins = np.logspace(np.log10(np.min(chisquares)), np.log10(np.max(chisquares)), 50)
        # ax[0, 3].hist(chisquares, facecolor="none", edgecolor="k", bins=50, histtype='step')
        ax[0, 3].hist(chisquares, facecolor="none", edgecolor="k", bins=log_bins, histtype='step')
        ax[0, 3].set_xscale('log')
        ax[0, 3].set_title('Chi-squares')
        ax[0, 3].set_xticks([])
        ax[0, 3].set_yticks([])
        ax[0, 3].axvline(chisquares_thres, color='r', linestyle='--')
        ax[1, 3].plot(chisquares, np.linspace(0, len(chisquares), len(chisquares)), "kx")
        ax[1, 3].set_xlabel('Chi-squares Statistics')
        ax[1, 3].set_yticks([])
        ax[1, 3].axvline(chisquares_thres, color='r', linestyle='--')
        ax[1, 3].set_xscale('log')
        for i in chisquares_outliers:
            ax[1, 3].axhline(i, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        for i in self.injected_idxes:
            ax[1, 3].axhline(i, color='g', linestyle='--', alpha=0.5, linewidth=0.5)
        # ## K-S test
        # log_bins = np.logspace(np.log10(np.min(self.get_binned_residual_kstest())), np.log10(np.max(self.get_binned_residual_kstest())), 50)
        # ax[0, 4].hist(self.get_binned_residual_kstest(), facecolor="none", edgecolor="k", bins=log_bins, histtype='step')
        # ax[0, 4].set_xscale('log')
        # ax[0, 4].set_title('K-S test')
        # ax[0, 4].set_xticks([])
        # ax[0, 4].set_yticks([])
        # ax[0, 4].axvline(kstest_thres, color='r', linestyle='--')
        # ax[1, 4].plot(self.get_binned_residual_kstest(), np.linspace(0, len(chisquares), len(chisquares)), "kx")
        # ax[1, 4].set_xlabel('K-S test Statistics')
        # ax[1, 4].set_yticks([])
        # ax[1, 4].axvline(kstest_thres, color='r', linestyle='--')
        # ax[1, 4].set_xscale('log')
        # for i in kstest_outliers:
        #     ax[1, 4].axhline(i, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        ## Set limits
        for i in range(len(ax[1, :])):
            ax[1, i].set_ylim(ax[1, 0].get_ylim())
            ax[0, i].set_xlim(ax[1, i].get_xlim())
        # ax[0, 1].set_ylim(ax[0, 0].get_ylim())
        ## Set y-ticks in time
        current_yticks = ax[1, 0].get_yticks()
        current_yticks = current_yticks[current_yticks >= 0]
        current_yticks = current_yticks[current_yticks < len(self.binned_mjds)]
        time_labels = [f"{self.binned_mjds[int(i)]:.2f}" for i in current_yticks]
        ax[1, 0].set_yticklabels(time_labels)
        ax[1, 0].set_yticks(current_yticks)
        ## Info text
        fig.text(0.001, 0, f"CHAMPSS Timing Pipeline ({utils.get_version_hash()}) ProfileAnalyzer | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MJDs on y-axis may not be uniform due to data gaps and binned profiles. ", fontsize=9, ha="left", va="bottom", family="monospace")

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()