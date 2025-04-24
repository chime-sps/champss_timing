import numpy as np
from scipy.stats import median_abs_deviation

class stats_utils:
    """
    Statistics Utility Class
    This class provides utility functions for statistical analysis.
    """

    @staticmethod
    def mad(data):
        """
        Calculate the Median Absolute Deviation (MAD) of a dataset.

        Parameters
        ----------
        data : list or numpy array
            The input data.

        Returns
        -------
        float
            The MAD of the data.
        """
        return median_abs_deviation(data, scale='normal')

    @staticmethod
    def mad_to_stdev(mad):
        """
        Convert median absolute deviation to standard deviation
        Ref: https://real-statistics.com/descriptive-statistics/measures-variability/relationship-between-std-and-mad/
        
        Parameters
        ----------
        mad : float
            Median absolute deviation
        Returns
        -------
        float
            Standard deviation
        """
        return 1.4826 * mad

    @staticmethod
    def mad_outlier_test(samples, point):
        """
        This function is used to determine if a point is an outlier or not. 
        It is used in the MAD outlier test. 

        Parameters
        ----------
        samples : array_like
            The sample of values to test the point against. 
        point : float
            The point to test. 

        Returns
        -------
        float
            The z-score of the point.
        """

        # get mad and median
        mad = median_abs_deviation(samples)
        median = np.median(samples)

        # estimate std from mad
        std = stats_utils.mad_to_stdev(mad)

        # sanity check
        if std == 0:
            return 0.0

        # calculate the z-score
        z_score = np.abs(point - median) / std

        return z_score

    @staticmethod
    def mad_outlier_thresholds(samples, z_score=3, return_interval=True):
        """
        This function is used to determine the threshold for the MAD outlier test. 
        It is used in the MAD outlier test. 

        Parameters
        ----------
        samples : array_like
            The sample of values to test the point against. 
        z_score : float
            The threshold z_score to use (n x sigma). 
            Default is 3.0, which corresponds to 99.7% confidence interval.
        return_interval : bool
            If True, return the lower and upper threshold. 
            If False, return the d_threshold (threshold off from the median). 
            Default is True.

        Returns
        -------
        tuple
            The lower and upper threshold for the MAD outlier test.
        """

        # get mad and median
        mad = median_abs_deviation(samples)
        median = np.median(samples)

        # estimate std from mad
        std = stats_utils.mad_to_stdev(mad)

        # sanity check
        if std == 0:
            return np.inf

        # calculate the threshold
        d_threshold = np.abs(z_score * std)
        threshold_upper = median + d_threshold
        threshold_lower = median - d_threshold

        if return_interval:
            return threshold_lower, threshold_upper
            
        return d_threshold

        