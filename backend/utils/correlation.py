import numpy as np
from scipy.ndimage import fourier_shift
from scipy.signal import butter, filtfilt

try:
    from scipy.signal import gaussian
except ImportError:
    from scipy.signal.windows import gaussian

class discrete_shifts:
    """
    Cross-correlate utilities based on discrete shifts.
    """

    @staticmethod
    def correlate(arr1, arr2):
        """
        Compute the discrete correlation between two arrays.
        Parameters:
            arr1 (np.ndarray): The first array.
            arr2 (np.ndarray): The second array.
        Returns:
            np.ndarray: The discrete correlation of the two arrays.
        """
        return np.correlate(arr1, arr2, mode='full')

    @staticmethod
    def find_shift(arr1, arr2):
        """
        Find the shift between two arrays using standard cross-correlation.
        Parameters
        ----------
        arr1 : np.ndarray
            The first array.
        arr2 : np.ndarray
            The second array (shifted version of arr1).
        Returns
        -------
        int
            The estimated shift.
        """
        # Compute the cross-correlation
        corr = discrete_shifts.correlate(arr1, arr2)

        # Find the index of the maximum correlation
        max_corr_idx = np.argmax(corr)

        # Calculate the shift
        shift = max_corr_idx - (len(arr1) - 1)

        return shift

    @staticmethod
    def roll(arr, shift):
        """
        Roll the array using discrete shifts.
        Parameters
        ----------
        arr : np.ndarray
            The input array to be shifted.
        shift : int
            The amount to shift the array. Must be an integer.
        Returns
        -------
        np.ndarray
            The shifted array.
        """
        if shift == 0:
            return arr

        # Roll the array
        return np.roll(arr, int(shift))

class fourier_shifts:
    """
    Cross-correlate utilities based on fourier shifts.
    """

    @staticmethod
    def correlate(arr1, arr2):
        """
        Compute the Fourier correlation between two arrays.
        Parameters:
            arr1 (np.ndarray): The first array.
            arr2 (np.ndarray): The second array.
        Returns:
            np.ndarray: The Fourier correlation of the two arrays.
        """

        # Compute the Fourier Transform of both arrays
        fft_arr1 = np.fft.fft(arr1)
        fft_arr2 = np.fft.fft(arr2)
        
        # Compute the cross-correlation in the frequency domain
        cross_corr = np.conj(fft_arr1) * fft_arr2
        
        # Inverse Fourier Transform to get the correlation in the spatial domain
        corr = np.fft.ifft(cross_corr)
        
        return np.real(corr), cross_corr

    @staticmethod
    def __fourier_find_shift(signal1, signal2, max_diff=0.5):
        """
        Estimate the sub-sample shift between two signals using Fourier Shift Theorem.

        Parameters:
            signal1 (np.ndarray): The first signal.
            signal2 (np.ndarray): The second signal (shifted version of signal1).
            max_diff (float): The maximum expected difference between the two signals.

        Returns:
            float: The estimated sub-sample shift.
        """

        # Compute the Fourier correlation
        corr, cross_corr = fourier_shifts.correlate(signal1, signal2)

        if max_diff is not None:
            # Apply a Gaussian filter to the correlation
            window = gaussian(len(corr), std=max_diff)
            corr = corr * window
        
        # Find the peak location of the correlation
        max_corr_idx = np.argmax(np.abs(corr))
        
        # Phase difference calculation
        phase_diff = np.angle(cross_corr[max_corr_idx])
        
        # Frequency resolution
        n = len(signal1)
        freq = np.fft.fftfreq(n)

        # Sanity check for frequency
        if freq[max_corr_idx] == 0:
            # If the frequency is zero, return 0
            return 0
        
        # Calculate the shift using the phase difference at the peak frequency
        shift_estimate = -phase_diff / (2 * np.pi * freq[max_corr_idx])
        
        return shift_estimate

    @staticmethod
    def find_shift(arr1, arr2):
        """
        Find the shift between two arrays using standard cross-correlation and then refine it into sub-samples using Fourier Transform.
        Parameters
        ----------
        arr1 : np.ndarray
            The first array.
        arr2 : np.ndarray
            The second array (shifted version of arr1).
        Returns
        -------
        float
            The estimated shift.
        """

        # Compute the cross-correlation
        corr = np.correlate(arr1, arr2, mode='full')

        # Find the index of the maximum correlation
        max_corr_idx = np.argmax(corr)

        # Calculate the shift
        shift = max_corr_idx - (len(arr1) - 1)

        # Shift the array to align with the first array
        arr2_shifted = np.roll(arr2, -shift)

        # Fine-tune the shift using Fourier Transform
        shift_estimate = fourier_shifts.__fourier_find_shift(arr1, arr2_shifted)
        
        # print(f"Shift: {shift}")
        # print(f"Shift estimate: {shift_estimate}")

        return shift + shift_estimate

    @staticmethod
    def roll(arr, shift):
        """
        Roll the array using Fourier Transform with a non-integer shift.
        Parameters
        ----------
        arr : np.ndarray
            The input array to be shifted.
        shift : float
            The amount to shift the array. Can be non-integer.
        Returns
        -------
        np.ndarray
            The shifted array.
        """

        if shift == 0:
            return arr

        # Compute the Fourier Transform
        fft_arr = np.fft.fft(arr)
        # Apply the Fourier shift
        shifted_fft = fourier_shift(fft_arr, shift)
        # Inverse Fourier Transform
        shifted_arr = np.fft.ifft(shifted_fft)

        return np.real(shifted_arr)