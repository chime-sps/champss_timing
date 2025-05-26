import numpy as np
from scipy.optimize import curve_fit, minimize
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

        return -shift

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
        
        return np.real(corr)

    @staticmethod
    def find_shift(arr1, arr2):
        """
        Find the shift between two arrays using standard cross-correlation and then refine it into sub-samples using Fourier Transform.
        Codes adopted from DMFITTER template_match: https://github.com/quantumfx/dmfitter/blob/master/dmfitter.py (Fang Xi Lin, Rob Main, 2019)
        Reference: Pulsar Timing and Relativistic Gravity, Taylor 1992, appendix A
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

        def chi_check(dt,z_f,z_var,pp_f,freqs,ngates):
            a = np.sum( (z_f*pp_f.conj()*np.exp(1j*2*np.pi*freqs*dt) + z_f.conj()*pp_f*np.exp(-1j*2*np.pi*freqs*dt))[1:] ) / np.sum( 2 * np.abs(pp_f[1:])**2 )
            num = (z_f - (a * pp_f * np.exp(-1j*2*np.pi*freqs*dt)))[1:] # consider k>0
            chisq = np.sum(np.abs(num)**2) / (z_var * ngates/2)
            return chisq

        # Get trials
        t = np.linspace(0, 1, len(arr1), endpoint=False)
        t += (t[1] - t[0]) / 2
        ppdt = t[1] - t[0]        

        # FFT of both arrays
        fft_arr1 = np.fft.rfft(arr1)
        fft_arr2 = np.fft.rfft(arr2)

        # Get frequency bins
        freqs = np.fft.rfftfreq(len(arr1), ppdt)

        # Calculate the variance
        arr2_var = np.sum(np.var(arr2[arr2 < np.median(arr2)]))

        # Find the initial shift
        profcorr = np.fft.irfft((fft_arr2 * fft_arr1.conj())[1:])
        x = np.fft.fftfreq(len(arr1))
        xguess = x[np.argmax(np.abs(profcorr))]

        # Minimize the chi-square
        minchisq = minimize(chi_check, x0=xguess, args=(fft_arr2, arr2_var, fft_arr1, freqs, len(arr1)), method="Nelder-Mead")
        if minchisq.success != True:
            print('Chi square minimization failed to converge. !!BEWARE!!')

        # Get the best fit
        best_shift = np.real(minchisq.x[0]) * len(arr1)

        return best_shift

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

class subsample_shifts():
    """
    Cross-correlate utilities based on discrete shifts with sub-sample accuracy.
    """

    correlate = discrete_shifts.correlate
    roll = fourier_shifts.roll

    @staticmethod
    def __correlate(arr1, arr2, range=None, step=0.1):
        """
        Compute the cross-correlation between two arrays with sub-sample accuracy.
        Parameters
        ----------
        arr1 : np.ndarray
            The first array.
        arr2 : np.ndarray
            The second array.
        range : tuple, optional
            The range of shifts to consider (default is None).
        step : float, optional
            The step size for the shifts (default is 0.1).
        Returns
        -------
        np.ndarray
            The cross-correlation of the two arrays.
        """
        
        # Get the range
        if range is None:
            range = [0, len(arr1)]

        # Create the array of shifts
        shifts = np.arange(range[0], range[1], step)

        # Correlate
        corr = np.zeros(len(shifts))
        for i, shift in enumerate(shifts):
            corr[i] = subsample_shifts.roll(arr1, shift).dot(arr2)

        return corr, shifts

    @staticmethod
    def find_shift(arr1, arr2):
        """
        Find the shift between two arrays using standard cross-correlation and then refine it into sub-samples using fitting.
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

        def hyperbola(x, a, b, c):
            return - a * (x - b) ** 2 + c

        def hyperbola_peak(a, b, c):
            return b
        
        # Find the initial shift
        init_shift = discrete_shifts.find_shift(arr1, arr2)

        # Compute the subsample cross-correlation
        refined_corr, refined_shifts = subsample_shifts.__correlate(arr1, arr2, range=[init_shift - 1, init_shift + 1], step=0.01)

        # Fit for the hyperbola
        popt, pcov = curve_fit(hyperbola, refined_shifts, refined_corr)
        
        # import matplotlib.pyplot as plt
        # fitted_curve = hyperbola(refined_shifts, *popt)
        # plt.plot(refined_shifts, refined_corr)
        # plt.plot(refined_shifts, fitted_curve)
        # plt.plot(refined_shifts, fitted_curve - refined_corr)
        # return

        # Find the maximum of the hyperbola
        refined_shift = hyperbola_peak(*popt)

        return refined_shift