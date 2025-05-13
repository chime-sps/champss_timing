import matplotlib.pyplot as plt
import psrchive
import copy
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, minimum_filter, maximum_filter
from multiprocessing import Pool

from ..utils.logger import logger

class _Filters():
    def __init__(self, ar, logger=logger()):
        self.archive = psrchive.Archive_load(ar)
        self.subint = self.archive.get_Integration(0)
        self.prof = self.subint.get_Profile(0,0)
        self.amp = self.get_amp()
        self.logger = logger

    def set_amps(self, new_amp):
        self.logger.debug("Overwriting profile with new time series")

        # check if new time series is the same length as the profile
        if np.shape(new_amp) != np.shape(self.amp):
            raise ValueError("Time series must be the same length as the profile")

        # overwrite the profile with the new time series
        for i, _ in enumerate(self.amp):
            for j, _ in enumerate(self.amp[i]):
                for k in range(len(self.amp[i][j])):
                    self.amp[i][j][k] = new_amp[i][j][k]

    def get_amp(self):
        amp = []
        for i in range(self.subint.get_npol()):
            amp.append([])
            for j in range(self.subint.get_nchan()):
                amp[i].append(self.subint.get_Profile(i,j).get_amps())

        return amp
    
    def amp_copy(self):
        return copy.deepcopy(self.amp)

    def plot(self, i_pol, i_chan, savefig=False):
        plt.plot(self.amp[i_pol][i_chan])
        if savefig:
            plt.savefig(f"{self.archive.get_filename}_profile_{i_chan}_{i_pol}.png")
        else:
            plt.show()

    def save(self, filename=None):
        if filename is None:
            filename = self.archive.get_filename() + ".ctp_filtered"
        self.archive.unload(filename)
        self.logger.debug(f"Filtered profile saved to {filename}")

    def _champss_filter(self, timeseries, pulse_width, meth="min"):
        """
        Apply a filter to saturated CHAMPSS observations.
        
        timeseries : array-like
            The input time series.
        pulse_width : int
            The width of the pulse to be removed.
        meth : str
            The filter to use. Must be 'min' or 'med'.

        Returns the filtered time series
        """

        # minimal filter
        if meth == "min" or meth == "minimum":
            mask = minimum_filter(timeseries, size=int(1.5 * pulse_width))
        elif meth == "med" or meth == "median":
            mask = median_filter(timeseries, size=int(1.5 * pulse_width))
        else:
            raise ValueError("meth must be 'min' or 'med'")

        # smooth with gaussian filter
        mask_smoothed = gaussian_filter(mask, sigma=pulse_width/2)

        return timeseries - mask_smoothed
    
    def _get_pulse_width(self, std_prof_ar):
        # get pulse width from a standard profile
        std_prof = psrchive.Archive_load(std_prof_ar)
        std_prof_subint = std_prof.get_Integration(0)
        std_prof_prof = std_prof_subint.get_Profile(0,0)
        std_prof_amps = std_prof_prof.get_amps()

        # get half max
        half_max = np.max(std_prof_amps) / 2
        
        # find the pulse width bt FWHM
        left = np.where(std_prof_amps > half_max)[0][0]
        right = np.where(std_prof_amps > half_max)[0][-1]
        pulse_width = (right - left) * 4

        return pulse_width

    def champss_filter(self, std_prof, meth="min"):
        """
        Apply a filter to saturated CHAMPSS observations.
        
        pulse_width : int
            The width of the pulse to be removed.
        meth : str
            The filter to use. Must be 'min' or 'med'.

        Returns the filtered time series
        """

        pulse_width = self._get_pulse_width(std_prof)
        amp_filtered = self.amp_copy()
        self.logger.debug(f"Filtering with CHAMPSS filter (pulse_width={pulse_width} and meth={meth})")

        for i, _ in enumerate(amp_filtered):
            for j, _ in enumerate(amp_filtered[i]):
                amp_filtered[i][j] = self._champss_filter(amp_filtered[i][j], pulse_width, meth)

        return self.set_amps(
            amp_filtered
        )

class Filters():
    def __init__(self, ars, outfiles=None, std_profile=None, n_pools=4, filters=[], logger=logger()):
        self.ars = ars
        self.outfiles = outfiles
        self.std_profile = std_profile
        self.filters = filters
        self.n_pools = n_pools
        self.logger = logger

        if self.outfiles is None:
            self.logger.info("No output file specified. Overwriting input files.")
            self.outfiles = ars

    def _filter(self, i):
        if len(self.filters) == 0:
            return
        
        filters_hdl = _Filters(self.ars[i], self.logger.copy())
        for this_filter in self.filters:
            if this_filter == "champss":
                if self.std_profile is None:
                    raise ValueError("Must specify a standard profile for CHAMPSS filter")
                filters_hdl.champss_filter(self.std_profile)
            else:
                raise ValueError("Unknown filter", self.filters)

        filters_hdl.save(self.outfiles[i])

    def filter(self):
        with Pool(self.n_pools) as p:
            p.map(self._filter, range(len(self.ars)))
        self.logger.debug(f"Filtering completed (n_files={len(self.ars)}, filters={self.filters})")