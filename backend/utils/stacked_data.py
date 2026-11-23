import copy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from dataclasses import dataclass

from ..utils.utils import utils
from ..utils.data_quality import data_quality_utils
from ..utils.correlation import fourier_shifts

K_DM = 1.0 / 2.41e-4 # use the DM constant consistent with PINT

def normalize_data(data):
    """
    Normalize the data by dividing each subint, pol, and freq by its mean and standard deviation (i.e., z-score).
    Parameters:
        data (np.ndarray): The data to normalize.
    Returns:
        np.ndarray: The normalized data.
    """

    for i_sub in range(data.shape[0]):
        for i_pol in range(data.shape[1]):
            for i_freq in range(data.shape[2]):
                this_median = np.nanmedian(data[i_sub, i_pol, i_freq])
                this_std = np.nanstd(data[i_sub, i_pol, i_freq])
                if this_std != 0:
                    data[i_sub, i_pol, i_freq] -= (this_median)
                    data[i_sub, i_pol, i_freq] /= this_std
                    # data[i_sub, i_pol, i_freq] /= this_mean
                else:
                    data[i_sub, i_pol, i_freq] = np.zeros_like(data[i_sub, i_pol, i_freq])

    return data

def downsample(arr, axis, n, method="mean"):
    # Move the axis to be downsample to the end
    arr = np.moveaxis(arr, axis, -1)

    # Find the length after downsampling
    m = arr.shape[-1] // n

    # Reshape to separate the downsampled blocks
    arr = arr[..., :m * n].reshape(*arr.shape[:-1], m, n)

    # Sum along the downsampled blocks
    if method == "mean":
        arr = arr.mean(axis=-1)
    elif method == "sum":
        arr = arr.sum(axis=-1)
    else:
        raise ValueError(f"Unsupported downsample method: {method}")

    # Move the axis back to its original position
    arr = np.moveaxis(arr, -1, axis)

    return arr


def compute_dispersive_delay(freq, ddm, ref_freq, k_dm=K_DM):
    """
    Calculate the time delay for a given frequency and DM step.

    Parameters
    ----------
    freq : float
        Frequency of the channel.
    ddm : float
        DM step.

    Returns
    -------
    float
        Time delay for the given frequency and DM step.

    Returns
    -------
    float
        Time delay for the given frequency and DM step.
    """
    
    return k_dm * ddm * (freq ** -2 - ref_freq ** -2)

@dataclass
class StackedDataMetadata:
    freq_lo: float
    freq_hi: float
    dm: float
    period: float
    model: str
    ref_epoch: float
    resample_phase_bins: bool
    remove_baseline: bool
    input_files: list

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def validate(self):
        """Validate the metadata."""

        for f in self.input_files:
            for k in ["location", "subint_epochs", "snr"]:
                if k not in f:
                    raise ValueError(f"Input file is missing required key: {k}")

        pass

class StackedData:
    def __init__(
        self, 
        data, 
        epochs,
        expos,
        meta
    ):
        # Initialize stacked data
        self.data = np.asarray(data, dtype=np.float32)
        self.epochs = np.asarray(epochs)
        self.expos = np.asarray(expos)
        self.meta = meta

        # Generate frequency channels
        self.freqs = None
        self.__generate_freqs()

        # Validate
        self.validate()

    @staticmethod
    def load(stackfile, format=None):
        """Load a StackedData instance from a file."""

        # Determine the file format if not explicitly provided
        if format is None:
            if stackfile.endswith(".npz"):
                format = "npz"
            elif stackfile.endswith(".pkl"):
                format = "pkl"
            elif stackfile.endswith(".json"):
                format = "json"
            else:
                raise ValueError("Unable to determine file format from filename. Please specify the format explicitly.")

        if format == "npz":
            data = np.load(stackfile, allow_pickle=True)
            return StackedData(
                data=data["data"],
                epochs=data["epochs"],
                expos=data["expos"],
                meta=StackedDataMetadata(**data["meta"].item())
            )
        elif format == "pkl":
            import pickle
            with open(stackfile, "rb") as f:
                data = pickle.load(f)
            return StackedData(
                data=np.array(data["data"]),
                epochs=np.array(data["epochs"]),
                expos=np.array(data["expos"]),
                meta=StackedDataMetadata(**data["meta"])
            )
        elif format == "json":
            import json
            with open(stackfile, "r") as f:
                data = json.load(f)
            return StackedData(
                data=np.array(data["data"]),
                epochs=np.array(data["epochs"]),
                expos=np.array(data["expos"]),
                meta=StackedDataMetadata(**data["meta"])
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def validate(self):
        """Validate the stacked data."""

        # Shape
        if self.data.ndim != 4:
            raise ValueError("Data must be a 4-dimensional array (subints, pols, freqs, n_bins).")

        # Num of subints == length of epochs
        if self.data.shape[0] != len(self.epochs):
            raise ValueError("Number of subints must match the length of epochs.")
        
        # Num of subints == length of expos
        if self.data.shape[0] != len(self.expos):
            raise ValueError("Number of subints must match the length of expos.")

        # Frequency axis
        if self.data.shape[2] != len(self.freqs):
            raise ValueError("Number of frequency channels must match the length of freqs.")

        # Type of metadata
        if not isinstance(self.meta, StackedDataMetadata):
            raise ValueError("Metadata must be an instance of StackedDataMetadata.")

        # Validate metadata
        self.meta.validate()

    def __generate_freqs(self):
        self.freqs = np.linspace(
            self.meta.freq_lo,
            self.meta.freq_hi,
            self.data.shape[2]
        )

    ####################################
    # Downsampling and resizing
    ####################################

    def normalize(self):
        """Normalize the data."""
        self.data = normalize_data(self.data)

    def scrunch(self, axis):
        """Scrunch (average) the data along the specified axis."""

        if axis < 0 or axis >= self.data.ndim:
            raise ValueError("Invalid axis for scrunching.")

        self.data = self.data.mean(axis=axis, keepdims=True)

        if axis == 0:  # Subint axis
            self.epochs = self.epochs.mean(keepdims=True)
            self.expos = self.expos.sum(keepdims=True)

        if axis == 2:  # Frequency axis
            self.__generate_freqs()

        self.validate()

    def downsample(self, factor, axis):
        """Downsample the data by a given factor along the specified axis."""

        if axis < 0 or axis >= self.data.ndim:
            raise ValueError("Invalid axis for downsampling.")

        if factor <= 0:
            raise ValueError("Downsample factor must be a positive integer.")

        if factor > self.data.shape[axis]:
            raise ValueError("Downsample factor cannot be greater than the size of the axis.")

        self.data = downsample(self.data, axis, factor, method="sum")

        if axis == 0:  # Subint axis
            self.epochs = downsample(self.epochs, 0, factor, method="mean")
            self.expos = downsample(self.expos, 0, factor, method="sum")

        if axis == 2:  # Frequency axis
            self.__generate_freqs()

        self.validate()

    def downsample_by_factor(self, factor, axis):
        """Downsample the data by a given factor along the specified axis."""

        self.downsample(factor, axis)
                             
    def downsample_to_length(self, length, axis):
        """Shortcut to downsample the data to a specific length along the given axis."""

        self.downsample_by_factor(
            factor=self.data.shape[axis] // length,
            axis=axis
        )

    def drop_partial_subints(self, threshold=0.5):
        """Drop subintegrations that are partially filled."""

        if self.data.shape[0] <= 1:
            return

        # Calculate the median exposure
        median_exposure = np.median(self.expos)

        # Find subints that are significantly below the median exposure
        good_i = self.expos >= median_exposure * threshold

        # Find start and end indices to only mask out subints at the beginning and end of the observation
        good_i = np.flatnonzero(good_i)
        if len(good_i) == 0:
            return
        start = good_i[0]
        end = good_i[-1] + 1

        self.data = self.data[start:end, :, :, :]
        self.epochs = self.epochs[start:end]
        self.expos = self.expos[start:end]

        self.validate()

    ####################################
    # Getting information
    ####################################

    def get_metadata(self):
        return self.meta

    def get_shape(self):
        return self.data.shape

    def get_subint_snrs(self):
        if self.data.shape[0] <= 1:
            return []
        
        return [
            data_quality_utils.boxcar_snr(subint) for subint in self.get_data(
                scrunch=(1, 2), keepdims=True
            )[:, 0, 0, :]
        ]
    
    def get_subint_metric(self, metric):
        """Get the subintegration metric for the given metric type."""

        if self.data.shape[3] <= 1:
            return 0.0

        if metric == "snr":
            method = data_quality_utils.boxcar_snr
        elif metric == "chi2":
            method = data_quality_utils.chi2
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return method(
            self.get_data(scrunch=(0, 1, 2), keepdims=True)[0, 0, 0, :]
        )

    def get_stacked_snr(self):
        """Shortcut to get the stacked SNR."""
        return self.get_subint_metric("snr")

    def get_subint_chi2(self):
        """Shortcut to get the subintegration chi-squared."""
        return self.get_subint_metric("chi2")

    def get_stacked_metric(self, metric):
        """Get the stacked metric for the given metric type."""

        if self.data.shape[3] <= 1:
            return 0.0

        if metric == "snr":
            method = data_quality_utils.boxcar_snr
        elif metric == "chi2":
            method = data_quality_utils.chi2
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return method(
            self.get_data(scrunch=(0, 1, 2), keepdims=True)[0, 0, 0, :]
        )
    
    def get_stacked_chi2(self):
        """Shortcut to get the stacked chi-squared."""
        return self.get_stacked_metric("chi2")

    def get_stacked_snr(self):
        """Shortcut to get the stacked SNR."""
        return self.get_stacked_metric("snr")

    def get_period(self):
        return self.meta.period

    def get_dm(self):
        return self.meta.dm

    def get_data(
            self, 
            downsample_by_factor=(), 
            downsample_to_length=(), 
            scrunch=(), 
            normalize=False, 
            keepdims=True
        ):
        """
        Get the data with optional processing applied.

        Args:
            downsample_by_factor (tuple): List of (axis, factor) pairs to downsample by.
            downsample_to_length (tuple): List of (axis, length) pairs to downsample to a specific length.
            scrunch (tuple): List of axes to scrunch (average) over.
            normalize (bool): Whether to normalize the data.
            keepdims (bool): Whether to keep the dimensions after processing.

        Returns:
            np.ndarray: The processed data.
        """
        
        self_copy = copy.deepcopy(self)

        if normalize:
            self_copy.normalize()

        for axis, factor in downsample_by_factor:
            self_copy.downsample_by_factor(factor, axis)

        for axis, length in downsample_to_length:
            self_copy.downsample_to_length(length, axis)

        for axis in scrunch:
            self_copy.scrunch(axis)

        if not keepdims:
            self_copy.data = np.squeeze(self_copy.data)
            
        return self_copy.data

    def get_profile(self, normalize=False):
        """Helper function to get pulse profile."""
        return self.get_data(scrunch=(0, 1, 2), keepdims=True, normalize=normalize)[0, 0, 0, :]

    def get_waterfall(self, normalize=False):
        """Helper function to get the waterfall data."""
        return self.get_data(scrunch=(0, 1), keepdims=True, normalize=normalize)[0, 0, :, :]

    def get_subints(self, normalize=False):
        """Helper function to get the subintegration data."""
        return self.get_data(scrunch=(1, 2), keepdims=True, normalize=normalize)[:, 0, 0, :]

    ####################################
    # Dedispersion and dealiasing
    ####################################

    def __calc_dispersion_dt(self, freq, ddm): 
        # Use the middle of the band as the reference frequency for DM correction
        ref_freq = (self.meta.freq_lo + self.meta.freq_hi) / 2
        
        return compute_dispersive_delay(freq, ddm, ref_freq)

    def refine_dm(self, ddm):
        """
        Dedisperse the data array by applying the specified dDM step.

        Parameters
        ----------
        ddm : float
            DM step to apply.
        """

        if ddm == 0:
            return

        # Apply the dDM steps
        for i, freq in enumerate(self.freqs):
            self.data[:, :, i, :] = fourier_shifts.roll(
                self.data[:, :, i, :], 
                shift=-self.__calc_dispersion_dt(freq, ddm) / self.meta.period * self.data.shape[3], # in phase bins
                axis=-1 # in phase axis
            )
            
        self.meta.dm += ddm # update ddm to metadata

        # Validate
        self.validate()

    def dedisperse(self, dm):
        """Shortcut to refine the DM to the specified value."""
        self.refine_dm(dm - self.meta.dm)

    def dededisperse(self):
        """Shortcut to dedisperse with dDM = - folding DM."""
        self.refine_dm(-self.meta.dm)

    def __calc_aliasing_drifts(self, alias_factor):
        """
        Calculate the expected aliasing drifts and change in period due to aliasing.

        Parameters
        ----------
        alias_factor : float
            The aliasing factor.

        Returns
        -------
        tuple
            A tuple containing the expected shifts (numpy array) and the change in period (float).
        """
        
        if alias_factor == 0:
            return np.zeros_like(self.epochs), 0
        
        # Get sidereal day from astropy
        sidereal_day = (1 * u.sday).to(u.day).value

        # Compute slope in the unit of phase/day
        slope = 1 / (sidereal_day / alias_factor)

        # Compute shifts
        shifts = slope * self.epochs

        # Compute change in period due to aliasing
        dp = self.meta.period / ((u.sday.to(u.s) * (1/alias_factor)) / self.meta.period)

        return shifts, dp

    def dealiasing(self, alias_factor):
        """
        Correct the data for aliasing given an aliasing factor.

        Parameters
        ----------
        alias_factor : float
            The aliasing factor to apply.
        """
        
        if alias_factor == 0:
            return

        if not alias_factor.is_integer():
            raise ValueError(f"Alias factor must be an integer: got {alias_factor}")
        
        # Calculate expected shifts
        this_expected_shifts, dp = self.__calc_aliasing_drifts(alias_factor)

        # Apply correction of the data
        for i, shift in enumerate(this_expected_shifts):
            self.data[i, :, :, :] = fourier_shifts.roll(
                self.data[i, :, :, :],
                shift=-shift * self.data.shape[3], # in phase bins
                axis=-1 # in phase axis
            )

        self.meta.period += dp # update period due to aliasing

        # Validate after dealiasing
        self.validate()

    ####################################
    # IO
    ####################################

    def save(self, filename, format=None):
        """
        Save the stacked data and metadata to a file.

        Args:
            filename (str): The name of the file to save the data to.
            format (str): The format to save the data in (options: npz, pkl, json).
        """

        if format is None:
            if filename.endswith(".npz"):
                format = "npz"
            elif filename.endswith(".pkl"):
                format = "pkl"
            elif filename.endswith(".json"):
                format = "json"
            else:
                raise ValueError("Cannot infer format from filename. Please specify the format explicitly.")

        # Put together payload
        payload = {"data": self.data, "epochs": self.epochs, "expos": self.expos, "meta": self.get_metadata().to_dict()}

        # Save the data and metadata
        if format == "npz":
            np.savez(filename, **payload)
        elif format == "pkl":
            with open(filename, "wb") as f:
                import pickle
                pickle.dump(utils.numpy_to_native(payload), f)
        elif format == "json":
            with open(filename, "w") as f:
                import json
                json.dump(utils.numpy_to_native(payload), f)
        else: 
            raise ValueError(f"Unsupported format: {format}. Supported formats are: npz, pkl, json.")

    def plot(self, savefig=None):
        """
        Plot the stacked data, including frequencies, subints, exposures, and subint epochs of input files.

        Args:
            savefig (str): If provided, save the figure to the specified file.
        """
        
        # Convert exposures to seconds
        epochs_seconds = self.epochs * 86400
        expos_seconds = self.expos * 86400

        # Generate phase bins
        phase = np.linspace(0, 1, self.data.shape[3])

        fig, ax = plt.subplots(4, 1, constrained_layout=True, figsize=(10, 10))

        ax[0].pcolormesh(
            self.freqs, 
            phase, 
            self.get_data(scrunch=(0, 1), normalize=True, keepdims=True)[0, 0, :, :].T, 
            shading='auto', 
            cmap='gray_r'
        )
        ax[0].set_title('Frequencies')
        ax[0].set_ylabel('Phase Bin')
        ax[0].set_xlabel('Frequency (MHz)')
        ax[0].invert_yaxis()

        ax[1].pcolormesh(
            epochs_seconds, 
            phase, 
            self.get_data(scrunch=(1, 2), normalize=True, keepdims=True)[:, 0, 0, :].T, 
            shading='auto', 
            cmap='gray_r'
        )
        ax[1].set_title('Subints')
        ax[1].set_ylabel('Phase Bin')
        ax[1].set_xlabel('Time since transit (s)')
        ax[1].invert_yaxis()

        ax[2].plot(
            epochs_seconds, 
            expos_seconds, 
            "kx"
        )
        ax[2].set_title('Exposures')
        ax[2].set_ylabel('Exposure (s)')
        ax[2].set_xlabel('Time since transit (s)')
        ax[2].set_xlim(ax[1].get_xlim())

        for i, f in enumerate(self.meta.input_files):
            ax[3].plot(f["subint_epochs"] * 86400, [i] * len(f["subint_epochs"]), "k-", ms=3, marker='|')
        
        ax[3].set_title('Subint Epochs of Input Files')
        ax[3].set_xlabel('Time since transit (s)')
        ax[3].set_ylabel('File Index')
        ax[3].set_xlim(ax[1].get_xlim())

        bin_sizes = np.diff(epochs_seconds)
        bin_edges = np.cumsum(bin_sizes) - bin_sizes / 2 + np.min(epochs_seconds)
        for bin_edge in bin_edges:
            ax[2].axvline(bin_edge, color='gray', alpha=0.3, linestyle='-', lw=0.5)
            ax[3].axvline(bin_edge, color='gray', alpha=0.3, linestyle='-', lw=0.5)

        if savefig:
            plt.savefig(savefig)
        plt.show()

    def copy(self):
        return copy.deepcopy(self)