import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ..utils.utils import utils
from ..utils.data_quality import data_quality_utils

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
                this_mean = np.nanmean(data[i_sub, i_pol, i_freq])
                this_std = np.nanstd(data[i_sub, i_pol, i_freq])
                if this_std != 0:
                    data[i_sub, i_pol, i_freq] -= (this_mean)
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

@dataclass
class StackedDataMetadata:
    freq_lo: float
    freq_hi: float
    dm: float
    period: float
    model: str
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

        # Validate the stacked data
        self.validate()

        # Validate metadata
        self.meta.validate()

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

    def __generate_freqs(self):
        self.freqs = np.linspace(
            self.meta.freq_lo,
            self.meta.freq_hi,
            self.data.shape[2]
        )

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

    def get_metadata(self):
        return self.meta

    def get_subint_snrs(self):
        if self.data.shape[0] <= 1:
            return []
        
        return [
            data_quality_utils.boxcar_snr(subint) for subint in self.get_data(
                scrunch=(1, 2), keepdims=True
            )[:, 0, 0, :]
        ]

    def get_stacked_snr(self):
        if self.data.shape[3] <= 1:
            return 0.0

        return data_quality_utils.boxcar_snr(
            self.get_data(scrunch=(0, 1, 2), keepdims=True)[0, 0, 0, :]
        )

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

        for axis, factor in downsample_by_factor:
            self_copy.downsample_by_factor(factor, axis)

        for axis, length in downsample_to_length:
            self_copy.downsample_to_length(length, axis)

        for axis in scrunch:
            self_copy.scrunch(axis)

        if normalize:
            self_copy.normalize()

        if not keepdims:
            self_copy.data = np.squeeze(self_copy.data)
            
        return self_copy.data

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
        