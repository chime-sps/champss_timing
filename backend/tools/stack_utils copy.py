import os
import gc
import copy
import tqdm
import shutil
import psrchive
import traceback
import numpy as np
from collections import Counter
from scipy.ndimage import zoom
from dataclasses import dataclass

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

from ..utils.utils import utils
from ..utils.logger import logger
from ..utils.data_quality import data_quality_utils
from ..processing.dspsr_shutils import dspsr_shutils
from ..processing.archive_shutils import archive_shutils

def _normalize_data(data):
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

def _resize_array(a, new_shape):
    """
    Resize an N-dimensional numpy array to a new shape using interpolation.

    Parameters:
        a (np.ndarray): The array to resize.
        new_shape (tuple): The desired shape (must be same number of dimensions as input).

    Returns:
        np.ndarray: Resized array.
    """
    
    # Make sure new_shape is a tuple of integers
    new_shape = tuple(int(n) for n in new_shape)

    # Skip resizing if the input array already has the desired shape
    if a.shape == new_shape:
        return a

    # Downsample each axis if the new size is smaller and an integer factor of the current size
    out = copy.deepcopy(a)
    for ax, (n_in, n_out) in enumerate(zip(a.shape, new_shape)):
        if n_out < n_in and n_in % n_out == 0:
            k = n_in // n_out
            shp = out.shape[:ax] + (n_out, k) + out.shape[ax + 1:]
            out = out.reshape(shp).mean(axis=ax + 1)

    if out.shape != new_shape: # Use scipy's built-in resize function to upsample or handle non-integer cases
        out = zoom(out, np.array(new_shape) / np.array(out.shape), order=1)

    return out


def _resize_psrchive(ar_obj, new_shape):
    """
    Resize a PSRCHIVE archive to a new shape using PSRCHIVE's built-in scrunch function.
    Parameters:
        ar_obj (psrchive.Archive): The PSRCHIVE archive object to resize.
        new_shape (tuple): The desired shape (must be same number of dimensions as input).
    Returns:
        ar_obj (psrchive.Archive): The resized PSRCHIVE archive object.
    """

    # Get shapes
    # data_n_subs, data_n_pols, data_n_freqs, data_n_bins = ar_obj.get_data().shape
    data_n_subs, data_n_pols, data_n_freqs, data_n_bins = ar_obj.get_nsubint(), ar_obj.get_npol(), ar_obj.get_nchan(), ar_obj.get_nbin()
    new_n_subs, new_n_pols, new_n_freqs, new_n_bins = new_shape
    # subints, pols, freqs, bins

    # Scrunch pols
    if new_n_pols <= 1:
        ar_obj.pscrunch() # it's only physically make sense to scrunch into 1 pol 
        
    # Scrunch subints
    if new_n_subs == 1:
        ar_obj.tscrunch()
    elif new_n_subs < data_n_subs:
        ar_obj.tscrunch_to_nsub(new_n_subs)

    # Scrunch freqs
    if new_n_freqs == 1:
        ar_obj.fscrunch()
    elif new_n_freqs < data_n_freqs:
        ar_obj.fscrunch_to_nchan(new_n_freqs)

    # Scrunch bins
    if new_n_bins < data_n_bins:
        ar_obj.bscrunch_to_nbin(new_n_bins)
    
    return ar_obj

def _stack_worker(args):
    """Worker function to stack the data from the input archive."""

    this_outfile = ""
    this_shape = None
    this_duration = None
    this_subint_durations = None
    this_snr = None
    freq_lo = None
    freq_hi = None

    # Unpack arguments
    task_i, file_info, config, logger = args

    this_workspace = None
    this_location = None
    try:
        # Check if the file info is valid
        if "location" not in file_info or "backend" not in file_info:
            raise ValueError("File info does not contain location or backend information.")
        
        # read file info and jump
        this_location = file_info["location"]
        this_backend = file_info["backend"]
        this_jump = 0
        if this_backend in config["jumps"]:
            this_jump = config["jumps"][this_backend][0]
        
        # create workspace for this archive
        this_workspace = os.path.join(config["tempdir"], f"{task_i:06d}__" + os.path.basename(this_location))
        this_outfile = os.path.join(config["tempdir"], f"stackfile__{task_i:06d}__" + os.path.basename(this_location) + ".npy")
        this_ar = this_workspace + "/data"
        this_par = this_workspace + "/parfile"
        if not os.path.exists(this_workspace):
            os.makedirs(this_workspace)
            
        # prepare the archive
        shutil.copy(this_location, this_ar)
        shutil.copy(config["parfile"], this_par)

        # convert filterbank data as needed
        if this_location.endswith(".fil"):
            this_fil = this_workspace + "/data.fil"
            
            # copy data to workspace
            shutil.move(this_ar, this_fil)

            # convert to ar using dspsr
            du = dspsr_shutils(n_pools=1)
            du.fil2ar(this_fil, this_ar, this_par, nbin=config["n_bins"])

            # rename converted ar
            shutil.move(this_ar + ".ar", this_ar)

            # remove the filterbank data
            os.remove(this_fil)
            del this_fil
        
        # Note: there's a weird bug that psrchive.Archive_load never release its memory 
        # until the end of the subprocess, even after gc.collect(). Here we use 
        # archive_shutils instead since it uses the cli interface that immediately releases
        # memory after the operation is completed. We want to use psrchive.Archive_load 
        # as later as we can to avoid it consuming too much memory early.

        # initialize archive shutils
        ashu = archive_shutils(this_ar)

        # scrunch polarization before clfd if required to save memory
        if config["n_pols"] == 1:
            ashu.scrunch(pol=True)

        # zap rfi
        ashu.clfd()

        # install parfile
        ashu.install_parfile(this_par, jump=this_jump)

        # load the archive
        this_arch = psrchive.Archive_load(this_ar)

        # dedisperse
        this_arch.dedisperse()

        # remove baseline if needed
        if config["remove_baseline"]:
            this_arch.remove_baseline()

        # downsample using PSRCHIVE scrunch
        _resize_psrchive(this_arch, (config["n_subs"], config["n_pols"], config["n_freqs"], config["n_bins"]))

        # get duration
        this_duration = (this_arch.end_time() - this_arch.start_time()).in_days()

        # get subint durations
        this_subint_durations = []
        for i in range(this_arch.get_nsubint()):
            integ = this_arch.get_Integration(i)
            duration = integ.get_duration() / 86400 # to days for consistency
            this_subint_durations.append(duration)

        # get frequency information
        freq_lo = this_arch.get_centre_frequency() - this_arch.get_bandwidth()/2.0
        freq_hi = this_arch.get_centre_frequency() + this_arch.get_bandwidth()/2.0

        # get the data
        this_data = this_arch.get_data() # (n_subs, n_pols, n_freqs, n_bins)

        # get the weights
        this_weights = this_arch.get_weights() # (n_subs, n_freqs)

        # apply weights to the data
        this_data *= this_weights[:, None, :, None]

        # use float32 for memory efficiency
        this_data = this_data.astype(np.float32)

        # free the archive from memory
        del this_arch
        gc.collect()

        # get_snr
        this_profile = np.sum(this_data, axis=(0, 1, 2))
        this_snr = data_quality_utils.boxcar_snr(this_profile)

        # get the shape of the data
        this_shape = this_data.shape

        # save data to tempdir
        np.save(this_outfile, this_data)
        
    except Exception:
        logger.warning(f"Thread {this_location} failed. Please refer to the traceback below")
        logger.warning(traceback.format_exc())
    finally:
        # remove the workspace
        if this_workspace is not None:
            shutil.rmtree(this_workspace, ignore_errors=True)

    return {
        "outfile": this_outfile, 
        "shape": this_shape, 
        "duration": this_duration,
        "subint_durations": this_subint_durations,
        "snr": this_snr, 
        "freq_lo": freq_lo, 
        "freq_hi": freq_hi, 
        "location": this_location
    }

@dataclass
class StackedDataMetadata:
    n_subs: int
    n_pols: int
    n_freqs: int
    n_bins: int
    n_stacked: int
    durations: list
    snrs: list
    freq_lo: list
    freq_hi: list
    model: str
    interpolate: str
    remove_baseline: bool
    allow_inconsistent_freq: bool
    duration_tol: float
    input_files: list

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

class StackedData:
    def __init__(
        self, 
        stacked_data, 
        n_subs,
        n_pols,
        n_freqs,
        n_bins,
        n_stacked,
        durations,
        snrs, 
        freq_lo, 
        freq_hi,
        model, 
        interpolate,
        remove_baseline, 
        allow_inconsistent_freq,
        duration_tol,
        input_files
    ):
        # Initialize stacked data
        self.data = stacked_data
        self.meta = StackedDataMetadata(
            n_subs=n_subs,
            n_pols=n_pols,
            n_freqs=n_freqs,
            n_bins=n_bins,
            n_stacked=n_stacked,
            durations=durations,
            snrs=snrs,
            freq_lo=freq_lo,
            freq_hi=freq_hi,
            model=model,
            interpolate=interpolate,
            remove_baseline=remove_baseline,
            allow_inconsistent_freq=allow_inconsistent_freq,
            duration_tol=duration_tol,
            input_files=input_files
        )

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
                stacked_data=data["data"],
                **data["metadata"].item()
            )
        elif format == "pkl":
            import pickle
            with open(stackfile, "rb") as f:
                data = pickle.load(f)
            return StackedData(
                stacked_data=np.array(data["data"]),
                **data["metadata"]
            )
        elif format == "json":
            import json
            with open(stackfile, "r") as f:
                data = json.load(f)
            return StackedData(
                stacked_data=np.array(data["data"]),
                **data["metadata"]
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_data(self, fscrunch=False, tscrunch=False, normalize=False, keepdims=True):
        data_copy = copy.deepcopy(self.data)

        if fscrunch:
            data_copy = data_copy.mean(axis=2, keepdims=True)

        if tscrunch:
            data_copy = data_copy.mean(axis=0, keepdims=True)

        if normalize:
            data_copy = _normalize_data(data_copy)

        if not keepdims:
            data_copy = np.squeeze(data_copy)
            
        return data_copy

    def get_metadata(self):
        return self.meta

    def get_stacked_snr(self):
        if self.meta.n_bins <= 1:
            return 0.0

        return data_quality_utils.boxcar_snr(
            self.get_data().sum(axis=(0, 1, 2))
        )

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

        # Save the data and metadata
        if format == "npz":
            np.savez(filename, data=self.data, metadata=self.get_metadata().to_dict())
        elif format == "pkl":
            with open(filename, "wb") as f:
                import pickle
                pickle.dump(utils.numpy_to_native({"data": self.data, "metadata": self.get_metadata().to_dict()}), f)
        elif format == "json":
            with open(filename, "w") as f:
                import json
                json.dump(utils.numpy_to_native({"data": self.data, "metadata": self.get_metadata().to_dict()}), f)
        else: 
            raise ValueError(f"Unsupported format: {format}. Supported formats are: npz, pkl, json.")
        
class StackUtils:
    def __init__(self, files, parfile, n_subs=16, n_pols=1, n_freqs=1024, n_bins=1024, n_pools=4, jumps={}, remove_baseline=False, interpolate="never", allow_inconsistent_freq=False, duration_tol=1, workspace="/tmp", logger=logger()):
        """
        Initialize the stack_utils class.

        Parameters:
        files (list): List of archive files to be stacked.
        parfile (str): Parameter file.
        n_subs (int): Number of sub-integrations.
        n_pols (int): Number of polarizations.
        n_freqs (int): Number of frequency channels.
        n_bins (int): Number of phase bins.
        n_pools (int): Number of parallel pools.
        jumps (dict): Dictionary of jumps.
        remove_baseline (bool): Whether to remove baseline.
        interpolate (str): Interpolation mode ("always", "minimal", "never").
                           always: always interpolate the data to match the target shape.
                           minimal: only interpolate when necessary to match the target shape.
                           never: never interpolate the data.
                           minimal or never: may result in the data not matching the target shape.
                           Note: interpolation may lose some sensitivity or introduce artifacts in the data.
        allow_inconsistent_freq (bool): Whether to allow inconsistent frequency channels across files.
        duration_tol (float): Tolerance for duration mismatch in seconds.
        workspace (str): Workspace directory.
        logger (logger): Logger instance.
        """

        self.n_subs = n_subs
        self.n_pols = n_pols
        self.n_freqs = n_freqs
        self.n_bins = n_bins
        self.files = files
        self.parfile = parfile
        self.logger = logger
        self.n_pools = n_pools
        self.jumps = jumps
        self.remove_baseline = remove_baseline
        self.interpolate = interpolate
        self.allow_inconsistent_freq = allow_inconsistent_freq
        self.duration_tol = duration_tol
        self.tempdir = workspace + f"/champss_timing__stack_utils/{utils.get_time_string()}__{utils.get_rand_string()}"

        # Show warning when n_pols is not 1
        if n_pols != 1:
            raise NotImplementedError("Stacking polarizations with this tool may not be physically meaningful (n_pols != 1)!")

        # Make sure interpolation mode is valid
        if self.interpolate not in ["always", "minimal", "never"]:
            raise ValueError("interpolate must be one of 'always', 'minimal', or 'never'")

        if not os.path.exists(workspace):
            raise Exception(f"Workspace {workspace} does not exist. Please create it first.")

    @staticmethod
    def load(filename):
        """Shortcut to load a StackedData instance from a file."""
        return StackedData.load(filename)

    def __find_most_common(self, tuples):
        return Counter(tuples).most_common(1)[0][0]

    # def __stack(self):
    #     # Create worker config
    #     worker_payload = {
    #         "jumps": self.jumps,
    #         "tempdir": self.tempdir,
    #         "parfile": self.parfile,
    #         "remove_baseline": self.remove_baseline,
    #         "interpolate": self.interpolate,
    #         "n_subs": self.n_subs,
    #         "n_pols": self.n_pols,
    #         "n_freqs": self.n_freqs,
    #         "n_bins": self.n_bins,
    #         "logger": self.logger
    #     }
    #     # Prepare data
    #     tasks = [(i, file_info, worker_payload, self.logger.copy()) for i, file_info in enumerate(self.files)]
    #     with Pool(self.n_pools, maxtasksperchild=1) as pool:
    #         sf_meta = list(tqdm.tqdm(
    #             pool.imap(_stack_worker, tasks),
    #             total=len(tasks),
    #             desc="Preparing archives",
    #         ))

    #     # Unpack the results
    #     n_subs, n_pols, n_freqs, n_bins = [], [], [], []
    #     _res_shapes, _res_frequencies, _res_durations = [], [], []
    #     for res in sf_meta:
    #         if res["shape"] is None:
    #             continue
            
    #         n_subs.append(res["shape"][0])
    #         n_pols.append(res["shape"][1])
    #         n_freqs.append(res["shape"][2])
    #         n_bins.append(res["shape"][3])

    #         _res_shapes.append(res["shape"])
    #         _res_durations.append(res["duration"])
    #         _res_frequencies.append((res["freq_lo"], res["freq_hi"]))

    #     if len(n_subs) == 0:
    #         raise Exception("No valid stack files found.")


    #     # Find median duration
    #     median_duration = np.median(_res_durations)

    #     # Find most common shapes
    #     most_common_shape = self.__find_most_common(_res_shapes)

    #     # Find most common frequency range
    #     most_common_frequency = self.__find_most_common(_res_frequencies)

    #     # If always interpolate, override the most common shape with the target shape
    #     if self.interpolate == "always":
    #         most_common_shape = (self.n_subs, self.n_pols, self.n_freqs, self.n_bins)
    #         self.logger.debug(f"Interpolating all stack files to the most common shape {most_common_shape} since interpolate=always.")

    #     # Initialize the stacked data array
    #     stacked_data = np.zeros(most_common_shape, dtype=np.float32)

    #     # Stack
    #     n_stacked, durations, snrs, freq_lo, freq_hi, input_files = 0, [], [], [], [], []
    #     for this_metadata in tqdm.tqdm(sf_meta, desc="Stacking archives"):
    #         # Make sure stack file exists before proceeding
    #         f = this_metadata["outfile"]
    #         if not os.path.exists(f):
    #             self.logger.warning(f"Stack file {f} does not exist. The processing thread might be failed or OOM killed. ")
    #             continue

    #         # Load the data
    #         this_data = np.load(f)

    #         # Get frequency information
    #         this_freq_lo, this_freq_hi = this_metadata["freq_lo"], this_metadata["freq_hi"]

    #         # Make sure the data has the most common duration
    #         if not np.isclose(this_metadata["duration"], median_duration, atol=self.duration_tol / 86400, rtol=0):
    #             self.logger.debug(f"Duration in stack file {f} ({this_metadata['duration']}) is inconsistent with the median duration ({median_duration}). Skipping.")
    #             continue

    #         # Check if frequency information is consistent with the first stack file
    #         if not np.isclose(this_freq_lo, most_common_frequency[0]) or not np.isclose(this_freq_hi, most_common_frequency[1]):
    #             self.logger.debug(f"Frequency information in stack file {f} ({this_freq_lo}-{this_freq_hi}) is inconsistent with the first stack file ({most_common_frequency[0]}-{most_common_frequency[1]}).")
    #             if not self.allow_inconsistent_freq:
    #                 self.logger.debug(f"Skipping stack file {f} due to inconsistent frequency information.")
    #                 continue

    #         # Ensure the data has the most common shape
    #         if this_metadata["shape"] != most_common_shape:
    #             if self.interpolate == "never":
    #                 self.logger.debug(f"Stack file {f} has shape {this_metadata["shape"]}, expected {most_common_shape}. Skipping.")
    #                 continue
    #             else:
    #                 if self.interpolate == "minimal":
    #                     self.logger.debug(f"Resizing stack file {f} from {this_metadata["shape"]} to {most_common_shape} despite trying to avoid interpolation as much as possible.")
    #                 this_data = _resize_array(this_data, most_common_shape)

    #         # Stack the data
    #         stacked_data += this_data

    #         # Append frequency information
    #         freq_lo.append(this_freq_lo)
    #         freq_hi.append(this_freq_hi)

    #         # Get the duration and snr
    #         durations.append(this_metadata["duration"])
    #         snrs.append(this_metadata["snr"])

    #         # Append input file
    #         input_files.append(this_metadata["location"])

    #         # Remove this_data from memory
    #         del this_data
    #         gc.collect()

    #         # Remove the stack file
    #         os.remove(f)

    #         # Increment the number of stacked files
    #         n_stacked += 1

    #     # Get new data shape
    #     n_subs, n_pols, n_freqs, n_bins = stacked_data.shape

    #     # Print the results
    #     self.logger.info(f"Stacked {n_stacked} out of {len(self.files)} files.")
    #     self.logger.info(f"Stacked data shape: {stacked_data.shape}")

    #     # Return stacked data
    #     return StackedData(
    #         stacked_data=stacked_data, 
    #         n_subs=n_subs,
    #         n_pols=n_pols,
    #         n_freqs=n_freqs,
    #         n_bins=n_bins,
    #         n_stacked=n_stacked,
    #         durations=durations,
    #         snrs=snrs, 
    #         freq_lo=freq_lo, 
    #         freq_hi=freq_hi,
    #         model=open(self.parfile, "r").read(), 
    #         interpolate=self.interpolate,
    #         remove_baseline=self.remove_baseline, 
    #         allow_inconsistent_freq=self.allow_inconsistent_freq,
    #         duration_tol=self.duration_tol,
    #         input_files=input_files
    #     )


    def __stack(self, sf_meta):

        # Create a mask for good stack files
        
        

        # Unpack the results
        n_subs, n_pols, n_freqs, n_bins = [], [], [], []
        _res_shapes, _res_frequencies, _res_durations = [], [], []
        for res in sf_meta:
            if res["shape"] is None:
                continue
            
            n_subs.append(res["shape"][0])
            n_pols.append(res["shape"][1])
            n_freqs.append(res["shape"][2])
            n_bins.append(res["shape"][3])

            _res_shapes.append(res["shape"])
            _res_durations.append(res["duration"])
            _res_frequencies.append((res["freq_lo"], res["freq_hi"]))

        if len(n_subs) == 0:
            raise Exception("No valid stack files found.")


        # Find median duration
        median_duration = np.median(_res_durations)

        # Find most common shapes
        most_common_shape = self.__find_most_common(_res_shapes)

        # Find most common frequency range
        most_common_frequency = self.__find_most_common(_res_frequencies)

        # If always interpolate, override the most common shape with the target shape
        if self.interpolate == "always":
            most_common_shape = (self.n_subs, self.n_pols, self.n_freqs, self.n_bins)
            self.logger.debug(f"Interpolating all stack files to the most common shape {most_common_shape} since interpolate=always.")

        # Initialize the stacked data array
        stacked_data = np.zeros(most_common_shape, dtype=np.float32)

        # Stack
        n_stacked, durations, snrs, freq_lo, freq_hi, input_files = 0, [], [], [], [], []
        for this_metadata in tqdm.tqdm(sf_meta, desc="Stacking archives"):
            # Make sure stack file exists before proceeding
            f = this_metadata["outfile"]
            if not os.path.exists(f):
                self.logger.warning(f"Stack file {f} does not exist. The processing thread might be failed or OOM killed. ")
                continue

            # Load the data
            this_data = np.load(f)

            # Get frequency information
            this_freq_lo, this_freq_hi = this_metadata["freq_lo"], this_metadata["freq_hi"]

            # Make sure the data has the most common duration
            if not np.isclose(this_metadata["duration"], median_duration, atol=self.duration_tol / 86400, rtol=0):
                self.logger.debug(f"Duration in stack file {f} ({this_metadata['duration']}) is inconsistent with the median duration ({median_duration}). Skipping.")
                continue

            # Check if frequency information is consistent with the first stack file
            if not np.isclose(this_freq_lo, most_common_frequency[0]) or not np.isclose(this_freq_hi, most_common_frequency[1]):
                self.logger.debug(f"Frequency information in stack file {f} ({this_freq_lo}-{this_freq_hi}) is inconsistent with the first stack file ({most_common_frequency[0]}-{most_common_frequency[1]}).")
                if not self.allow_inconsistent_freq:
                    self.logger.debug(f"Skipping stack file {f} due to inconsistent frequency information.")
                    continue

            # Ensure the data has the most common shape
            if this_metadata["shape"] != most_common_shape:
                if self.interpolate == "never":
                    self.logger.debug(f"Stack file {f} has shape {this_metadata["shape"]}, expected {most_common_shape}. Skipping.")
                    continue
                else:
                    if self.interpolate == "minimal":
                        self.logger.debug(f"Resizing stack file {f} from {this_metadata["shape"]} to {most_common_shape} despite trying to avoid interpolation as much as possible.")
                    this_data = _resize_array(this_data, most_common_shape)

            # Stack the data
            stacked_data += this_data

            # Append frequency information
            freq_lo.append(this_freq_lo)
            freq_hi.append(this_freq_hi)

            # Get the duration and snr
            durations.append(this_metadata["duration"])
            snrs.append(this_metadata["snr"])

            # Append input file
            input_files.append(this_metadata["location"])

            # Remove this_data from memory
            del this_data
            gc.collect()

            # Remove the stack file
            os.remove(f)

            # Increment the number of stacked files
            n_stacked += 1

        # Get new data shape
        n_subs, n_pols, n_freqs, n_bins = stacked_data.shape

        # Print the results
        self.logger.info(f"Stacked {n_stacked} out of {len(self.files)} files.")
        self.logger.info(f"Stacked data shape: {stacked_data.shape}")

        # Return stacked data
        return StackedData(
            stacked_data=stacked_data, 
            n_subs=n_subs,
            n_pols=n_pols,
            n_freqs=n_freqs,
            n_bins=n_bins,
            n_stacked=n_stacked,
            durations=durations,
            snrs=snrs, 
            freq_lo=freq_lo, 
            freq_hi=freq_hi,
            model=open(self.parfile, "r").read(), 
            interpolate=self.interpolate,
            remove_baseline=self.remove_baseline, 
            allow_inconsistent_freq=self.allow_inconsistent_freq,
            duration_tol=self.duration_tol,
            input_files=input_files
        )

    def stack(self): 
        try:
            # Create tempdir
            os.makedirs(self.tempdir)

            # Prepare worker payload
            worker_payload = {
                "jumps": self.jumps,
                "tempdir": self.tempdir,
                "parfile": self.parfile,
                "remove_baseline": self.remove_baseline,
                "interpolate": self.interpolate,
                "n_subs": self.n_subs,
                "n_pols": self.n_pols,
                "n_freqs": self.n_freqs,
                "n_bins": self.n_bins,
                "logger": self.logger
            }

            # Prepare data
            tasks = [(i, file_info, worker_payload, self.logger.copy()) for i, file_info in enumerate(self.files)]
            with Pool(self.n_pools, maxtasksperchild=1) as pool:
                sf_meta = list(tqdm.tqdm(
                    pool.imap(_stack_worker, tasks),
                    total=len(tasks),
                    desc="Preparing archives",
                ))

            # Run stacking
            return self.__stack(sf_meta)
            
        except Exception:
            # Tempdir can still be cleanned up when task failed. 
            self.logger.warning(traceback.format_exc())
            raise
        finally:
            # Clean up tempdir
            shutil.rmtree(self.tempdir, ignore_errors=True)
            