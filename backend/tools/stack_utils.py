import os
import gc
import copy
import tqdm
import shutil
import psrchive
import traceback
import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool

from ..utils.utils import utils
from ..utils.logger import logger
from ..utils.data_quality import data_quality_utils
from ..processing.dspsr_shutils import dspsr_shutils
from ..processing.archive_shutils import archive_shutils

def _normalize_data(data):
    """
    Normalize the data by dividing each subint, pol, and freq by its maximum value.
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
        str: The path to the resized PSRCHIVE archive.
    """

    # Get shapes
    data_n_subs, data_n_pols, data_n_freqs, data_n_bins = ar_obj.get_data().shape
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
    """
    Worker function to stack the data from the input archive.
    Parameters:
        args (tuple): A tuple containing (file_info, normalize, config, logger).
            file_info (dict): Information about the input file, including location and backend.
            normalize (bool): Whether to normalize the data.
            config (dict): Configuration dictionary containing various settings 
                (jumps, tempdir, parfile, remove_baseline, n_subs, n_pols, n_freqs, n_bins).
            logger (logging.Logger): Logger for logging messages.

    Returns:
        str: Path to the output stacked file.
    """

    this_outfile = ""
    this_shape = [0, 0, 0, 0]

    # Unpack arguments
    file_info, config, logger = args

    # Check if the file info is valid
    if "location" not in file_info or "backend" not in file_info:
        logger.error(f"File info {file_info} does not contain location or backend information. Skipping.")
        return this_outfile, this_shape

    try:
        # read file info and jump
        this_location = file_info["location"]
        this_backend = file_info["backend"]
        this_jump = 0
        if this_backend in config["jumps"]:
            this_jump = config["jumps"][this_backend][0]
        
        # create workspace for this archive
        this_workspace = os.path.join(config["tempdir"], os.path.basename(this_location))
        this_outfile = os.path.join(config["tempdir"], "stackfile__" + os.path.basename(this_location) + ".npz")
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
        np.savez(this_outfile, data=this_data, duration=this_duration, snr=this_snr)
        
    except Exception:
        logger.warning(f"Thread {this_location} failed. Please refer to the traceback below")
        logger.warning(traceback.format_exc())
    
    # remove the workspace
    shutil.rmtree(this_workspace, ignore_errors=True)

    return this_outfile, this_shape

class stack_utils():
    def __init__(self, files, parfile, n_subs=16, n_pols=3, n_freqs=1024, n_bins=1024, n_pools=4, jumps={}, remove_baseline=False, interpolate="always", workspace="/tmp", logger=logger()):
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
        self.tempdir = workspace + f"/champss_timing__stack_utils/{utils.get_time_string()}__{utils.get_rand_string()}"

        # Make sure interpolation mode is valid
        if self.interpolate not in ["always", "minimal", "never"]:
            raise ValueError("interpolate must be one of 'always', 'minimal', or 'never'")

        # Some data necessary for alias_utils
        self.durations = []
        self.snrs = []
        self.n_stacked = 0

        if not os.path.exists(workspace):
            raise Exception(f"Workspace {workspace} does not exist. Please create it first.")

        # Check if npol make sense
        if n_pols != 1 and n_pols != 4:
            raise ValueError("n_pols must be 1 or 4 to be physically meaningful")

        # Initialize the stacked data array: subints, pols, freqs, bins
        self.stacked_data = None

    def stack(self): 
        # Create tempdir
        os.makedirs(self.tempdir)

        # Create worker config
        worker_config = {
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

        # Run stacking
        try:
            # Prepare data
            tasks = [(file_info, worker_config, self.logger.copy()) for file_info in self.files]
            with Pool(self.n_pools) as pool:
                stack_files_res = list(tqdm.tqdm(
                    pool.imap(_stack_worker, tasks),
                    total=len(tasks),
                    desc="Preparing archives",
                ))

            # Unpack the results
            stack_files = []
            n_subs, n_pols, n_freqs, n_bins = [], [], [], []
            for res in stack_files_res:
                stack_files.append(res[0])
                n_subs.append(res[1][0])
                n_pols.append(res[1][1])
                n_freqs.append(res[1][2])
                n_bins.append(res[1][3])

            # Get the most common shape among the stack files
            most_common_shape = (
                max(set(n_subs), key=n_subs.count),
                max(set(n_pols), key=n_pols.count),
                max(set(n_freqs), key=n_freqs.count),
                max(set(n_bins), key=n_bins.count)
            )

            # If always interpolate, override the most common shape with the target shape
            if self.interpolate == "always":
                most_common_shape = (self.n_subs, self.n_pols, self.n_freqs, self.n_bins)
                self.logger.debug(f"Interpolating all stack files to the most common shape {most_common_shape} since interpolate=always.")

            # Initialize the stacked data array
            self.stacked_data = np.zeros(most_common_shape, dtype=np.float32)

            # Stack
            for f in tqdm.tqdm(stack_files, desc="Stacking archives"):
                if not os.path.exists(f):
                    self.logger.warning(f"Stack file {f} does not exist. The processing thread might be failed or OOM killed. ")
                    continue

                # Load the data
                this_data = dict(np.load(f))

                # Ensure the data has the most common shape
                if this_data["data"].shape != most_common_shape:
                    if self.interpolate == "never":
                        self.logger.debug(f"Stack file {f} has shape {this_data['data'].shape}, expected {most_common_shape}. Skipping.")
                        continue
                    else:
                        this_data["data"] = _resize_array(this_data["data"], most_common_shape)
                        if self.interpolate == "minimal":
                            self.logger.debug(f"Stack file {f} has been resized from {this_data['data'].shape} to {most_common_shape} despite trying to avoid interpolation as much as possible.")

                # Stack the data
                self.stacked_data += this_data["data"]

                # Get the duration and snr
                self.durations.append(this_data["duration"])
                self.snrs.append(this_data["snr"])

                # Remove this_data from memory
                del this_data
                gc.collect()

                # Remove the stack file
                os.remove(f)

                # Increment the number of stacked files
                self.n_stacked += 1

            # Update the size
            self.n_subs, self.n_pols, self.n_freqs, self.n_bins = self.stacked_data.shape
        except Exception:
            # Tempdir can still be cleanned up when task failed. 
            self.logger.warning(traceback.format_exc())

        # Clean up tempdir
        shutil.rmtree(self.tempdir)

        # Print the results
        self.logger.info(f"Stacked {self.n_stacked} out of {len(self.files)} files.")
        self.logger.info(f"Stacked data shape: {self.stacked_data.shape}")
    
    def get_data(self, fscrunch=False, tscrunch=False, normalize=False, keepdims=False):
        data_copy = copy.deepcopy(self.stacked_data)

        if fscrunch:
            data_copy = data_copy.mean(axis=2, keepdims=keepdims)

        if tscrunch:
            data_copy = data_copy.mean(axis=0, keepdims=keepdims)

        if normalize:
            data_copy = _normalize_data(data_copy)
            
        return data_copy

    def get_stacked_snr(self):
        if self.n_bins <= 1:
            return 0.0

        return data_quality_utils.boxcar_snr(
            self.get_data().sum(axis=(0, 1, 2))
        )

    def save(self, filename, format="npz"):
        """
        Save the stacked data and metadata to a file.

        Args:
            filename (str): The name of the file to save the data to.
            format (str): The format to save the data in (options: npz, pkl, json).
        """
        # Gather metadata
        metadata = {
            "n_subs": self.n_subs,
            "n_pols": self.n_pols,
            "n_freqs": self.n_freqs,
            "n_bins": self.n_bins,
            "n_stacked": self.n_stacked,
            "durations": self.durations,
            "snrs": self.snrs, 
            "model": open(self.parfile, "r").read(), 
            "remove_baseline": self.remove_baseline, 
            "input_files": self.files
        }

        # Save the data and metadata
        if format == "npz":
            np.savez(filename, data=self.stacked_data, metadata=metadata)
        elif format == "pkl":
            with open(filename, "wb") as f:
                import pickle
                pickle.dump(utils.numpy_to_native({"data": self.stacked_data, "metadata": metadata}), f)
        elif format == "json":
            with open(filename, "w") as f:
                import json
                json.dump(utils.numpy_to_native({"data": self.stacked_data, "metadata": metadata}), f)
        else: 
            raise ValueError(f"Unsupported format: {format}. Supported formats are: npz, pkl, json.")
