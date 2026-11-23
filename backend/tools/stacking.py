import os
import gc
import copy
import tqdm
import shutil
import psrchive
import traceback
import numpy as np
import astropy.units as u

from collections import Counter
from scipy.ndimage import zoom
from scipy.signal import resample
from scipy.stats import circmean

try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool

from ..utils.utils import utils
from ..utils.logger import logger
from ..utils.data_quality import data_quality_utils
from ..processing.dspsr_shutils import dspsr_shutils
from ..processing.archive_shutils import archive_shutils
from ..utils.stacked_data import StackedData, StackedDataMetadata


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

    # Unpack arguments
    task_i, file_info, config, logger = args

    meta = None
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

        # scrunch polarization before clfd to save memory
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

        # # downsample using PSRCHIVE scrunch
        # _resize_psrchive(this_arch, (config["n_subs"], config["n_pols"], config["n_freqs"], config["n_bins"]))

        # get the start time
        this_start_mjd = this_arch.start_time().in_days()

        # get subint durations
        this_subint_durations = []
        this_subint_epochs = []
        this_subint_folding_periods = []
        for i in range(this_arch.get_nsubint()):
            integ = this_arch.get_Integration(i)
            this_subint_durations.append(integ.get_duration() / 86400) # to days for consistency
            this_subint_epochs.append(integ.get_epoch().in_days())
            this_subint_folding_periods.append(integ.get_folding_period())

        # get frequency information
        freq_lo = this_arch.get_centre_frequency() - this_arch.get_bandwidth()/2.0
        freq_hi = this_arch.get_centre_frequency() + this_arch.get_bandwidth()/2.0

        # get dm
        this_dm = this_arch.get_dispersion_measure()

        # get mean period
        this_period = np.mean(this_subint_folding_periods)

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

        meta = {
            "outfile": this_outfile, 
            "shape": this_shape, 
            "start": this_start_mjd,
            "durations": this_subint_durations,
            "subint_epochs": this_subint_epochs,
            "snr": this_snr, 
            "freq_lo": freq_lo, 
            "freq_hi": freq_hi, 
            "dm": this_dm,
            "period": this_period,
            "input_file": this_location
        }
        
    except Exception:
        logger.warning(f"Thread {this_location} failed. Please refer to the traceback below")
        logger.warning(traceback.format_exc())
    finally:
        # remove the workspace
        if this_workspace is not None:
            shutil.rmtree(this_workspace, ignore_errors=True)

    return meta

class Stacking:
    def __init__(
            self, 
            files,
            parfile, 
            ncpus=1, 
            jumps={}, 
            remove_baseline=False,
            resample_phase_bins=False, 
            workspace="/tmp", 
            verbose=False, 
            logger=logger()
        ):
        """
        Initialize the stack_utils class.

        Parameters:
        files (list): List of archive files to be stacked.
        parfile (str): Parameter file.
        n_subs (int): Number of sub-integrations.
        n_pols (int): Number of polarizations.
        n_freqs (int): Number of frequency channels.
        n_bins (int): Number of phase bins.
        ncpus (int): Number of parallel pools.
        jumps (dict): Dictionary of jumps.
        remove_baseline (bool): Whether to remove baseline.
        resample_phase_bins (bool): Whether to resample phase bins.
        duration_tol (float): Tolerance for duration mismatch in seconds.
        workspace (str): Workspace directory.
        logger (logger): Logger instance.
        """

        self.files = files
        self.parfile = parfile
        self.logger = logger
        self.verbose = verbose
        self.ncpus = ncpus
        self.jumps = jumps
        self.remove_baseline = remove_baseline
        self.resample_phase_bins = resample_phase_bins
        self.tempdir = workspace + f"/champss_timing__stack_utils/{utils.get_time_string()}__{utils.get_rand_string()}"

        if not os.path.exists(workspace):
            raise Exception(f"Workspace {workspace} does not exist. Please create it first.")

    @staticmethod
    def load(filename):
        """Shortcut to load a StackedData instance from a file."""
        return StackedData.load(filename)

    def __find_most_common(self, arr):
        return Counter(
            [tuple(a) for a in arr]
        ).most_common(1)[0][0]

    def __compute_best_binning_offset(self, subint_epochs, window_size):
        # The distance from each subint epoch to the distance to the nearest stacked data bin
        d = subint_epochs % window_size

        # Mean of d is ~ an estimator of the offset the minimizes the distance to the nearest stacked data bin
        best_offset = circmean(d, high=window_size, low=0)

        return best_offset % window_size

    def __compute_stacked_subint_epochs(self, subint_epochs):
        # Remove the likely partial subints at the beginning and end of the observation
        # (...to avoid they biasing the statistics)
        subint_epochs_dropped_partial = subint_epochs.copy()
        for i in range(subint_epochs_dropped_partial.shape[0]):
            finite_indices = np.flatnonzero(np.isfinite(subint_epochs_dropped_partial[i]))
            if finite_indices.size > 0:
                subint_epochs_dropped_partial[i, finite_indices[0]] = np.nan
                subint_epochs_dropped_partial[i, finite_indices[-1]] = np.nan

        # Calculate "nominal" window size (i.e., the stacked subint size)
        # window_size = np.nanmean(np.diff(subint_epochs_dropped_partial, axis=1).flatten())
        # window_size = np.nanmedian(np.diff(subint_epochs_dropped_partial, axis=1).flatten())
        window_size = np.nanmedian(
            np.nanmean(
                np.diff(
                    subint_epochs_dropped_partial, 
                    axis=1), 
                axis=1
            ) # mean of the gaps between subints for each obs. mean can account for jitters in psrchive subint lengths
        ) # median to reduce the influence of outlier obs that may be folded at different subint lengths

        # Compute the best binning offset
        best_offset = self.__compute_best_binning_offset(
            subint_epochs_dropped_partial[np.isfinite(subint_epochs_dropped_partial)].flatten(), 
            window_size
        )

        # Generate the stacked subint epochs
        stacked_epochs = np.arange(
            np.round(np.nanmin(subint_epochs - best_offset) / window_size), 
            np.round(np.nanmax(subint_epochs - best_offset) / window_size) + 1
        ) * window_size

        # Apply offset
        stacked_epochs += best_offset

        return stacked_epochs

    def __stack_subints(self, metadata, expected_shape):
        def wrap_by_sidereal_day(epochs):
            ref = np.nanpercentile(np.ravel(epochs), 50, method="lower")
            d = (epochs - ref) % sday
            return np.where(d < sday / 2, d, d - sday), ref

        # Sanity check: each data has the same sizes except for subints
        for this_meta in metadata:
            if this_meta["shape"][1:3] != expected_shape[1:3]:
                raise ValueError("Inconsistent data shapes")

        # Unpack subint epochs
        subint_epochs = np.full((len(metadata), max([m["shape"][0] for m in metadata])), np.nan) # use np.nan as placeholder for shorter subints
        for i, this_meta in enumerate(metadata):
            subint_epochs[i, :this_meta["shape"][0]] = this_meta["subint_epochs"]

        # Rewrite the epoch of each subint respect to sidereal time
        sday = (1.0 * u.sday).to(u.day).value
        subint_epochs, ref_epoch = wrap_by_sidereal_day(subint_epochs)
        subint_epochs = subint_epochs - np.nanmedian(subint_epochs)

        # Determine the epochs of stacked data
        # stacked_subint_epochs = np.nanmedian(subint_epochs, axis=0)
        stacked_subint_epochs = self.__compute_stacked_subint_epochs(subint_epochs)

        # Initialize the stacked data array
        stacked_data = np.zeros((
            len(stacked_subint_epochs),
            metadata[0]["shape"][1],
            metadata[0]["shape"][2],
            metadata[0]["shape"][3]
        ))

        # Initialize the exposure array
        stacked_expos = np.zeros(len(stacked_subint_epochs))

        # Stack subints
        for i, this_meta in enumerate(metadata):
            # Load the actual data
            this_data = np.load(this_meta["outfile"])

            # Resample phsae bins
            if this_meta["shape"][3] != expected_shape[3]:
                self.logger.info(f"Resampling phase bins from {this_meta['shape'][3]} to {expected_shape[3]}")
                this_data = resample(
                    this_data, 
                    expected_shape[3],
                    axis=3
                )

            # Loop over the subint
            for j, this_subint in enumerate(this_data):
                # Find the stacked subint that has the closest epoch
                nearest_idx = np.argmin(
                    np.abs(stacked_subint_epochs - subint_epochs[i, j])
                )

                # Stack the subint
                stacked_data[nearest_idx] += this_subint

                # Add the durations
                stacked_expos[nearest_idx] += this_meta["durations"][j]

            # Remove this_data from memory
            del this_data
            gc.collect()

        return stacked_data, stacked_subint_epochs, stacked_expos, subint_epochs, ref_epoch

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
                "resample_phase_bins": self.resample_phase_bins,
                "logger": self.logger
            }

            # Prepare data
            tasks = [(i, file_info, worker_payload, self.logger.copy()) for i, file_info in enumerate(self.files)]
            with Pool(self.ncpus, maxtasksperchild=1) as pool:
                sf_meta = np.array(
                    list(
                        tqdm.tqdm(
                            pool.imap(_stack_worker, tasks),
                            total=len(tasks),
                            desc="Preparing archives",
                        )
                    ), 
                    dtype=object
                )
            self.logger.debug(f"Number of files processed: {len(sf_meta)}")

            # Create a mask to keep track of valid files
            mask = np.ones(len(sf_meta), dtype=bool)

            # Filter out files that are failed to prepare
            for i, meta in enumerate(sf_meta):
                if meta is None:
                    mask[i] = False
            self.logger.debug(f"Number of files successfully prepared: {np.sum(mask)}")

            # Sanity check if data were folding at the same dm
            for meta in sf_meta[mask]:
                if not np.isclose(meta["dm"], sf_meta[mask][0]["dm"]):
                    raise ValueError("Data were not folding at the same dm.")

            # Find the expected size of the data
            expected_shape = self.__find_most_common([meta["shape"] for meta in sf_meta])

            # Make sure all the data has the expected shape
            for i, meta in enumerate(sf_meta):
                if not self.resample_phase_bins:
                    mask[i] = (meta["shape"][1:] == expected_shape[1:])
                else:
                    mask[i] = (meta["shape"][1:3] == expected_shape[1:3])
            self.logger.debug(f"Number of files with consistent shape: {np.sum(mask)} (shape={expected_shape})")
            self.logger.debug(f"Resampling phase bins? {"yes" if self.resample_phase_bins else "no"}", layer=1)

            # Unpack frequency ranges
            freq_lo_vals = np.array([meta["freq_lo"] for meta in sf_meta])
            freq_hi_vals = np.array([meta["freq_hi"] for meta in sf_meta])

            # Find the most common frequency ranges
            most_common_freqs = self.__find_most_common(
                list(zip(freq_lo_vals[mask], freq_hi_vals[mask]))
            )

            # Filter out data with frequency ranges that deviate significantly from the median
            mask &= np.isclose(freq_lo_vals, most_common_freqs[0])
            mask &= np.isclose(freq_hi_vals, most_common_freqs[1])
            self.logger.debug(f"Number of files with consistent frequency ranges: {np.sum(mask)} (freq_lo={most_common_freqs[0]}, freq_hi={most_common_freqs[1]})")

            # Sanity check on the mask to ensure there is data to stack
            if not np.any(mask):
                raise ValueError("No data to stack. ")

            # Run stacking
            stacked_data, stacked_epochs, stacked_expos, per_file_subint_epochs, ref_epoch = self.__stack_subints(sf_meta[mask], expected_shape=expected_shape)

            # Put together stacked metadata
            stacked_metadata = StackedDataMetadata(
                freq_lo=most_common_freqs[0], 
                freq_hi=most_common_freqs[1],
                dm=np.mean([meta["dm"] for meta in sf_meta[mask]]), 
                period=np.mean([meta["period"] for meta in sf_meta[mask]]),
                model=open(self.parfile, "r").read(), 
                ref_epoch=ref_epoch,
                resample_phase_bins=self.resample_phase_bins,
                remove_baseline=self.remove_baseline, 
                input_files=[]
            )
            
            for i, meta in enumerate(sf_meta[mask]):
                stacked_metadata.input_files.append({
                    "location": meta["input_file"],
                    "subint_epochs": per_file_subint_epochs[i], 
                    "snr": meta["snr"]
                })

            return StackedData(
                data=stacked_data, 
                epochs=stacked_epochs,
                expos=stacked_expos,
                meta=stacked_metadata
            )
            
        except Exception:
            # Tempdir can still be cleanned up when task failed. 
            self.logger.warning(traceback.format_exc())
            raise
        finally:
            # Clean up tempdir
            if not self.verbose:
                shutil.rmtree(self.tempdir, ignore_errors=True)
            