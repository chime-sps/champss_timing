import os
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
from ..processing.dspsr_shutils import dspsr_shutils
from ..processing.archive_shutils import archive_shutils

class stack_utils():
    def __init__(self, files, parfile, n_subs=16, n_pols=3, n_freqs=1024, n_bins=1024, n_pools=4, jumps={}, workspace="/tmp", logger=logger()):
        self.n_subs = n_subs
        self.n_pols = n_pols
        self.n_freqs = n_freqs
        self.n_bins = n_bins
        self.files = files
        self.parfile = parfile
        self.logger = logger
        self.n_pools = n_pools
        self.jumps = jumps
        self.tempdir = workspace + f"/champss_timing__stack_utils/{utils.get_time_string()}__{utils.get_rand_string()}"

        # Some data necessary for alias_utils
        self.durations = []
        self.snrs = []
        self.n_stacked = 0

        if not os.path.exists(workspace):
            raise Exception(f"Workspace {workspace} does not exist. Please create it first.")

        # Check if npol make sense
        if n_pols != 1 and n_pols != 4:
            raise ValueError("n_pols must be 1 or 3 to be physically meaningful")

        # Initialize the stacked data array: subints, pols, freqs, bins
        self.stacked_data = np.zeros((self.n_subs, self.n_pols, self.n_freqs, self.n_bins), dtype=np.float32)

    def stack(self, normalize=True): 
        # Create tempdir
        os.makedirs(self.tempdir)

        # Run stacking
        try: # Tempdir can still be cleanned up when task failed. 
            with Pool(self.n_pools) as pool:
                stack_files = pool.starmap(
                    self._stack, 
                    tqdm.tqdm([(file_info, normalize) for file_info in self.files], desc="Preparing archives")
                )
                
            for f in tqdm.tqdm(stack_files, desc="Stacking archives"):
                if not os.path.exists(f):
                    self.logger.warning(f"Stack file {f} does not exist. The processing thread might be failed or OOM killed. ")
                    continue
                # Load the data
                this_data = np.load(f)

                # Stack the data
                self.stacked_data += this_data["data"]

                # Get the duration and snr
                self.durations.append(this_data["duration"])
                self.snrs.append(this_data["snr"])

                # Remove the stack file
                os.remove(f)

                # Increment the number of stacked files
                self.n_stacked += 1
        except Exception:
            self.logger.warning(traceback.format_exc())

        # Clean up tempdir
        shutil.rmtree(self.tempdir)

        # Print the results
        self.logger.info(f"Stacked {self.n_stacked} out of {len(self.files)} files.")
        self.logger.info(f"Stacked data shape: {self.stacked_data.shape}")
    
    def _stack(self, file_info, normalize=True):
        """
        Stack the data from the input archive.
        """

        this_outfile = ""

        # Check if the file info is valid
        if "location" not in file_info or "backend" not in file_info:
            self.logger.error(f"File info {file_info} does not contain location or backend information. Skipping.")
            return this_outfile

        try:
            # read file info and jump
            this_location = file_info["location"]
            this_backend = file_info["backend"]
            this_jump = 0
            if this_backend in self.jumps:
                this_jump = self.jumps[this_backend][0]
            
            # create workspace for this archive
            this_workspace = os.path.join(self.tempdir, os.path.basename(this_location))
            this_outfile = os.path.join(self.tempdir, "stackfile__" + os.path.basename(this_location) + ".npz")
            this_ar = this_workspace + "/data"
            this_par = this_workspace + "/parfile"
            if not os.path.exists(this_workspace):
                os.makedirs(this_workspace)
                
            # prepare the archive
            shutil.copy(this_location, this_ar)
            shutil.copy(self.parfile, this_par)

            # convert filterbank data as needed
            if this_location.endswith(".fil"):
                this_fil = this_workspace + "/data.fil"
                
                # copy data to workspace
                shutil.move(this_ar, this_fil)
    
                # convert to ar using dspsr
                du = dspsr_shutils(n_pools=1)
                du.fil2ar(this_fil, this_ar, this_par, nbin=self.n_bins)

                # rename converted ar
                shutil.move(this_ar + ".ar", this_ar)
    
                # remove the filterbank data
                os.remove(this_fil)
                del this_fil
                
    
            # # downsample the archive first as needed to save memory
            # this_arch = psrchive.Archive_load(this_ar) # load the archive
            # this_arch.dedisperse() # dedisperse
            # self.resize_psrchive(this_arch, (self.n_subs, self.n_pols, self.n_freqs, self.n_bins)) # downsample using PSRCHIVE scrunch
            # this_arch.unload(this_ar) # overwrite the archive
            
            # initialize archive shutils
            ashu = archive_shutils(this_ar)
    
            # zap rfi
            ashu.clfd()
    
            # install parfile
            ashu.install_parfile(this_par, jump=this_jump)
    
            # load the archive
            this_arch = psrchive.Archive_load(this_ar)
    
            # dedisperse
            this_arch.dedisperse()
    
            # downsample using PSRCHIVE scrunch
            self.resize_psrchive(this_arch, (self.n_subs, self.n_pols, self.n_freqs, self.n_bins))
    
            # get the data
            this_data = this_arch.get_data()
                
            # upsample using scipy as needed
            if this_data.shape != (self.n_subs, self.n_pols, self.n_freqs, self.n_bins):
                self.logger.warning(f"Upsampling {this_ar} from {this_data.shape} to {(self.n_subs, self.n_pols, self.n_freqs, self.n_bins)}", layer=1)
                this_data = self.resize_array(this_data, (self.n_subs, self.n_pols, self.n_freqs, self.n_bins))
    
            # normalize
            if normalize:
                this_data = self.normalize_data(this_data)

            # get_snr
            this_profile = np.sum(this_data, axis=(0, 1, 2))
            this_snr = np.nanmax(this_profile) / np.nanstd(this_profile)

            # get duration
            this_duration = (this_arch.end_time() - this_arch.start_time()).in_days()
    
            # save data to tempdir
            np.savez(this_outfile, data=this_data, duration=this_duration, snr=this_snr)
            
        except Exception:
            self.logger.warning(f"Thread {this_location} failed. Please refer to the traceback below")
            self.logger.warning(traceback.format_exc())
        
        # remove the workspace
        shutil.rmtree(this_workspace, ignore_errors=True)

        return this_outfile

    def normalize_data(self, data):
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

    def resize_psrchive(self, ar_obj, new_shape):
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

    def resize_array(self, input_array, new_shape):
        """
        Resize an N-dimensional numpy array to a new shape using interpolation.

        Parameters:
            input_array (np.ndarray): The array to resize.
            new_shape (tuple): The desired shape (must be same number of dimensions as input).
            order (int): Interpolation order (0=nearest, 1=linear, 3=cubic, etc.)

        Returns:
            np.ndarray: Resized array.
        """
        input_shape = np.array(input_array.shape)
        new_shape = np.array(new_shape)

        if len(input_shape) != len(new_shape):
            raise ValueError("new_shape must have same number of dimensions as input_array")

        if np.all(input_shape == new_shape):
            return input_array

        return zoom(input_array, new_shape / input_shape, order=1) # Use scipy's built-in resize function
    
    def get_data(self, n_subs=None, n_pols=None, n_freqs=None, n_bins=None, normalize=False):
        data_shape = [self.n_subs, self.n_pols, self.n_freqs, self.n_bins]
        data_copy = copy.deepcopy(self.stacked_data)

        if n_subs is not None:
            if n_subs > data_shape[0]:
                self.logger.warning(f"Upsampling n_subs from {data_shape[0]} to {n_subs}")
            data_shape[0] = n_subs
            
        if n_pols is not None:
            if n_pols > data_shape[1]:
                self.logger.warning(f"Upsampling n_subs from {data_shape[1]} to {n_pols}")
            data_shape[1] = n_pols
            
        if n_freqs is not None:
            if n_freqs > data_shape[2]:
                self.logger.warning(f"Upsampling n_subs from {data_shape[2]} to {n_freqs}")
            data_shape[2] = n_freqs
            
        if n_bins is not None:
            if n_bins > data_shape[3]:
                self.logger.warning(f"Upsampling n_subs from {data_shape[3]} to {n_bins}")
            data_shape[3] = n_bins

        if data_shape == (self.n_subs, self.n_pols, self.n_freqs, self.n_bins): 
            if normalize:
                return self.normalize_data(data_copy)
            return data_copy

        if normalize:
            return self.normalize_data(self.resize_array(data_copy, data_shape))
            
        return self.resize_array(data_copy, data_shape)

