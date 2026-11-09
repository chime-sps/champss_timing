import numpy as np
import pint.logging
import tqdm
import shutil
import os
from io import StringIO
from pint.models import get_model
from multiprocessing import Pool

from ..utils.phase_coherent_search import PhaseCoherentSearch
from ..utils.logger import logger

from ..tools.ephm_install import EphmInstall
from ..io.archive import ArchiveReader
from ..processing.archive_shutils import archive_shutils
from ..utils.tempfile import TempDir

def InitialTimingSolutionOptimizer__load_archive_worker(payload):
    """
    Worker function to load an archive and apply the ephemeris model.

    Parameters
    ----------
    payload : tuple
        A tuple containing the archive file path and the model.

    Returns
    -------
    tuple
        A tuple containing the archive file path, the processed profile, and the epoch.
    """

    # Set logging level
    pint.logging.setup(level="WARNING")

    # Unpack the payload into archive file and model
    archive_file, model = payload

    # Load archive
    archive = ArchiveReader(
        archive_file, 
        dedisperse=True, 
        remove_baseline=True
    )

    # Get profile
    profile = archive.get_amps(tolist=True)

    # Get metadata
    epoch = archive.get_epoch()
    freq = archive.get_freq()
    site = archive.get_telescope()

    # Install ephemeris
    profile = EphmInstall(
        amps=profile, 
        freq=freq, 
        epoch=epoch, 
        site=site
    ).install_model(model)

    return archive_file, profile, epoch

class InitialTimingSolution:
    def __init__(self, model, best_pcs_state, logger=logger()):
        # Set logging level
        pint.logging.setup(level="WARNING")
        
        self.model = model
        self.best_pcs_state = best_pcs_state
        self.logger = logger

        # Get the best spindown position from the search results
        best_df0, best_df1 = best_pcs_state.df0, best_pcs_state.df1
        self.model.F0.value -= best_df0
        self.model.F1.value -= best_df1

    def plot(self, *args, **kwargs):
        return self.best_pcs_state.plot(*args, **kwargs)

    def as_parfile(self):
        return self.model.as_parfile()

    def write_parfile(self, filename):
        self.model.write_parfile(filename)

    def write_archive(self, filename, overwrite=True):
        from ..io.template_writer import TemplateWriter # Import the TemplateWriter here since TemplateWriter uses psrchive package but there's no reason for web server to install it. 

        with TemplateWriter(filename, overwrite=overwrite) as writer:
            # Write the data to the archive
            writer.write(self.best_pcs_state.get_stacked_profile(), interpolate=True)

            # Set the metadata
            if hasattr(self.model, "PSR"):
                writer.set_source(self.model.PSR.value)
            else:
                self.logger.warning("Model has no PSR attribute; setting source to 'Unknown'.")
                writer.set_source("Unknown")
            if hasattr(self.model, "DM"):
                writer.set_dm(float(self.model.DM.value))
            else:
                self.logger.warning("Model has no DM attribute; setting DM to '0.0'.")
                writer.set_dm(0.0)
            
            # Unload the archive
            writer.unload()

class InitialTimingSolutionOptimizer:
    def __init__(self, parfile, archive_files, max_n_obs=None, max_duty_cycle=None, ncpus=1, max_f1=2e-12, reset_f1=True, logger=logger()):
        """
        Initialize the InitialTimingSolutionOptimizer.

        parfile: str
            The path to the input parfile.
        archive_files: list of str
            The list of archive files to be optimized.
        max_n_obs: int, optional
            Cap on the number of observations to optimize.
        max_duty_cycle: float, optional
            The maximum duty cycle to search when computing boxcar SNR.
        ncpus: int, optional
            The number of CPUs to use.
        max_f1: float, optional
            The maximum absolute value of F1 to search.
        reset_f1: bool, optional
            Whether to reset F1 to zero in the input model.
        logger: logging.Logger, optional
            The logger to use for logging messages.
        """
        
        self.logger = logger
        self.max_n_obs = max_n_obs
        self.max_duty_cycle = max_duty_cycle
        self.max_width = None
        self.ncpus = ncpus
        self.max_f1 = max_f1
        self.reset_f1 = reset_f1

        # Load model
        self.logger.debug(f"Loading model...")
        self.model = self.__get_model(parfile)

        # Load archives
        self.logger.debug(f"Loading {len(archive_files)} archive files...")
        self.archive_files, self.profiles, self.epochs = self.__load_archives(archive_files, self.model, self.max_n_obs)

        # Calculate max width
        if self.max_duty_cycle is not None:
            self.max_width = int(len(self.profiles[0]) * self.max_duty_cycle)  # Assuming 4096 bins per profile as a default

        # Set pepoch to the center of the observation span
        center_epoch = np.median(self.epochs) # Use median to avoid a few observations taken far away skewing the center
        self.logger.debug(f"Changing PEPOCH from {self.model.PEPOCH.value} to {center_epoch} (the middle of the observation span)")
        self.model.change_pepoch(center_epoch)

    def __get_model(self, parfile):
        """Helper function to get model and ensure the model is valid."""
        m = get_model(parfile)

        if "Spindown" not in m.components:
            raise ValueError("Model must have a Spindown component.")

        if not hasattr(m, "F0"):
            raise ValueError("Model must have an F0 attribute.")

        if not hasattr(m, "PEPOCH"):
            raise ValueError("Model must have a PEPOCH attribute.")

        # Add F1 if not present
        c = m.components["Spindown"]
        if not hasattr(m, "F1"):
            self.logger.debug("F1 attribute not found in the model. Adding F1...", layer=1)
            c.add_param(m.F0.new_param(1), setup=True)
            p = getattr(m, "F1")
            p.quantity = 0.0 * p.units
            p.frozen = True
        
        # Set F1 to zero and freeze it if reset_f1 is True
        if self.reset_f1:
            p = getattr(m, "F1")
            p.quantity = 0.0 * p.units
            p.frozen = True

        # Add AbsPhase component if not present
        if "AbsPhase" not in m.components:
            self.logger.warning("AbsPhase component not found in the model. Adding AbsPhase...", layer=1)

            # Add component
            from pint.models.absolute_phase import AbsPhase
            m.add_component(AbsPhase(), validate=False)

            # Set default values for the new component
            m.TZRMJD.value = m.PEPOCH.value # use PEPOCH as the reference MJD for the absolute phase
            m.TZRSITE.value = "ssb"
            m.TZRFRQ.value = np.inf
            m.validate()

            self.logger.warning(f"TZRMJD={m.TZRMJD.value} (=PEPOCH), TZRSITE={m.TZRSITE.value}, TZRFRQ={m.TZRFRQ.value}", layer=2)

        return m

    def __load_archives(self, archive_files, model, max_n_obs):
        """Helper function to load multiple archives."""

        # Use multiprocessing to load archives in parallel
        with Pool(processes=self.ncpus) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(
                        InitialTimingSolutionOptimizer__load_archive_worker, 
                        [(archive_file, model) for archive_file in archive_files]
                    ), 
                    total=len(archive_files), 
                    bar_format="(Loading {n_fmt}/{total_fmt} archive files, {remaining} remaining)" if len(archive_files) > 14 else "",
                    leave=False
                )
            )

        # Unpack the results
        files, profiles, epochs = zip(*results)
        files, profiles, epochs = list(files), list(profiles), list(epochs)
        
        self.logger.debug(f"\nFinished loading {len(archive_files)} archive files.", layer=1)

        # Sort the profiles and epochs by epochs
        sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
        files = [files[i] for i in sorted_indices]
        profiles = [profiles[i] for i in sorted_indices]
        epochs = [epochs[i] for i in sorted_indices]

        # Limit the number of observations if max_n_obs is specified
        if max_n_obs is not None:
            # Pick archives
            indices = self.__pick_archives(epochs, max_n_obs)
            self.logger.debug(f"Using {len(indices)} out of {len(epochs)} archive files from {epochs[indices[0]]} to {epochs[indices[-1]]}.", layer=1)

            # Apply the mask
            files = [files[i] for i in indices]
            profiles = [profiles[i] for i in indices]
            epochs = [epochs[i] for i in indices]

        return files, profiles, epochs

    def __pick_archives(self, epochs, max_n_obs):
        """
        Pick the max_n_obs observations spanning the shortest time (i.e., closest together).
        
        epochs: list
            Epoch values (oldest -> newest). 
        max_n_obs: int or None
            The maximum number of observations to pick. If None, all observations are used.

        Returns the indices of the selected observations.
        """

        # Sanity check
        if len(epochs) <= max_n_obs:
            return np.arange(len(epochs))

        # Slide a window of size max_n_obs across the epochs and time span
        epochs = np.array(epochs)
        spans = epochs[max_n_obs - 1:] - epochs[:len(epochs) - max_n_obs + 1]

        # Find the start--end that minimizes the time span
        idx_start = int(np.argmin(spans))
        idx_end = idx_start + max_n_obs

        return np.arange(idx_start, idx_end)

    def __calc_grid_steps(self, tol):
        """
        Calculate the grid steps for F0 and F1 based on the specified tolerance.
        
        tol: float
            The tolerance of phase change per step. 
        """

        # Calculate the max time span of the observations
        max_dt = np.max(np.abs(self.epochs - self.model.PEPOCH.value)) * 86400 # in seconds

        # Step size in F0
        # (dphi = F0 * dt)
        df0_step = tol / max_dt

        # Step size in F1
        # (dphi = 0.5 * F1 * dt^2)
        df1_step = tol / (0.5 * max_dt**2)

        return df0_step, df1_step

    def __calc_grid_range(self, initial_f0, initial_f1):
        # Calculate F0 lower and upper bounds of one sidereal day aliasing
        n_phase_per_day = 86164.1 * initial_f0
        f0_lower = (n_phase_per_day - 0.5) / 86164.1
        f0_upper = (n_phase_per_day + 0.5) / 86164.1

        # Use an empirical range for F1
        f1_lower = -self.max_f1
        f1_upper = self.max_f1

        return f0_lower, f0_upper, f1_lower, f1_upper

    def optimize(self, max_n_df0_trials=2048, max_n_df1_trials=1024, tol=0.03):
        """
        Find an initial guess for the timing solution based on the loaded profiles and the model.

        max_n_df0_trials: int, optional
            The maximum number of F0 trials to search during the grid search.
        max_n_df1_trials: int, optional
            The maximum number of F1 trials to search during the grid search.
        tol: float, optional
            The tolerance of phase change per step used to calculate the grid step sizes.
        """

        # Initialize parameters
        initial_f0 = self.model.F0.value
        initial_f1 = self.model.F1.value
        pepoch = self.model.PEPOCH.value

        # Calculate search ranges
        f0_lower, f0_upper, f1_lower, f1_upper = self.__calc_grid_range(initial_f0, initial_f1)

        # Calculate step sizes
        df0_step, df1_step = self.__calc_grid_steps(tol)

        # Calculate the number of trials
        n_df0_trials = int(np.ceil((f0_upper - f0_lower) / df0_step))
        n_df1_trials = int(np.ceil((f1_upper - f1_lower) / df1_step))
        self.logger.debug(f"With tolerance {tol} of the phase, calculated n_df0_trials={n_df0_trials}, n_df1_trials={n_df1_trials}")

        # Cap the number of trials to the maximum allowed
        if n_df0_trials > max_n_df0_trials:
            self.logger.info(f"Capping n_df0_trials to max_n_df0_trials: {max_n_df0_trials}", layer=1)
            n_df0_trials = max_n_df0_trials
        if n_df1_trials > max_n_df1_trials:
            self.logger.info(f"Capping n_df1_trials to max_n_df1_trials: {max_n_df1_trials}", layer=1)
            n_df1_trials = max_n_df1_trials

        # Calculate the grid search range
        df0_vals=np.linspace(
            f0_lower - initial_f0,
            f0_upper - initial_f0,
            n_df0_trials
        )
        df1_vals=np.linspace(f1_lower - initial_f1, f1_upper - initial_f1, n_df1_trials)

        # Initialize PCS
        pcs = PhaseCoherentSearch(profiles=self.profiles, epochs=self.epochs, center_epoch=pepoch, max_width=self.max_width)
        best_pcs_state = pcs.search(df0_vals=df0_vals,df1_vals=df1_vals, ncpus=self.ncpus)

        return InitialTimingSolution(model=self.model, best_pcs_state=best_pcs_state, logger=self.logger)