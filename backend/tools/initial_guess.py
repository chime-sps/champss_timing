import numpy as np
from pint.models import get_model
from ..utils.phase_coherent_search import PhaseCoherentSearch
from ..utils.logger import logger

from ..tools.ephm_install import EphmInstall
from ..io.archive import ArchiveReader

class InitialTimingSolution:
    def __init__(self, model, best_pcs_state, logger=logger()):
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
    def __init__(self, parfile, archive_files, max_n_obs=None, max_duty_cycle=None, logger=logger()):
        self.logger = logger
        self.max_n_obs = max_n_obs
        self.max_duty_cycle = max_duty_cycle
        self.max_width = None

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
        center_epoch = (np.max(self.epochs) + np.min(self.epochs)) / 2
        self.logger.debug(f"Changing PEPOCH from {self.model.PEPOCH.value} to {center_epoch} (the middle of the observation span)")
        self.model.change_pepoch(center_epoch)

    def __get_model(self, parfile):
        """Helper function to get model and ensure the model is valid."""
        m = get_model(parfile)

        if "Spindown" not in m.components:
            raise ValueError("Model must have a Spindown component.")
        
        c = m.components["Spindown"]
        if not hasattr(m, "F1"):
            self.logger.debug("F1 attribute not found in the model. Adding F1...", layer=1)
            c.add_param(m.F0.new_param(1), setup=True)
            p = getattr(m, "F1")
            p.quantity = 0.0 * p.units
            p.frozen = True

        return m


    def __load_archives(self, archive_files, model, max_n_obs):
        """Helper function to load multiple archives."""
        files = []
        profiles = []
        epochs = []
        for i, archive_file in enumerate(archive_files):
            if i > 14: # Show progress when too many archives to be loaded
                self.logger.debug(f"It may take a while to load {len(archive_files)} archives ({len(archive_files) - i} left)... ", layer=1, end="\r")

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

            files.append(archive_file)
            profiles.append(profile)
            epochs.append(epoch)
        

        if i > 14:
            self.logger.debug(f"\nFinished loading {len(archive_files)} archives.", layer=1)

        # Sort the profiles and epochs by epochs
        sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
        files = [files[i] for i in sorted_indices]
        profiles = [profiles[i] for i in sorted_indices]
        epochs = [epochs[i] for i in sorted_indices]

        # Limit the number of observations if max_n_obs is specified
        if max_n_obs is not None:
            files = files[:max_n_obs]
            profiles = profiles[:max_n_obs]
            epochs = epochs[:max_n_obs]

        return files, profiles, epochs

    def optimize(self, n_df0_trials=512, n_df1_trials=256, ncpus=1):
        """
        Find an initial guess for the timing solution based on the loaded profiles and the model.
        """

        # Initialize parameters
        initial_f0 = self.model.F0.value
        initial_f1 = self.model.F1.value
        pepoch = self.model.PEPOCH.value

        # Calculate F0 lower and upper bounds of one sidereal day aliasing
        n_phase_per_day = 86164.1 * initial_f0
        f0_lower = (n_phase_per_day + 0.5) / 86164.1
        f0_upper = (n_phase_per_day - 0.5) / 86164.1

        # Calculate the grid search range
        df0_vals=np.linspace(
            f0_lower - initial_f0,
            f0_upper - initial_f0,
            n_df0_trials
        )
        df1_vals=np.linspace(-2e-12, 2e-12, n_df1_trials)

        # Initialize PCS
        pcs = PhaseCoherentSearch(profiles=self.profiles, epochs=self.epochs, center_epoch=pepoch, max_width=self.max_width)
        best_pcs_state = pcs.search(df0_vals=df0_vals,df1_vals=df1_vals, ncpus=ncpus)

        return InitialTimingSolution(model=self.model, best_pcs_state=best_pcs_state, logger=self.logger)