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
    def __init__(self, parfile, archive_files, logger=logger()):
        self.model = get_model(parfile)
        self.logger = logger
        self.profiles, self.epochs = self.__load_archives(archive_files, self.model)

    def __load_archives(self, archive_files, model):
        """Helper function to load multiple archives."""
        profiles = []
        epochs = []
        for archive_file in archive_files:
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

            profiles.append(profile)
            epochs.append(epoch)

        return profiles, epochs

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
        pcs = PhaseCoherentSearch(profiles=self.profiles, epochs=self.epochs, center_epoch=pepoch)
        best_pcs_state = pcs.search(df0_vals=df0_vals,df1_vals=df1_vals, ncpus=ncpus)

        return InitialTimingSolution(model=self.model, best_pcs_state=best_pcs_state, logger=self.logger)