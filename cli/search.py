import os
from io import StringIO

from backend.tools.initial_guess import InitialTimingSolutionOptimizer
from backend.io.archive import ArchiveReader
from backend.utils.utils import utils
from backend.utils.logger import logger

# Mute logging from PINT to avoid flushing the terminal with too many messages
from pint import logging
logging.setup(level="ERROR")

class CLIInitialTimingSolutionSearch:
    def __init__(self, archive_files, parfile=None, params=None, max_n_obs=60, max_duty_cycle=None, search_mode="default", ncpus=1, logger=logger()):
        self.archive_files = archive_files
        self.logger = logger
        self.ncpus = ncpus
        self.max_n_obs = max_n_obs
        self.max_duty_cycle = max_duty_cycle
        self.search_mode = search_mode

        if params is not None and params["ra"] is not None: # Create parfile if parameters are provided
            self.logger.info(f"Creating parfile from provided model parameters.")
            self.parfile = utils.create_parfile(**params)
        elif parfile is not None: # Use the provided parfile instead if a path to parfile is provided
            self.logger.info(f"Using parfile: {parfile}")
            self.parfile = open(parfile, 'r').read()
        else: # If neither parfile nor parameters are provided, use ephemeris from the first archive file
            self.logger.info(f"No parfile or model parameters provided. Using ephemeris from the first archive file: {archive_files[0]}. ")
            self.parfile = ArchiveReader(archive_files[0]).get_ephem()

        # Setup search parameters
        self.search_tol = 0.03
        self.search_max_f1 = 2e-12
        self.search_reset_f1 = True

        if search_mode == "fine":
            self.search_tol = 0.01
            self.search_max_f1 = 1e-12
            self.search_reset_f1 = False
        
        if search_mode == "coarse":
            self.search_tol = 0.05
            self.search_max_f1 = 1e-11
            self.search_reset_f1 = True

        # Initialize optimizer
        self.optimizer = InitialTimingSolutionOptimizer(
            StringIO(self.parfile), 
            self.archive_files, 
            max_n_obs=self.max_n_obs,
            max_duty_cycle=self.max_duty_cycle,
            ncpus=self.ncpus, 
            max_f1=self.search_max_f1,
            reset_f1=self.search_reset_f1,
            logger=logger.copy()
        )

    def optimize(self, output_dir, make_directory=False, preview=False):
        # Create output directory if it doesn't exist and make_directory is True
        if make_directory:
            self.logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Saving initial timing solution to {output_dir}. ")
        plot_path = f"{output_dir}/search.pdf"
        parfile_path = f"{output_dir}/search.par"
        archive_path = f"{output_dir}/search.ar"

        # Optimize the initial timing solution
        solution = self.optimizer.optimize(tol=self.search_tol)
        solution.plot(savefig=plot_path)
        self.logger.success(f"Diagnostic plot -> {plot_path}. ", layer=1)
        solution.write_parfile(parfile_path)
        self.logger.success(f"Parfile -> {parfile_path}. ", layer=1)
        solution.write_archive(archive_path)
        self.logger.success(f"Archive -> {archive_path}. ", layer=1)

        if preview:
            pdf_viewer = None
            for viewer in ["mupdf", "evince", "okular", "xdg-open"]:
                if os.system(f"which {viewer}") == 0:
                    pdf_viewer = viewer
                    break
            if pdf_viewer is not None:
                os.system(f"{pdf_viewer} {output_dir}/search.pdf &")
            else:
                self.logger.warning(f"No PDF viewer found to preview the diagnostic plot at {output_dir}/search.pdf.")