import os
from io import StringIO

from backend.tools.initial_guess import InitialTimingSolutionOptimizer
from backend.utils.utils import utils
from backend.utils.logger import logger

class CLIInitialTimingSolutionSearch:
    def __init__(self, archive_files, parfile=None, params=None, ncpus=1, logger=logger()):
        if parfile is None and params is None:
            raise ValueError("Either parfile or model parameters must be provided.")
            
        self.archive_files = archive_files
        self.logger = logger
        self.ncpus = ncpus

        # Create parfile if it is not provided but model parameters are given
        if parfile is None:
            self.parfile = utils.create_parfile(**params)
        else:
            self.parfile = open(parfile, 'r').read()

        # Initialize optimizer
        self.optimizer = InitialTimingSolutionOptimizer(
            StringIO(self.parfile), 
            self.archive_files, 
            logger=logger.copy()
        )

    def optimize(self, output_dir, make_directory=False):
        # Create output directory if it doesn't exist and make_directory is True
        if make_directory:
            self.logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        self.logger.success(f"Saving initial timing solution to {output_dir}. ")
        plot_path = f"{output_dir}/initial.pdf"
        parfile_path = f"{output_dir}/initial.par"
        archive_path = f"{output_dir}/initial.ar"

        # Optimize the initial timing solution
        solution = self.optimizer.optimize(ncpus=self.ncpus)
        solution.plot(savefig=plot_path)
        self.logger.success(f"Diagnostic plot -> {plot_path}. ", layer=1)
        solution.write_parfile(parfile_path)
        self.logger.success(f"Parfile -> {parfile_path}. ", layer=1)
        solution.write_archive(archive_path)
        self.logger.success(f"Archive -> {archive_path}. ", layer=1)