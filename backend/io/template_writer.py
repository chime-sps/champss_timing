import numpy as np
import psrchive
import zipfile
import os
import shutil

class TemplateWriter:
    def __init__(self, filename, template=None, overwrite=True):
        """
        Initialize the ArchiveWriter with a filename and a template.
        :param filename: The name of the output file.
        :param template: The template file to use for the archive.
        """

        # Get the template file
        if template is None:
            template = os.path.join(os.path.dirname(__file__), "data/template.ar")

        # Set the overwrite flag
        self.overwrite = overwrite

        # Check if the template has to be unzipped
        if not os.path.exists(template):
            if os.path.exists(template + ".zip"):
                with zipfile.ZipFile(template + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(template))
            else:
                raise FileNotFoundError(f"Template file {template} not found and no zip file found. Please check the installation :(")
            

        # Set the filename
        self.filename = filename

        # Check if the file already exists
        if os.path.exists(self.filename):
            if self.overwrite:
                os.remove(self.filename)
            else:
                raise FileExistsError(f"File {self.filename} already exists. Use overwrite=True to overwrite it.")
            

        # Copy the template file to the output file
        shutil.copyfile(template, self.filename)

        # Load the template archive
        self.archive = psrchive.Archive.load(self.filename)
        self.subint = self.archive.get_Integration(0)

    def write(self, data, interpolate=False):
        """
        Write the data to the archive.
        :param data: The data to write to the archive. The data should be a 1D numpy array with length of 512. 
        :param interpolate: If True, interpolate into 512 samples the data before writing.
        """
        
        # Make sure the data is 512 samples long
        if len(data) != 512:
            if interpolate:
                # Interpolate the data to 512 samples
                data = np.interp(np.linspace(0, len(data), 512), np.arange(len(data)), data)
            else:
                raise ValueError("Data must be a numpy array with length of 512. Use interpolate=True to interpolate the data.")
            
        # Overwrite the data into the archive
        for i_pol in range(self.subint.get_npol()):
            for i_chan in range(self.subint.get_nchan()):
                self.subint.get_Profile(i_pol, i_chan).get_amps()[:] = data[:]

    def set_dm(self, dm):
        """
        Set the dispersion measure (DM) for the archive.
        :param dm: The DM value to set.
        """
        
        # Set the DM value
        self.archive.set_dispersion_measure(dm)

    def set_source(self, source):
        """
        Set the source name for the archive.
        :param source: The source name to set.
        """
        
        # Set the source name
        self.archive.set_source(source)

    def unload(self):
        """
        Unload the archive and free up resources.
        """

        # Unload the archive
        self.archive.unload()

    def __enter__(self):
        """
        Enter the context manager.
        :return: The ArchiveWriter instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager and unload the archive.
        :param exc_type: The type of exception raised.
        :param exc_value: The value of the exception raised.
        :param traceback: The traceback object.
        """
        self.unload()