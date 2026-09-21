import subprocess
import os
import shutil

class CLINewPulsar:
    def __init__(self, sourcedir):
        self.sourcedir = sourcedir

        # Make sure the source directory exists
        if not os.path.exists(self.sourcedir):
            raise FileNotFoundError(f"The source directory '{self.sourcedir}' does not exist.")

        self.psrdir = None
        self.psrdir_tmp = None

    def __copy_to_psrdir(self, src, dest, temp=True):
        if not os.path.exists(src):
            print(f"The file '{src}' does not exist.")
            return False

        print(f"Copying '{src}' to '{self.get_path(dest, temp=temp)}'...")
        shutil.copy(src, self.get_path(dest, temp=temp))

        return True

    def __create_text_file_in_psrdir(self, filename, temp=True):
        editor = os.environ.get("EDITOR", None)

        if editor is None:
            if shutil.which('nano') is not None:
                editor = 'nano'
            elif shutil.which('vi') is not None:
                editor = 'vi'
            else:
                print("No default text editor found.")
                editor = input("Please enter the command for your preferred text editor: ")

        print(f"Opening '{self.get_path(filename, temp=temp)}' with editor '{editor}'...")
        os.system(f"{editor} {self.get_path(filename, temp=temp)}")

        # Check if the file was created successfully
        if not os.path.exists(self.get_path(filename, temp=temp)):
            print(f"File '{filename}' was not created.")
            return False

        print(f"File '{filename}' created successfully in '{self.get_path(filename, temp=temp)}'.")

        return True
        
    def get_psrdir(self, temp=True):
        if self.psrname is None:
            raise ValueError("Pulsar name has not been set.")

        path = self.psrdir_tmp if temp else self.psrdir

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def get_path(self, filename, temp=True):
        return os.path.join(self.get_psrdir(temp=temp), filename)

    def add_psr_name(self, psrname):
        # Make sure the pulsar name is not empty
        if not psrname:
            return False

        # Format name
        psrname = psrname.strip()

        # Make sure no existing pulsar directory with the same name exists
        if os.path.exists(f"{self.sourcedir}/{psrname}"):
            print(f"A pulsar with the name '{psrname}' already exists.")
            return False

        # Make sure the name is valid containing no characters that are not allowed in directory names
        if any(c in psrname for c in r'\/:*?"<>|'):
            print(f"The pulsar name '{psrname}' contains invalid characters.")
            return False

        # Initialize pulsar name attribute
        self.psrname = psrname
        self.psrdir = f"{self.sourcedir}/{self.psrname}"
        self.psrdir_tmp = f"{self.sourcedir}/__new_{self.psrname}"

        return True

    def add_psr_parfile(self, parfile_path):
        return self.__copy_to_psrdir(parfile_path, "pulsar.par")

    def create_psr_parfile(self):
        return self.__create_text_file_in_psrdir("pulsar.par")

    def add_psr_stdfile(self, stdfile_path):
        return self.__copy_to_psrdir(stdfile_path, "paas.std")

    def create_psr_stdfile(self, archive_file):
        os.system(f"paas -i {archive_file} -s {self.get_path('paas.std')} -w {self.get_path('paas.m')} -j {self.get_path('paas.txt')}")

        if not os.path.exists(self.get_path("paas.std")):
            return False

        return True

    def add_psr_config(self, config):
        return self.__copy_to_psrdir(config, "champss_timing.config")

    def load_from_existing_pulsar(self, psr):
        config_path = os.path.join(self.sourcedir, psr, "champss_timing.config")
        if not os.path.exists(config_path):
            return False
        
        return self.add_psr_config(config_path)

    def create_psr_config(self):
        return self.__create_text_file_in_psrdir("champss_timing.config")

    def commit(self):
        if self.psrdir_tmp and os.path.exists(self.psrdir_tmp):
            # Check if all required files are present in the temporary pulsar directory
            required_files = ["pulsar.par", "paas.std"]
            for f in required_files:
                if not os.path.exists(self.get_path(f, temp=True)):
                    raise FileNotFoundError(f"Required file '{f}' is missing in the temporary pulsar directory.")

            # Remove the existing pulsar directory if it exists before moving the temporary directory
            if os.path.exists(self.psrdir):
                input(f"The pulsar directory '{self.psrdir}' already exists. Press Enter to overwrite it...")
                shutil.rmtree(self.psrdir)

            # Commit
            shutil.move(self.psrdir_tmp, self.psrdir)