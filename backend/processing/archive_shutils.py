import os
import subprocess
import glob
import copy
import signal
import time

from ..utils.utils import  utils

class archive_shutils:
    def __init__(self, archive):
        '''
        Initialize the archive_shutils class.

        Parameters
        ----------
        archive : str
            Path to the archive file.
        '''

        self.archive = archive

        # Check if the archive exists
        if not os.path.exists(archive):
            raise FileNotFoundError(f"Archive {archive} does not exist.")

    def scrunch(self, freq=False, time=False, pol=False, ext=None, return_format="filename"):
        '''
        Scrunch the archive. 

        Parameters
        ----------
        freq : bool
            Scrunch frequency (default: False)
        time : bool
            Scrunch time (default: False)
        pol : bool
            Scrunch polarization (default: False)
        ext : str, optional
            Extension of the output archive. If None, the file will be overwritten.
        return_format : str, optional
            Format of the return value. Can be "filename" or "shutils". Default is "filename".

        Returns
        -------
        str
            The path of new archive file.
        '''

        # Get tempfile ID
        tmp_id, logfile = self.__get_tmp_id_and_logfile(self.archive, action="scrunch")

        # Setup scrunch flag
        scrunch_flag = "-"
        if freq: 
            scrunch_flag += "F"
        if time: 
            scrunch_flag += "T"
        if pol: 
            scrunch_flag += "p"

        # Setup the command
        cmd = ['pam', scrunch_flag , "-e", tmp_id, self.archive]

        # Execute the command
        self.__run(cmd, logfile=logfile)
        
        # Finish the processing
        outfile = self.__move_tempfile_to_outfile(
            infile=self.archive, 
            tempfile=self.__rename_with_extension(self.archive, tmp_id, keep_existing=False), 
            logfile=logfile, 
            ext=ext, 
            action="install parfile"
        )

        return self.__format_return(outfile, return_format)

        

    def clfd(self, ext=None, return_format="filename"):
        '''
        RFI cleanning by clfd package. 

        Parameters
        ----------
        ext : str, optional
            Extension of the output archive. If None, the file will be overwritten.

        Returns
        -------
        str
            The path of new archive file.
        '''

        # Get tempfile ID
        tmp_id, logfile = self.__get_tmp_id_and_logfile(self.archive, action="clfd")

        # Setup the command
        cmd = ['clfd', "--no-report" , "-e", tmp_id, "-p", "1", self.archive]

        # Execute the command
        self.__run(cmd, logfile=logfile)

        # Finish the processing
        outfile = self.__move_tempfile_to_outfile(
            infile=self.archive, 
            tempfile=self.__rename_with_extension(self.archive, tmp_id, keep_existing=True), 
            logfile=logfile, 
            ext=ext, 
            action="clean RFI"
        )

        return self.__format_return(outfile, return_format)
    
    def install_parfile(self, parfile, jump=0.0, ext=None, return_format="filename"):
        '''
        Install a parfile from the archive (pam -E). 

        Parameters
        ----------
        parfile : str
            Path to the parfile to be installed.
        jump : float
            Phase jump to be applied to the archive. (default: 0.0)
            *Note the this function do not read jump from the parfile.*
        ext : str, optional
            Extension of the output archive. If None, the file will be overwritten.
        return_format : str, optional
            Format of the return value. Can be "filename" or "shutils". Default is "filename".

        Returns
        -------
        str
            The path of new archive file.
        '''

        # Get tempfile ID
        tmp_id, logfile = self.__get_tmp_id_and_logfile(self.archive, action="ephm_install")

        # Check if the parfile exists
        if not os.path.exists(parfile):
            raise FileNotFoundError(f"Parfile {parfile} does not exist.")

        # remove TZRSITE to fix a problem with psrchive for CHIME observations
        open(parfile + "." + tmp_id, "w").write(
            open(parfile).read().replace("TZRSITE", "# TZRSITE")
        )

        # Set up the command to install the parfile
        cmd = ['pam', '-E', parfile + "." + tmp_id, "-e", tmp_id, self.archive]

        # Execute the command
        self.__run(cmd, logfile=logfile)

        # Finish the processing
        outfile = self.__move_tempfile_to_outfile(
            infile=self.archive, 
            tempfile=self.__rename_with_extension(self.archive, tmp_id, keep_existing=False), 
            logfile=logfile, 
            ext=ext, 
            action="install parfile"
        )

        # Apply the jump if needed
        if jump != 0.0:
            # Apply the jump
            outfile = self.copy_shutils(new_filename=outfile).apply_jump(parfile, jump)

        return self.__format_return(outfile, return_format)

    def apply_jump(self, parfile, jump, ext=None, return_format="filename"):
        '''
        Apply a phase jump to the archive.

        Parameters
        ----------
        parfile : str
            Path to the parfile to be used for the jump.
        jump : float
            Phase jump to be applied to the archive.
        ext : str, optional
            Extension of the output archive. If None, the file will be overwritten.
        return_format : str, optional
            Format of the return value. Can be "filename" or "shutils". Default is "filename".

        Returns
        -------
        str
            The path of new archive file.
        '''

        # Get tempfile ID
        tmp_id, logfile = self.__get_tmp_id_and_logfile(self.archive, action="jump")

        # Get F0 from parfile
        f0 = None
        with open(parfile, "r") as f:
            for l in f:
                if l.strip().startswith("F0"):
                    f0 = float(l.strip().split()[1])
                    break
        
        # Check if F0 was found
        if f0 is None:
            raise Exception("Failed to read F0 from parfile")
        
        # Calculate phase offset
        if jump < 0:
            jump = (1/f0) + jump
        phase_offset = - ((jump / (1/f0)) % 1)

        # Setup the command to apply the jump
        cmd = ['pam', '-m', '-r', str(phase_offset), "-e", tmp_id, self.archive]

        # Execute the command
        self.__run(cmd, logfile=logfile)

        # Finish the processing
        outfile = self.__move_tempfile_to_outfile(
            infile=self.archive, 
            tempfile=self.__rename_with_extension(self.archive, tmp_id, keep_existing=False), 
            logfile=logfile, 
            ext=ext, 
            action="apply jump"
        )

        return self.__format_return(outfile, return_format)
    
    def copy_shutils(self, new_filename=None):
        '''
        Create a copy of this archive_shutils object. 

        Parameters
        ----------
        new_filename : str, optional
            New filename for the copied object. If None, the original filename is used.

        Returns
        -------
        archive_shutils
        '''

        # Copy self
        self_ = copy.deepcopy(self)

        # Overwrite the archive name into the new one
        if new_filename is not None:
            self_.archive = new_filename

        return self_
    
    def __format_return(self, outfile, type):
        if type == "filename":
            return outfile
        elif type == "shutils":
            return self.copy_shutils(new_filename=outfile)
        else:
            raise ValueError("Unknown type for return format.")

    # def __run(self, cmd, logfile=None):
    #     '''
    #     Run a command in the shell.

    #     Parameters
    #     ----------
    #     cmd : list or str
    #         The command to run. If shell features are needed (e.g., redirection), pass as a string.
    #     logfile : str or None
    #         Path to a logfile. If provided, stdout and stderr will be redirected there.
    #     '''
    #     try:
    #         if logfile:
    #             with open(logfile, 'w') as logf:
    #                 subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True, shell=isinstance(cmd, str))
    #         else:
    #             result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, shell=isinstance(cmd, str))
    #             return result.stdout.decode('utf-8')  # Optional: return the output if no logfile is used
    #     except subprocess.CalledProcessError as e:
    #         if logfile:
    #             print(f"Command failed. See {logfile} for details.")
    #         else:
    #             print(f"Command failed with error: {e.stderr.decode('utf-8')}")
    #         raise

    def __run(self, cmd, logfile=None, timeout=300, retries=1, retry_delay=5):
        '''
        Run a command in the shell with timeout and retry on OOM kill.

        Parameters
        ----------
        cmd : list or str
            The command to run. If shell features are needed (e.g., redirection), pass as a string.
        logfile : str or None
            Path to a logfile. If provided, stdout and stderr will be redirected there.
        timeout : int
            Maximum number of seconds to allow the process to run.
        retries : int
            Number of times to retry if process was OOM killed.
        retry_delay : int
            Seconds to wait before retrying.
        '''
        attempt = 0
        shell_flag = isinstance(cmd, str)

        while attempt <= retries:
            try:
                if logfile:
                    with open(logfile, 'w') as logf:
                        subprocess.run(
                            cmd,
                            stdout=logf,
                            stderr=subprocess.STDOUT,
                            check=True,
                            shell=shell_flag,
                            timeout=timeout
                        )
                else:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        shell=shell_flag,
                        timeout=timeout
                    )
                    return result.stdout.decode('utf-8')

                return  # success, exit
            except subprocess.TimeoutExpired:
                print(f"Command timed out after {timeout} seconds.")
                raise
            except subprocess.CalledProcessError as e:
                # Detect if killed by signal 9 (SIGKILL)
                if e.returncode == -signal.SIGKILL:
                    print(f"Command likely OOM-killed (SIGKILL). Attempt {attempt + 1} of {retries}.")
                    attempt += 1
                    if attempt > retries:
                        print("Max retries reached. Giving up.")
                        raise
                    else:
                        time.sleep(retry_delay)
                else:
                    if logfile:
                        print(f"Command failed. See {logfile} for details.")
                    else:
                        print(f"Command failed with error: {e.stderr.decode('utf-8')}")
                    raise

    def __rename_with_extension(self, filename, ext, keep_existing=True):
        '''
        Rename the archive file with a new extension.

        Parameters
        ----------
        filename : str

        ext : str
            The new extension for the archive file.
        keep_existing : bool, optional
            If True, keep the existing extension. If False, overwrite it.
            Default is True.
        '''

        # Check if the new extension is valid
        if not ext.startswith('.'):
            ext = '.' + ext

        # Remove the existing extension if keep_existing is False
        if not keep_existing:
            filename = ".".join(filename.split('.')[:-1])

        return filename + ext

    def __move_tempfile_to_outfile(self, infile, tempfile, logfile, ext, action="unknown"):
        # Check if the temporary file exists
        if not os.path.exists(tempfile):
            raise RuntimeError(f"Failed to {action} as the temporary file was not created by PSRCHIVE, please check the log file {logfile} for details.")

        # Create the output file name
        if ext == None:
            outfile = infile
        else:
            outfile = self.__rename_with_extension(in_file, ext)

        # Rename
        os.rename(tempfile, outfile)

        # Remove the log file
        if os.path.exists(logfile):
            os.unlink(logfile)

        return outfile

    def __get_tmp_id_and_logfile(self, in_file, action="unknown"):
        tmp_id = "tmp_" + utils.get_rand_string()
        logfile = in_file + "__" + tmp_id + "." + action + ".log"

        return tmp_id, logfile