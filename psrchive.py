from .exec import exec
from .archive_utils import archive_utils

import os

class psrchive_handler():
    def __init__(self, self_super, use_get_bad_channel_list_py=False):
        self.exec_handler = self_super.exec_handler
        self.n_pools = self_super.n_pools
        self.logger = self_super.logger.copy()
        self.use_get_bad_channel_list_py = use_get_bad_channel_list_py
        self.bad_percentage = []

        # Check for whether commands exists
        self.logger.debug("Initializing PSRCHIVE modules... ")
        self.cmd_checklist = ["clfd", "pam", "pat"]
        if self.use_get_bad_channel_list_py:
            self.cmd_checklist.append("get_bad_channel_list.py")
        self.cmd_check()
        self.logger.debug("Initializing PSRCHIVE modules... Done. ")
        
    def _get_log_path(self, fs):
        # get parent directory of the first file
        return "/".join(fs[0].split("/")[:-1])

    def cmd_check(self):
        exec_hlr = self.exec_handler(n_pools=self.n_pools)
        
        for cmd in self.cmd_checklist:
            exec_hlr.append("which %s >/dev/null 2>&1 || { echo \"0\"; }" % cmd)

        for i, res in enumerate(exec_hlr.run()):
            if (res["stdout"][-1].strip() == "0"):
                raise Exception("Command %s not found" % self.cmd_checklist[i])
            
        return True
    
    def file_check(self, files):
        for f in files:
            if not os.path.exists(f):
                return False

        return True

    def zap_bad_channel(self, fs, ext=""):
        # Scrunch pols (clfd requires pam -p first)
        self.scrunch(fs, scrunch_flag="p", ext="", overwrite=True)

        # get_bad_channel_list.py
        if self.use_get_bad_channel_list_py:
            self.logger.debug("Getting bad channel list... (using get_bad_channel_list.py)")
            output_files = []
            exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/get_bad_channel_list.log")
            for f in fs:
                output_files.append(f"{f}{ext}.zapfile")
                exec_hlr.append(f"get_bad_channel_list.py --fmt clfd --type timer --out {f}{ext}.zapfile {f}{ext}")
                self.bad_percentage.append(0) # Not getting bad percentage from get_bad_channel_list.py, but we can trust it.
            exec_hlr.run()

            # Check output files exist
            if not self.file_check(output_files):
                raise Exception("Failed to get bad channel list")

            # if(not exec_hlr.check()):
            #     raise Exception("Failed to get bad channel list")
        else:
            self.logger.debug("Getting bad channel list... (using internal method)")
            for f in fs:
                zapfile_clfd, bad_percentage = archive_utils(f"{f}{ext}").get_bad_channels(output_format="clfd")
                open(f"{f}{ext}.zapfile", "w").write(zapfile_clfd)

                if bad_percentage > 0.5:
                    self.logger.warning(f"Bad channels exceed 50% ({bad_percentage * 100}%) in {f}{ext}")
                else:
                    self.logger.debug(f"{bad_percentage * 100}% of channels are bad.")

                self.bad_percentage.append(bad_percentage)
        
        # clfd
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/clfd.log")
        for f in fs:
            output_files.append(f"{f}{ext}.zapfile")
            exec_hlr.append(f"clfd -z {f}{ext}.zapfile -e clfd --no-report {f}{ext}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to zap bad channels")

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to zap bad channels")
        
        return True
    
    def scrunch(self, fs, scrunch_flag="FTp", ext=".clfd", overwrite=False):
        # pam -e
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/pam.log")
        for f in fs:
            if overwrite:
                output_files.append(f)
                exec_hlr.append(f"pam -m -{scrunch_flag} {f}{ext}")
            else:
                output_files.append(f"{f}{ext}.{scrunch_flag}")
                exec_hlr.append(f"pam -e {ext}.{scrunch_flag} -{scrunch_flag} {f}{ext}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to scrunch")

        if(not exec_hlr.check()):
            raise Exception("Failed to scrunch")
        
        return True
    
    def get_toas(self, fs, template="paas.std", output="pulsar.tim", ext=".clfd.FTp"):
        # pat -A FDM -f "tempo2" -s
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/pat.log")
        for f in fs:
            output_files.append(f"{f}{ext}.tim")
            exec_hlr.append(f"pat -A FDM -f \"tempo2\" -s {template} {f}{ext} > {f}{ext}.tim")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to get TOAs")

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to get TOAs")
        
        # Merge TOAs
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/pat_merge.log")
        for f in fs:
            exec_hlr.append(f"awk FNR!=1 {f}{ext}.tim >> {output}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check([output]):
            raise Exception("Failed to merge TOAs")

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to merge TOAs")
        
        return True
        