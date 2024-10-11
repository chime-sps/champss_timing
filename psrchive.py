from .exec import exec

import os

class psrchive_handler():
    def __init__(self, self_super):
        self.exec_handler = self_super.exec_handler
        self.n_pools = self_super.n_pools
        self.logger = self_super.logger

        # Check for whether commands exists
        self.logger("Initializing PSRCHIVE modules... ")
        self.cmd_checklist = ["get_bad_channel_list.py", "clfd", "pam", "pat"]
        self.cmd_check()
        self.logger("Initializing PSRCHIVE modules... Done. ")
        
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
        # get_bad_channel_list.py
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/get_bad_channel_list.log")
        for f in fs:
            output_files.append(f"{f}{ext}.zapfile")
            exec_hlr.append(f"get_bad_channel_list.py --fmt clfd --type timer --out {f}{ext}.zapfile {f}{ext}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to get bad channel list")
        
        # if(not exec_hlr.check()):
        #     raise Exception("Failed to get bad channel list")
        
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
    
    def scrunch(self, fs, scrunch_flag="FTp", ext=".clfd"):
        # pam -e
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/pam.log")
        for f in fs:
            output_files.append(f"{f}{ext}.{scrunch_flag}")
            exec_hlr.append(f"pam -e {ext}.{scrunch_flag} -{scrunch_flag} {f}{ext}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to scrunch")

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to scrunch")
        
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
        

    def update_model(self, fs, parfile, ext=".clfd.FTp"):
        # pam -e .pam -E pulsar.par xxx.ar.clfd.FTp
        output_files = []
        exec_hlr = self.exec_handler(n_pools=self.n_pools, log=self._get_log_path(fs) + "/pam_update.log")
        for f in fs:
            output_files.append(f"{f}{ext}.pam")
            exec_hlr.append(f"pam -e .FTp.pam -E {parfile} {f}{ext}")
        exec_hlr.run()

        # Check output files exist
        if not self.file_check(output_files):
            raise Exception("Failed to update model")
        
        # if(not exec_hlr.check()):
        #     raise Exception("Failed to update model")

        return True
        