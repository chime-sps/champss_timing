from ..utils.exec import exec
from ..utils.utils import utils
from ..io.archive import ArchiveReader

import os
import tqdm
import shutil
import multiprocessing

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

    def _zap_bad_channel__get_bad_channels(self, f, ext):
        zapfile_clfd, bad_percentage = ArchiveReader(f"{f}{ext}").get_bad_channels(output_format="clfd")
        open(f"{f}{ext}.zapfile", "w").write(zapfile_clfd)

        if bad_percentage > 0.5:
            self.logger.warning(f"[{f}{ext}] Bad channels exceed 50% ({bad_percentage * 100}%). ")
        else:
            self.logger.debug(f"[{f}{ext}] {bad_percentage * 100}% of channels are bad.")

        return bad_percentage

    def zap_bad_channel(self, fs, ext=""):
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
            # for f in fs:
            #     zapfile_clfd, bad_percentage = ArchiveReader(f"{f}{ext}").get_bad_channels(output_format="clfd")
            #     open(f"{f}{ext}.zapfile", "w").write(zapfile_clfd)

            #     if bad_percentage > 0.5:
            #         self.logger.warning(f"[{f}{ext}] Bad channels exceed 50% ({bad_percentage * 100}%). ")
            #     else:
            #         self.logger.debug(f"[{f}{ext}] {bad_percentage * 100}% of channels are bad.")

            #     self.bad_percentage.append(bad_percentage)

            get_bad_channels_params = []
            for f in fs:
                get_bad_channels_params.append((f, ext))

            with multiprocessing.Pool(self.n_pools) as pool:
                # self.bad_percentage = pool.starmap(self._zap_bad_channel__get_bad_channels, get_bad_channels_params)
                 self.bad_percentage = list(tqdm.tqdm(pool.starmap(self._zap_bad_channel__get_bad_channels, get_bad_channels_params), total=len(get_bad_channels_params)))

        # Scrunch pols (clfd requires pam -p first)
        self.scrunch(fs, scrunch_flag="p", ext="", overwrite=True)

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

    def dedisperse(self, fs, parfile):
        # Remove TZRSITE to fix a problem with psrchive for CHIME observations
        open(f"{parfile}.tmp", "w").write(
            open(parfile).read().replace("TZRSITE", "# TZRSITE")
        )

        # pam -d {dm} -e .dd1
        exec_hlr = self.exec_handler(n_pools=self.n_pools)
        for f in fs:
            exec_hlr.append(f"pam -E {parfile}.tmp --update_dm -e .dd1 {f} > {f}.dd1.log")
        exec_hlr.run()

        if(not exec_hlr.check()):
            raise Exception("Failed to dedisperse")

        # check if .dd1 exists
        dd1_files = []
        for f in fs:
            dd1_files.append(utils.no_extension(f) + ".dd1")
            if not os.path.exists(dd1_files[-1]):
                raise Exception("Failed to dedisperse (no .dd1 for %s)" % f)
        
        # move .dd1 to replace original file
        move_args = []
        for i, f in enumerate(fs):
            self.logger.debug(f"Overwriting {f} with {dd1_files[i]} (dedispersed archive)... ", layer=1)
            move_args.append((dd1_files[i], f))
        with multiprocessing.Pool(self.n_pools) as pool:
            list(tqdm.tqdm(pool.starmap(shutil.move, move_args), total=len(move_args)))
            
        # # pam -D -e .dd2
        # exec_hlr = self.exec_handler(n_pools=self.n_pools)
        # for f in dd1_files:
        #     exec_hlr.append(f"pam -D -e .dd2 {f}")
        # exec_hlr.run()

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to dedisperse")

        # # check if .dd2 exists
        # dd2_files = []
        # for f in dd1_files:
        #     dd2_files.append(utils.no_extension(f) + ".dd2")
        #     if not os.path.exists(dd2_files[-1]):
        #         raise Exception("Failed to dedisperse (no .dd2 for %s)" % f)

        # # move .dd2 to replace original file
        # move_args = []
        # for i, f in enumerate(fs):
        #     self.logger.debug(f"Overwriting {f} with {dd2_files[i]} (dedispersed archive)... ", layer=1)
        #     move_args.append((dd2_files[i], f))
        # with multiprocessing.Pool(self.n_pools) as pool:
        #     list(tqdm.tqdm(pool.starmap(shutil.move, move_args), total=len(move_args)))

        return True

    def dedisperse_legacy(self, fs, dm):
        # pam -d {dm} -e .dd1
        exec_hlr = self.exec_handler(n_pools=self.n_pools)
        for f in fs:
            exec_hlr.append(f"pam -d {dm} -e .dd1 {f} > {f}.dedisperse.log")
        exec_hlr.run()

        if(not exec_hlr.check()):
            raise Exception("Failed to dedisperse")

        # check if .dd1 exists
        dd1_files = []
        for f in fs:
            dd1_files.append(utils.no_extension(f) + ".dd1")
            if not os.path.exists(dd1_files[-1]):
                raise Exception("Failed to dedisperse (no .dd1 for %s)" % f)
            
        # # pam -D -e .dd2
        # exec_hlr = self.exec_handler(n_pools=self.n_pools)
        # for f in dd1_files:
        #     exec_hlr.append(f"pam -D -e .dd2 {f}")
        # exec_hlr.run()

        # if(not exec_hlr.check()):
        #     raise Exception("Failed to dedisperse")

        # # check if .dd2 exists
        # dd2_files = []
        # for f in dd1_files:
        #     dd2_files.append(utils.no_extension(f) + ".dd2")
        #     if not os.path.exists(dd2_files[-1]):
        #         raise Exception("Failed to dedisperse (no .dd2 for %s)" % f)

        # # move .dd2 to replace original file
        # move_args = []
        # for i, f in enumerate(fs):
        #     self.logger.debug(f"Overwriting {f} with {dd2_files[i]} (dedispersed archive)... ", layer=1)
        #     move_args.append((dd2_files[i], f))
        # with multiprocessing.Pool(self.n_pools) as pool:
        #     list(tqdm.tqdm(pool.starmap(shutil.move, move_args), total=len(move_args)))

        # move .dd2 to replace original file
        move_args = []
        for i, f in enumerate(fs):
            self.logger.debug(f"Overwriting {f} with {dd1_files[i]} (dedispersed archive)... ", layer=1)
            move_args.append((dd1_files[i], f))
        with multiprocessing.Pool(self.n_pools) as pool:
            list(tqdm.tqdm(pool.starmap(shutil.move, move_args), total=len(move_args)))

        raise Exception("stop")

        return True