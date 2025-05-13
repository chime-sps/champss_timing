import os
import numpy as np
import subprocess
import multiprocessing

class dspsr_shutils:
    def __init__(self, n_pools="auto"):
        self.n_pools = n_pools

        if self.n_pools == "auto":
            # If slurm environment
            try:
                self.n_pools = int(os.environ["SLURM_CPUS_PER_TASK"])
                if self.n_pools <= 0:
                    raise Exception("SLURM_CPUS_PER_TASK <= 0")
                print(f"SLURM environment detected. ")
            except:
                self.n_pools = multiprocessing.cpu_count()

    def fil2ar(self, fin, fout, parfile, nbin=512): # Adopted from CHAMPSS Software (https://github.com/chime-sps/champss_software/blob/main/champss/folding/folding/fold_candidate.py)
        # read f0 from parfile
        f0 = None
        with open(parfile, "r") as f:
            for l in f:
                if l.strip().startswith("F0"):
                    f0 = float(l.strip().split()[1])
                    break
        
        if f0 is None:
            raise Exception("Failed to read F0 from parfile")

        # Remove TZRSITE to fix a problem with psrchive for CHIME observations
        open(f"{parfile}.fil2ar.tmp", "w").write(
            open(parfile).read().replace("TZRSITE", "# TZRSITE")
        )

        # set number of turns, roughly equalling 10s 
        turns = int(np.ceil(10 * f0))
        if turns <= 2:
            intflag = "-turns"
        else:
            intflag = "-L"
            turns = 10

        # dspsr -t 1 -L 10 -A -k chime -E ../pulsar.par -O ${f}.ar ${f} 
        subprocess.run(
            [
                "dspsr",
                "-t",
                f"{self.n_pools}",
                f"{intflag}",
                f"{turns}",
                "-A",
                "-k",
                "chime",
                "-b", 
                f"{nbin}",
                "-E",
                f"{parfile}.fil2ar.tmp",
                "-O",
                f"{fout}",
                f"{fin}",
                f">{fout}.fil2ar.log",
                f"2>{fout}.fil2ar.err"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        # Check output files exist
        if not os.path.exists(f"{fout}.ar"):
            raise Exception(f"Failed to convert fil to ar, please check the log file -> {fout}.fil2ar.log and error file -> {fout}.fil2ar.err")

        # Remove temp parfile
        if os.path.exists(f"{parfile}.fil2ar.tmp"):
            os.remove(f"{parfile}.fil2ar.tmp")

# du = dspsr_shutils(n_pools=8)
# du.fil2ar("/home/wenkexia/wenkexia/testfiles_fil/test.fil", "/home/wenkexia/wenkexia/testfiles_fil/test.fil", "/home/wenkexia/wenkexia/testfiles_fil/pulsar.par")