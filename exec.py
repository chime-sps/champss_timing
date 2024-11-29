import subprocess
import multiprocessing
import tqdm
import os
import random

from .utils import utils

class exec():
    def __init__(self, n_pools="auto", log=""):
        self.cmds = []
        self.cmds_finished = []
        self.log = log
        self.n_pools = n_pools
        self.res = None

        if self.n_pools == "auto":
            # Try if slurm environment
            try:
                self.n_pools = int(os.environ["SLURM_CPUS_PER_TASK"])
                print(f"SLURM environment detected. ")
            except:
                self.n_pools = multiprocessing.cpu_count()

        print(f"Using {self.n_pools} CPUs.")
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_pools)

    def _exec(self, cmd, log=""):
        # Run the command
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        stdout_formatted = [f"> {cmd}"]
        for line in p.stdout:
                stdout_formatted.append(line.decode("utf-8"))
        
        # Save the output to a log file
        if log != "":
            log = log + f"__{utils.get_rand_string()}"
            with open(log, "w") as f:
                for line in stdout_formatted:
                    f.write(line)
            # print(f"Log file saved to {log}")
        
        # Wait for the command to finish
        p.wait()

        # If finished with an error, raise an exception
        if p.returncode != 0:
            utils.print_error("\n".join(stdout_formatted))
            raise Exception("Command failed with exit code %d: %s (log > \"%s\")" % (p.returncode, cmd, log))
        
        # Return the exit code
        return {"success": (p.returncode == 0), "returncode": p.returncode, "stdout": stdout_formatted}

    def append(self, cmd):
        self.cmds.append(cmd)
    
    def set(self, cmds):
        self.cmds = cmds

    def run(self):
        self.pool_args = []
        
        for cmd in self.cmds:
            self.pool_args.append((cmd, self.log))

        with multiprocessing.Pool(processes=self.n_pools) as pool:
            # self.res = pool.starmap(self._exec, tqdm.tqdm(self.pool_args))
            self.res = pool.starmap(self._exec, tqdm.tqdm(self.pool_args, total=len(self.pool_args)))
            pool.close()
            pool.join()

        return self.res
    
    def check(self):
        if self.res is None:
            raise Exception("No commands have been run yet")
        
        for r in self.res:
            if not r["success"]:
                return False
        
        return True