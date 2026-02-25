import subprocess
from backend.datastores.database import database

class PintkUtils:
    def __init__(self, psrdir, sourcedb):
        self.psrdir = psrdir
        self.sourcedb = sourcedb

    def create_parfile(self, path, initial=False):
        if initial:
            with open(f"{self.psrdir}/parfile_bak/initial_parfile.bak", 'r') as f:
                return f.read()

        return self.sourcedb.get_all_timing_info()[-1]["notes"]["fitted_parfile"]

    def create_timfile(self, path, mjd_range=[0, 1e32], add_rcvr_flag=True):
        timfiles = ""

        if not isinstance(mjd_range, list) or len(mjd_range) != 2:
            raise ValueError("mjd_range must be a list of two values [start_mjd, end_mjd].")

        for toa in self.sourcedb.get_all_toas():
            # Skip TOAs with empty raw_tim field
            if toa['raw_tim'].strip() == "":
                continue

            # Check if the TOA is within the specified MJD range
            if toa['toa'] < mjd_range[0] or toa['toa'] > mjd_range[1]:
                continue

            # Add the raw_tim value to the timfiles string
            timfiles += f" {toa['raw_tim']}"

            # Add receiver flag if requested and available
            if add_rcvr_flag:
                if 'rcvr' in toa['notes']:
                    timfiles += f" -rcvr {toa['notes']['rcvr']}"
                else:
                    print(f"Warning: Receiver information not found for TOA with MJD {toa['toa']}.")

            timfiles += f" \n"

        return timfiles.strip()

    def write_parfile(self, path, use_initial=False):
        parfile_content = self.create_parfile(path, initial=use_initial)
        with open(path, 'w') as parfile:
            parfile.write(parfile_content)

    def write_timfile(self, path, mjd_range=[0, 1e32], add_rcvr_flag=True):
        timfile_content = self.create_timfile(path, mjd_range, add_rcvr_flag)
        with open(path, 'w') as timfile:
            timfile.write(timfile_content)

    def run_pintk(self, parfile_path, timfile_path):
        try:
            subprocess.run(["pintk", parfile_path, timfile_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error starting pintk: {e}")