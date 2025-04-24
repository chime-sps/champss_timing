import time
import random
import datetime
import traceback
import subprocess
from hashlib import md5
import os

class utils:
    @staticmethod
    def print_warning(string):
        print(f"\033[93m{string}\033[0m")

    @staticmethod
    def print_error(string):
        print(f"\033[91m{string}\033[0m")

    @staticmethod
    def print_success(string):
        print(f"\033[92m{string}\033[0m")

    @staticmethod
    def print_info(string):
        print(f"\033[94m{string}\033[0m")

    @staticmethod
    def get_time_string():
        return str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
    
    @staticmethod
    def get_rand_string():
        return md5(str(random.random()).encode()).hexdigest()[0:6]

    @staticmethod
    def get_md5sum(filename):
        return md5(open(filename, 'rb').read()).hexdigest()
    
    @staticmethod
    def get_archive_id(archive):
        arid = ""
        arname_splitted = archive.split('/')[-1].split('.')

        for i in range(len(arname_splitted)):
            if i == 0:
                arid += arname_splitted[i]
            else:
                try:
                    float(arname_splitted[i-1][-1] + "." + arname_splitted[i][0])
                    arid += "." + arname_splitted[i]
                except:
                    break
            
        return arid
    
    @staticmethod
    def no_extension(filename):
        filename_ = ".".join(filename.split('.')[:-1])

        if filename_ == "":
            return filename

        return filename_

    @staticmethod
    def no_overwriting_name(name):
        name_ = name
        i = 0

        while os.path.exists(name_):
            i += 1
            if "." in name:
                name_ = f"{'.'.join(name.split('.')[:-1])}_{i}.{name.split('.')[-1]}"
            else:
                name_ = f"{name}_{i}"

        return name_

    @staticmethod
    def get_version_hash():
        try:
            return "v2." + subprocess.check_output(['git', '-C', os.path.dirname(os.path.realpath(__file__)), 'rev-parse', '--short', 'HEAD']).decode().strip()
        except:
            return "v2." + "unknown"

    @staticmethod
    def mjd_to_timestamp(mjd):
        return (mjd - 40587) * 86400

    @staticmethod
    def mjd_to_datetime(mjd, utc=True):
        if utc:
            return datetime.datetime.utcfromtimestamp(utils.mjd_to_timestamp(mjd))
        else:
            return datetime.datetime.fromtimestamp(utils.mjd_to_timestamp(mjd))

    @staticmethod
    def timestamp_to_mjd(timestamp):
        return timestamp / 86400 + 40587

    @staticmethod
    def mjd_now():
        return utils.timestamp_to_mjd(time.time())

    @staticmethod
    def read_f0_from_parfile(parfile, raise_exception=True):
        with open(parfile, "r") as f:
            for line in f:
                if "F0" == line.split()[0].strip():
                    return float(line.split()[1])
        
        if raise_exception:
            raise Exception("Failed to read F0 from parfile")

        return None

    @staticmethod
    def read_dm_from_parfile(parfile, raise_exception=True):
        with open(parfile, "r") as f:
            for line in f:
                if "DM" == line.split()[0].strip():
                    return float(line.split()[1])

        if raise_exception:
            raise Exception("Failed to read DM from parfile")

        return None
    
    @staticmethod
    def f02p0(f0):
        return 1.0 / f0
    
    @staticmethod
    def f12p1(f0, f1):
        return - f1 / (f0 ** 2)
    
    @staticmethod
    def deg2dms(deg):
        sign = -1 if deg < 0 else 1
        deg = abs(deg)

        d = int(deg)
        m = int((deg - d) * 60)
        s = (deg - d - m / 60) * 3600

        d *= sign

        if format:
            return f"{d:02d}:{abs(m):02d}:{abs(s):05.3f}"
        else:
            return (d, m, s)

    @staticmethod
    def get_raw_data_format(filename, raise_exception=True):
        from champss_timing.io.archive import ArchiveReader
        from champss_timing.io.filterbank import FilterbankReader

        try:
            ArchiveReader(filename)
            return "archive"
        except:
            pass

        try:
            FilterbankReader(filename)
            return "filterbank"
        except:
            pass

        if raise_exception:
            raise Exception("Failed to determine raw data format")

        return None