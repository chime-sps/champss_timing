import time
import random
import datetime
import traceback
import subprocess
import os
from hashlib import md5
from astropy.coordinates import SkyCoord
from astropy.time import Time

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
    def create_parfile(ra, dec, f0, f1, dm, psrname=None, pepoch=None):
        # Parse coordinate
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        ra_str = coord.ra.to_string(unit="hour", sep=":", pad=True)
        dec_str = coord.dec.to_string(unit="deg", sep=":", alwayssign=True, pad=True)

        # Parse psrname
        if psrname is None:
            psrname = "J" + "".join(ra_str.split(":")[0:2]) + "".join(dec_str.split(":")[0:2])

        # Parse pepoch
        if pepoch is None:
            pepoch = Time.now().mjd

        # Create the parfile content as a string
        parfile = ""
        parfile += f"PSRJ {psrname}\n"
        parfile += f"RAJ {ra_str}\n"
        parfile += f"DECJ {dec_str}\n"
        parfile += f"DM {dm}\n"
        parfile += f"F0 {f0}\n"
        parfile += f"F1 {f1}\n"
        parfile += f"PEPOCH {pepoch}\n"
        parfile += f"DMEPOCH {pepoch}\n"
        parfile += f"EPHVER 2\n"
        parfile += f"UNITS TDB\n"

        return parfile

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
    def read_start_end_from_parfile(parfile, raise_exception=True):
        mjd_start = None
        mjd_end = None
        with open(parfile, "r") as f:
            for line in f:
                if "START" in line:
                    mjd_start = float(line.split()[1])
                if "FINISH" in line:
                    mjd_end = float(line.split()[1])

        if mjd_start is None or mjd_end is None:
            if raise_exception:
                raise Exception("Failed to read START/FINISH from parfile")

        if mjd_start is None:
            mjd_start = 0

        if mjd_end is None:
            mjd_end = 999999

        return mjd_start, mjd_end
    
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
    def read_file_header(filename, num_bytes=1024, parse_string=False):
        with open(filename, "rb") as f:
            header = f.read(num_bytes)
            if parse_string:
                parsed = "".join(
                    chr(byte) if 32 <= byte < 127 else "."
                    for byte in header
                )
                return parsed
            return header

    @staticmethod
    def get_raw_data_format(filename, raise_exception=True, ignore_extension=False):
        # Make sure the file exists
        if not os.path.isfile(filename):
            if raise_exception:
                raise Exception("File does not exist")
            return None

        # Try to distinguish the format base of the extensions first since it is always faster than trying to read the file
        if not ignore_extension:
            ext = filename.split('.')[-1].lower()
            if ext in ["fil"]:
                return "filterbank"
            elif ext in ["ar", "clfd"]:
                return "archive"

        # Try to identify from the header
        parsed_header = utils.read_file_header(filename, num_bytes=1024, parse_string=True)
        if "HEADER_START" in parsed_header or "HEADER_END" in parsed_header:
            return "filterbank"
        elif "TimerArchive" in parsed_header or "ChebyModel" in parsed_header:
            return "archive"

        # If the above methods fail, try to actually read the file with both readers and see which one succeeds
        try:
            from ..io.filterbank import FilterbankReader
            FilterbankReader(filename)
            return "filterbank"
        except:
            pass

        try:
            from ..io.archive import ArchiveReader
            ArchiveReader(filename)
            return "archive"
        except:
            pass

        if raise_exception:
            raise Exception("Failed to determine raw data format")

        return None

    @staticmethod
    def is_pulsar_name(name):
        if name.startswith("J") or name.startswith("B"):
            if len(name) >= 8:
                if name[5] == "+" or name[5] == "-":
                    return True
        
        return False