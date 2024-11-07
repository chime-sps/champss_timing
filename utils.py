import random
import datetime
import traceback
from hashlib import md5
import os

class utils:
    def print_warning(string):
        print(f"\033[93m{string}\033[0m")

    def print_error(string):
        print(f"\033[91m{string}\033[0m")

    def print_success(string):
        print(f"\033[92m{string}\033[0m")

    def print_info(string):
        print(f"\033[94m{string}\033[0m")

    def get_time_string():
        return str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
    
    def get_rand_string():
        return md5(str(random.random()).encode()).hexdigest()[0:6]
    
    def get_archive_id(archive):
        return archive.split("/")[-1].split(".")[0]

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

    def get_version_hash():
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode("utf-8")
        except:
            return "unknown"