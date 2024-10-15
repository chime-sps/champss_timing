import random
import datetime
from hashlib import md5

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