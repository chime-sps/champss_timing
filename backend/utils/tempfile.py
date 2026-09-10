import shutil
import os
from .utils import utils

class TempFile:
    def __init__(self, tempdir="/tmp", create=False):
        self.tempdir = tempdir
        self.path = os.path.join(self.tempdir, utils.get_rand_string())

        if create:
            with open(self.path, 'w') as f:
                pass

    def cleanup(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
            self.path = None

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

class TempDir:
    def __init__(self, tempdir="/tmp"):
        self.tempdir = tempdir
        self.path = os.path.join(self.tempdir, utils.get_rand_string())

        # Create the directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def cleanup(self):
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path)
            self.path = None

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()


# with TempFile() as tmp_path:
#     print(tmp_path)