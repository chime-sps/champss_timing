import glob
import os
import time
import json
import threading
import numpy as np

from .src_loader import src_loader
from ..utils import utils

class dir_loader():
    def __init__(self, psr_dir, app):#, auto_update = False):
        self.app = app
        self.psr_dir = psr_dir
        self.sources = []
        self.update_checker_thread = None
        self.running = False
        # self.auto_update = auto_update
        self.heatmap = {}

    def initialize(self):
        self.running = True

        # Load sources
        self.load_sources()
        print(f"{len(self.sources)} sources loaded")

        # Initialize sources
        for source in self.sources:
            source.initialize()

        # Get heatmap
        self.get_heatmap()

        # # Start update checker
        # if self.auto_update:
        #     self.update_checker_thread = threading.Thread(target = self.update_checker)
        #     self.update_checker_thread.start()

    def get_heatmap(self, n_max=1050, reverse=False):
        heatmap = {}

        for source in self.sources:
            for toa in source.db.get_all_toas():
                if "remark" in toa["notes"]:
                    if "INVALID_TOA" in toa["notes"]["remark"]:
                        continue

                mjd = (np.floor(toa["toa"]))

                if mjd not in heatmap:
                    heatmap[mjd] = 0
                heatmap[mjd] += 1

        # add a value for those days do not have any TOAs
        for i in range(int(np.max(list(heatmap.keys()))) - n_max, int(np.max(list(heatmap.keys())))):
            if i not in heatmap:
                heatmap[i] = 0

        heatmap = dict(sorted(heatmap.items(), key=lambda item: item[0]))

        heatmap_keys = []
        heatmap_val = []

        for key, val in heatmap.items():
            heatmap_keys.append(utils.mjd_to_datetime(key).strftime("%Y-%m-%d"))
            heatmap_val.append(val)

        if reverse:
            heatmap_keys = heatmap_keys[::-1]
            heatmap_val = heatmap_val[::-1]

            if len(heatmap_keys) > n_max:
                heatmap_keys = heatmap_keys[:n_max]
                heatmap_val = heatmap_val[:n_max]
        else:
            if len(heatmap_keys) > n_max:
                heatmap_keys = heatmap_keys[-n_max:]
                heatmap_val = heatmap_val[-n_max:]

        self.heatmap = {
            "key": json.dumps(list(heatmap_keys)),
            "val": json.dumps(list(heatmap_val)),
            "val_normalized": json.dumps(list(heatmap_val / np.max(heatmap_val)))
        }

        return self.heatmap

    def cleanup(self):
        self.running = False

        for source in self.sources:
            source.cleanup()

        # # Stop update checker
        # if self.auto_update:
        #     print("Stopping update checker...")
        #     self.update_checker_thread.join(3)

    # def update_checker(self):
    #     count = 0
    #     checking_freq = 1
    #     while True:
    #         threading.Event().wait(1)
    #         count += 1
    #
    #         if time.time() - self.app.last_request < 30:
    #             checking_freq = 5
    #         else:
    #             checking_freq = 30
    #
    #         if self.running == False:
    #             break
    #
    #         if count >= checking_freq:
    #             count = 0
    #             continue
    #
    #         if time.time() - self.app.last_request < 300:
    #             print("Checking for updates...")
    #             for source in self.sources:
    #                 source.update_checker()
    #             self.get_heatmap()

    def load_sources(self):
        self.sources = []

        for source_dir in glob.glob(self.psr_dir + "/*"):
            db = source_dir + "/champss_timing.sqlite3.db"
            pdf = source_dir + "/champss_diagnostic.pdf"

            # Check if directory
            if not os.path.isdir(source_dir):
                continue

            # Check if db and pdf exists
            if not os.path.exists(db) or not os.path.exists(pdf):
                print(f"Skipping {source_dir} due to missing files")
                continue

            # Add sources to dictionary
            print(f"Adding {source_dir} to sources")
            self.sources.append(src_loader(source_dir))

        # Sort sources by psr_id
        self.sources.sort(key = lambda x: x.psr_id)

    def get_sources(self):
        return self.sources

    # Handle with get item
    def __getitem__(self, key):
        for source in self.sources:
            if source.psr_id == key:
                return source

        raise KeyError(f"Source {key} not found")

    # Handle with loop
    def __iter__(self):
        return iter(self.sources)

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False