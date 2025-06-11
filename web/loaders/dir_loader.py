import glob
import os
import json
import time
import datetime
import threading
import numpy as np

from .src_loader import src_loader
from backend.utils.utils import utils

class dir_loader():
    def __init__(self, psr_dir, app, query_simbad=True):#, auto_update = False):
        self.app = app
        self.psr_dir = psr_dir
        self.query_simbad = query_simbad
        self.sources = []
        self.update_checker_thread = None
        self.running = False
        # self.auto_update = auto_update
        self.heatmap = {}
        self.plots = {}
        self.tags = []
        self.last_updated = 0

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

        # Get plots
        self.get_plots()

        # Get tags
        self.get_tags()

        # Set last_updated timestamp
        self.last_updated = time.time()

        # # Start update checker
        # if self.auto_update:
        #     self.update_checker_thread = threading.Thread(target = self.update_checker)
        #     self.update_checker_thread.start()

        # preload the on_diagnostic_request events
        threading.Thread(target=self.__preload_on_diagnostic_request).start()
        
    def __preload_on_diagnostic_request(self):
        """
        Preload the on_diagnostic_request events for all sources.
        This is to ensure that the event handlers are ready when the app starts.
        """
        for source in self.sources:
            source.on_diagnostic_request()

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
            heatmap_keys.append(utils.mjd_to_datetime(key, utc=False).strftime("%Y-%m-%d"))
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
    
    def get_plots(self):
        self.plots["tags_avail"] = self.get_tags()
        self.plots["skymap"] = {}
        self.plots["ppdot"] = {}
        self.plots["pdm"] = {}
        self.plots["ntoachi2r"] = {}

        for this_tag in self.plots["tags_avail"]:
            self.plots["skymap"][this_tag] = {"x": [], "y": [], "links": [], "psr_id": []}
            self.plots["ppdot"][this_tag] = {"x": [], "y": [], "links": [], "psr_id": []}
            self.plots["pdm"][this_tag] = {"x": [], "y": [], "links": [], "psr_id": []}
            self.plots["ntoachi2r"][this_tag] = {"x": [], "y": [], "links": [], "psr_id": []}

        for source in self.sources:
            if source.last_timing_info["fitted_params"]["CHI2R"] > 10 or max(source.last_timing_info["notes"]["fitted_mjds"]) - min(source.last_timing_info["notes"]["fitted_mjds"]) < 180:
                continue

            this_tag = source.config["metadata"]["tag"]

            # skymap
            self.plots["skymap"][this_tag]["x"].append(source.last_timing_info["fitted_params"]["RAJ"])
            self.plots["skymap"][this_tag]["y"].append(source.last_timing_info["fitted_params"]["DECJ"])
            self.plots["skymap"][this_tag]["links"].append(f"/diagnostics/{source.psr_id}")
            self.plots["skymap"][this_tag]["psr_id"].append(source.psr_id)
    
            # p-pdot
            self.plots["ppdot"][this_tag]["x"].append(utils.f02p0(source.last_timing_info["fitted_params"]["F0"]))
            self.plots["ppdot"][this_tag]["y"].append(utils.f12p1(source.last_timing_info["fitted_params"]["F0"], source.last_timing_info["fitted_params"]["F1"]))
            self.plots["ppdot"][this_tag]["links"].append(f"/diagnostics/{source.psr_id}")
            self.plots["ppdot"][this_tag]["psr_id"].append(source.psr_id)

            # p-dm
            self.plots["pdm"][this_tag]["x"].append(utils.f02p0(source.last_timing_info["fitted_params"]["F0"]))
            self.plots["pdm"][this_tag]["y"].append(source.last_timing_info["fitted_params"]["DM"])
            self.plots["pdm"][this_tag]["links"].append(f"/diagnostics/{source.psr_id}")
            self.plots["pdm"][this_tag]["psr_id"].append(source.psr_id)

            # p-chi2r
            self.plots["ntoachi2r"][this_tag]["x"].append(source.last_timing_info["fitted_params"]["NTOA"])
            self.plots["ntoachi2r"][this_tag]["y"].append(source.last_timing_info["fitted_params"]["CHI2R"])
            self.plots["ntoachi2r"][this_tag]["links"].append(f"/diagnostics/{source.psr_id}")
            self.plots["ntoachi2r"][this_tag]["psr_id"].append(source.psr_id)

    def get_tags(self):
        self.tags = []

        for source in self.sources:
            this_tag = source.config["metadata"]["tag"]
            if this_tag not in self.tags:
                self.tags.append(this_tag)

        return self.tags

    def cleanup(self):
        # Skip cleanup if not running
        if not self.running:
            return
        
        # Set running to False to stop any ongoing processes
        self.running = False

        # Cleanup sources
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
        """
        Load sources from the psr_dir directory.
        """

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
            self.sources.append(src_loader(source_dir, query_simbad=self.query_simbad))

        # Sort sources by psr_id
        self.sources.sort(key = lambda x: x.psr_id)

    def get_sources(self):
        """
        Get the list of sources.
        """

        return self.sources
    
    def get_last_updated(self, format=True):
        """
        Get the last update time of the sources.
        
        Parameters
        ----------
        format : bool
            If True, return the last update time as a formatted string.
            If False, return the last update time as a timestamp.
        """

        if format:
            if time.time() - self.last_updated < 3600:
                return "Updated just now"
            if time.time() - self.last_updated < 3 * 3600:
                return "Updated hours ago"
            if time.time() - self.last_updated < 24 * 3600:
                return "Updated today"
            if time.time() - self.last_updated < 2 * 24 * 3600:
                return "Updated yesterday"
            return "Updated on " + datetime.datetime.fromtimestamp(self.last_updated).strftime("%Y-%m-%d %H:%M:%S")

        return self.last_updated
    
    def is_updated_recently(self, threshold=86400):
        """
        Check if the sources are updated recently.
        
        Parameters
        ----------
        threshold : int
            The threshold in seconds to consider the sources as updated recently.
            Default is 24 hours (86400 seconds).
        """

        return time.time() - self.last_updated < threshold

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