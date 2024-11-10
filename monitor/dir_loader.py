import glob
import os
import time
import threading

from .src_loader import src_loader

class dir_loader():
    def __init__(self, psr_dir, app, auto_update = False):
        self.app = app
        self.psr_dir = psr_dir
        self.sources = []
        self.update_checker_thread = None
        self.running = False
        self.auto_update = auto_update

        # Load sources
        self.load_sources()
        print(f"{len(self.sources)} sources loaded")

    def initialize(self):
        self.running = True

        for source in self.sources:
            source.initialize()

        # Start update checker
        if self.auto_update:
            self.update_checker_thread = threading.Thread(target = self.update_checker)
            self.update_checker_thread.start()

    def cleanup(self):
        self.running = False

        for source in self.sources:
            source.cleanup()

        # Stop update checker
        if self.auto_update:
            print("Stopping update checker...")
            self.update_checker_thread.join(3)

    def update_checker(self):
        count = 0
        checking_freq = 1
        while True:
            threading.Event().wait(1)
            count += 1

            if time.time() - self.app.last_request < 30:
                checking_freq = 5
            else:
                checking_freq = 30

            if self.running == False:
                break

            if count >= checking_freq:
                count = 0
                continue

            if time.time() - self.app.last_request < 300:
                print("Checking for updates...")
                for source in self.sources:
                    source.update_checker()

    def load_sources(self):
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