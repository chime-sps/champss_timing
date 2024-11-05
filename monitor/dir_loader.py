import glob
import os

from .src_loader import src_loader

class dir_loader():
    def __init__(self, psr_dir):
        self.psr_dir = psr_dir
        self.sources = []

        # Load sources
        self.load_sources()
        print(f"{len(self.sources)} sources loaded")

    def load_sources(self):
        for source_dir in glob.glob(self.psr_dir + "/*"):
            db = source_dir + "/champss_timing.sqlite3.db"
            pdf = source_dir + "/champss_diagnostic.pdf"
            
            # Check if directory
            if not os.path.isdir(source_dir):
                continue

            # Check if db and pdf exists
            if not os.path.exists(db) or not os.path.exists(pdf):
                continue
                
            # Add sources to dictionary
            self.sources.append(src_loader(source_dir))

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