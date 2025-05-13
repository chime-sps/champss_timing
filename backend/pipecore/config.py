import json
import os

from ..utils.logger import logger
from ..datastores.database import database

class config():
    def __init__(self, path=False, logger=logger(), db_path=None, db_hdl=None):
        self.db_hdl = db_hdl
        self.db_path = db_path
        self.logger = logger
        self.data = {
            "settings": {
                "fit_every_n_days": 1,
                "reset_params": True, 
                "fit_params": ["F0", "F1", "RAJ", "DECJ"], 
                "use_filters": []
            }, 
            "ignore_mjds": {
                "earlier_than": 0,
            }, 
            "metadata": {
                "tag": "untagged"
            }
        }

        # load config file
        if path is not False:
            if os.path.exists(path):
                self.load(path)
            else:
                self.logger.warning(f"Config file '{path}' does not exist. Using default config.")

        # check database connection
        if self.db_hdl is None and self.db_path is None:
            raise Exception("Either db_hdl or db_path must be provided.")
        
        # create database connection
        if self.db_path is not None:
            self.db_hdl = database(self.db_path)
            self.db_hdl.initialize()

        # Sync to database and make sure the config matches the first time the timing was started
        self.sync_to_db()

    def load(self, path):
        with open(path, "r") as file:
            self.data_loaded = json.load(file)
            self.logger.info("Config file loaded.")
            
        for key in self.data_loaded:
            if key not in self.data:
                raise ValueError(f"Unknown key '{key}' in config file.")

            for sub_key in self.data_loaded[key]:
                if sub_key not in self.data[key]:
                    self.logger.warning(f"Unknown sub-key '{sub_key}' in config file.")
                    continue

                self.data[key][sub_key] = self.data_loaded[key][sub_key]
                self.logger.info(f"Set '{key}.{sub_key}' to '{self.data[key][sub_key]}'.")
                
    def to_dict(self):
        return self.data

    def compare_config(self, config1, config2):
        for key in config1:
            if key not in config2:
                return False

            for sub_key in config1[key]:
                if sub_key not in config2[key]:
                    return False

                if config1[key][sub_key] != config2[key][sub_key]:
                    return False

        return True

    def check_no_missing_param(self, db_config, loaded_config):
        for key in loaded_config:
            if key not in db_config:
                return False

            for sub_key in loaded_config[key]:
                if sub_key not in db_config[key]:
                    return False

        return True

    def sync_to_db(self):
        db_config = self.db_hdl.get_all_config()

        if db_config == {} or self.check_no_missing_param(db_config, self.data) == False:
            self.logger.success("No config found in database or config file has more parameters than database. Inserting config to database.")
            self.db_hdl.insert_config(self.data)
        else:
            if not self.compare_config(self.data, db_config):
                self.logger.error("Config file and database config do not match. This may be due to a change in the config file after the timing was first started. Please resolve this issue manually.")
                self.logger.error("Local config:")
                self.logger.error(self.data, layer=1)
                self.logger.error("Database config:")
                self.logger.error(db_config, layer=1)
                raise Exception("Config file and database config do not match. ")
        