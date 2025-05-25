import os
import json

class CLIConfig:
    def __init__(self, config_file='.champss_timing.config', load_error=True):
        # Initialize default config
        self.config = {
            "version": 2,  # Version of the config
            "slack_token": {
                "chime": {
                    "CHANNEL_ID": "CHANNEL_ID_HERE", 
                    "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN_HERE",
                    "SLACK_APP_TOKEN": "SLACK_APP_TOKEN_HERE"
                }, 
                "test": {
                    "CHANNEL_ID": "CHANNEL_ID_HERE", 
                    "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN_HERE",
                    "SLACK_APP_TOKEN": "SLACK_APP_TOKEN_HERE"
                }
            }, 
            "backends": {
                "champss": {
                    "label": "CHAMPSS Fold Mode", 
                    "data_path": "/PATH/TO/CHAMPSS/DATA",
                    "jump": [0, 0]
                }, 
                "chimepsr": {
                    "label": "CHIME/Pulsar Fold Mode",
                    "data_path": "/PATH/TO/CHIMEPSR/DATA",
                    "jump": [0, 0]
                }
            }, 
            "user_defined": {}
        }
        self.config_file = config_file

        # Load config
        self.load_config(load_error=load_error)

    def get_config(self):
        return self.config

    def load_config(self, load_error=True):
        # If not exists, create it
        if not os.path.exists(self.config_file):
            self.save()
            return

        # Load existing config
        with open(self.config_file, 'r') as f:
            loaded_config = json.load(f)

        # Merge with default config
        if "slack_token" in loaded_config:
            self.config["slack_token"] = loaded_config["slack_token"]
        if "backends" in loaded_config:
            self.config["backends"] = loaded_config["backends"]
        if "user_defined" in loaded_config:
            self.config["user_defined"] = loaded_config["user_defined"]

        # Convert the config to v1 for compatibility -- there are still bunch of codes using older config :(
        toa_jumps = {}
        data_paths = {}
        for backend, data in self.config["backends"].items():
            if "jump" in data:
                toa_jumps[backend] = data["jump"]
            if "data_path" in data:
                data_paths[backend] = data["data_path"]
        self.config["toa_jumps"] = toa_jumps
        self.config["data_paths"] = data_paths
        self.config["version"] = 2

    def save(self):
        # Save the current config to the file
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)