import os
import json

class CLIConfig:
    def __init__(self, config_file='.champss_timing.config', load_error=True):
        # Initialize default config
        self.config = {
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
            "toa_jumps": {
                "chimepsr_fil": [0.25165824, 0.00039286]
            }, 
            "data_paths": {
                "champss": "PATH_TO_CHAMPSS_DATA",
                "chimepsr_fm": "PATH_TO_CHIMEPSR_FODEMODE_DATA",
                "chimepsr_fil": "PATH_TO_CHIMEPSR_FILTERBANK_DATA",
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
        try:
            with open(self.config_file, 'r') as f:
                self.config = self.recursively_update_dict(self.config, json.load(f))
        except Exception as e:
            print(f"Error loading config file: {e}. Using default config.")
            if load_error:
                raise e

    def save(self):
        # Save the current config to the file
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def recursively_update_dict(self, dest, source):
        for k, v in dest.items():
            if k not in source:
                print(f"Key '{k}' not found in the config. Using default value.")
                continue
            
            # Merge user-defined keys
            if k == "user_defined":
                dest[k] = source[k]
                continue

            if isinstance(v, dict):
                dest[k] = self.recursively_update_dict(v, source[k])
            else:
                dest[k] = source[k]

        return dest