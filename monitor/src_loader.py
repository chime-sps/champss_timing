import datetime
import base64
import numpy as np
from champss_timing.database import database

class src_loader():
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.db = database(source_dir + "/champss_timing.sqlite3.db", readonly = True)
        self.pdf = source_dir + "/champss_diagnostic.pdf"
        self.psr_id = source_dir.split("/")[-1]

        # Load database
        self.db.initialize()

    def get_resids(self):
        timing_info = self.db.get_last_timing_info()

        mjds = timing_info["notes"]["fitted_mjds"]
        resids = timing_info["residuals"]

        return {"mjd": mjds, "val": resids["val"], "err": resids["err"]}
    
    def get_parameter_info(self):
        all_timing_info = self.db.get_all_timing_info()
        mjd = []
        parameter_info = {}
        # parameter_info = {"F0": [], "F1": [], "RAJ": [], "DECJ": [], "PX": [], "PMRA": [], "PMDEC": []}

        for param in all_timing_info[-1]["unfreeze_params"]:
            parameter_info[param] = []

        for timing_info in all_timing_info:
            mjd.append(max(timing_info["obs_mjds"]))
            for key in parameter_info.keys():
                parameter_info[key].append(timing_info["fitted_params"][key])
        
        for key in parameter_info.keys():
            parameter_info[key] = {"val": parameter_info[key], "mjd": mjd}

        return parameter_info
    
    def get_statistics(self):
        all_timing_info = self.db.get_all_timing_info()
        mjd = []
        statistics = {"chi2": [], "chi2r": [], "rms": []}

        for timing_info in all_timing_info:
            mjd.append(max(timing_info["obs_mjds"]))
            statistics["chi2"].append(timing_info["chi2"])
            statistics["chi2r"].append(timing_info["chi2_reduced"])
            statistics["rms"].append(np.sqrt(np.mean(np.square(timing_info["residuals"]["val"]))))

        for key in statistics.keys():
            statistics[key] = {"val": statistics[key], "mjd": mjd}

        return statistics
    
    def get_parfile(self):
        return self.db.get_last_timing_info()["notes"]["fitted_parfile"]
    
    def get_last_updated(self):
        # return in YYYY-MM-DD 
        return datetime.datetime.utcfromtimestamp(self.db.get_last_timing_info()["timestamp"]).strftime('%Y-%m-%d')
    
    def get_diagnostic_pdf(self):
        with open(self.pdf, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    