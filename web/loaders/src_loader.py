import os
import datetime
import requests
import json
import base64
import numpy as np
import hashlib
from scipy.spatial import KDTree
from backend.pipecore.checker import checker
from backend.datastores.database import database

from backend.utils.utils import utils


class src_loader():
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.db = None
        self.db_md5 = None
        self.config = None
        self.checker_warnings = {}
        self.checker_warnings_length = 0
        self.pdf = source_dir + "/champss_diagnostic.pdf"
        self.logfile = source_dir + "/champss_timing.log"
        self.dealias_logfile = source_dir + "/dealias/champss_timing.log"
        self.psr_id = source_dir.split("/")[-1]
        self.psr_id_esc = self.psr_id.replace("+", "p").replace("-", "m")

        self.last_timing_info = {}
        self.stats = {}
        self.parameter_info = {}
        self.source_coincidences = []
        self.source_coincidences_radius = 0.25 # deg
        self.source_coincidences_catalogs = []
        self.source_coincidences_map_default = "SIMBAD Query"

    def connect_db(self):
        self.db = database(self.source_dir + "/champss_timing.sqlite3.db", readonly=True)
        self.db.initialize()

    def initialize(self):
        # Get md5
        self.db_md5 = self.get_db_md5()

        # Load database
        self.connect_db()

        # Get config
        self.config = self.db.get_all_config()

        # Get last_timing_info
        self.last_timing_info = self.db.get_last_timing_info()

        # Get statistics
        self.stats = self.get_statistics(from_db=True)

        # Get parameter info
        self.parameter_info = self.get_parameter_info(ra_in_deg=True)

        # Get source coincidences
        self.source_coincidences, self.source_coincidences_map_default = self.source_coincidence_query(
            self.last_timing_info["fitted_params"]["RAJ"] / 24 * 360,
            self.last_timing_info["fitted_params"]["DECJ"],
            # radius=0.0166667
            radius=self.source_coincidences_radius, 
            simbad=False
        )

        # Get checker warnings
        self.checker_warnings = self.get_checker_warnings()
        self.checker_warnings_length = len(self.checker_warnings)

    def cleanup(self):
        # Close database
        self.db.close()

    def get_db_md5(self):
        with open(self.source_dir + "/champss_timing.sqlite3.db", "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_resids(self):
        timing_info = self.last_timing_info

        mjds = timing_info["notes"]["fitted_mjds"]
        resids = timing_info["residuals"]

        period = 1 / timing_info["fitted_params"]["F0"]
        resids_val =[float(t) for t in list(np.array(resids["val"]) / period * 1e-6)]
        resids_err = [float(t) for t in list(np.array(resids["err"]) / period * 1e-6)]

        return {"mjd": mjds, "val": resids_val, "err": resids_err, "updated": utils.mjd_to_datetime(np.max(mjds), utc=False).strftime("%Y-%m-%d")}

    def get_parameter_info(self, ra_in_deg=False):
        all_timing_info = self.db.get_all_timing_info()
        mjd = []
        parameter_info = {}
        # parameter_info = {"F0": [], "F1": [], "RAJ": [], "DECJ": [], "PX": [], "PMRA": [], "PMDEC": []}

        for param in all_timing_info[-1]["unfreeze_params"]:
            parameter_info[param] = []

        for timing_info in all_timing_info:
            mjd.append(max(timing_info["obs_mjds"]))
            for key in parameter_info.keys():
                if key in timing_info["fitted_params"]:
                    parameter_info[key].append(timing_info["fitted_params"][key])
                else:
                    parameter_info[key].append("nan")
        
        if ra_in_deg:
            if "RAJ" in parameter_info:
                for i in range(len(parameter_info["RAJ"])):
                    parameter_info["RAJ"][i] = parameter_info["RAJ"][i] / 24 * 360
            if "PMRA" in parameter_info:
                for i in range(len(parameter_info["PMRA"])):
                    parameter_info["PMRA"][i] = parameter_info["PMRA"][i] / 24 * 360

        for key in parameter_info.keys():
            parameter_info[key] = {"val": parameter_info[key], "mjd": mjd}

        return parameter_info

    def get_statistics(self, from_db=False):
        if not from_db:
            if self.stats != {}:
                return self.stats

        all_timing_info = self.db.get_all_timing_info()
        period = 1 / self.last_timing_info["fitted_params"]["F0"]
        mjd = []
        statistics = {"chi2": [], "chi2r": [], "rms": []}

        for timing_info in all_timing_info:
            mjd.append(max(timing_info["obs_mjds"]))
            statistics["chi2"].append(timing_info["chi2"])
            statistics["chi2r"].append(timing_info["chi2_reduced"])
            statistics["rms"].append(
                float(np.sqrt(np.mean(np.square(timing_info["residuals"]["val"])))) / period * 1e-6
            )

        for key in statistics.keys():
            for i in range(len(statistics[key])):
                if statistics[key][i] == np.inf or statistics[key][i] == np.nan or statistics[key][i] is None:
                    statistics[key][i] = 0

        for key in statistics.keys():
            statistics[key] = {"val": statistics[key], "mjd": mjd}

        return statistics

    def get_parfile(self, derived_params=False):
        parfile = self.last_timing_info["notes"]["fitted_parfile"]
        # parfile = "\n".join([line for line in parfile.split("\n") if not line.startswith("#")])

        if derived_params:
            if len(self.get_summary().split("Derived Parameters")) > 1:
                derived_params = "Derived Parameters" + self.get_summary().split("Derived Parameters")[1]
                derived_params = "\n".join(["# " + line for line in derived_params.split("\n") if line.strip() != ""])
                parfile += "\n" + derived_params

        return parfile
    
    def get_timfile(self):
        return self.db.create_timfile()
    
    def get_ephms(self):
        return self.last_timing_info["fitted_params"]
    
    def get_summary(self):
        return self.last_timing_info["notes"]["fitted_summary"]
        
    def get_last_updated(self):
        # return in YYYY-MM-DD
        # return datetime.datetime.utcfromtimestamp(self.last_timing_info["timestamp"]).strftime('%Y-%m-%d')
        return utils.mjd_to_datetime(np.max(self.last_timing_info["notes"]["fitted_mjds"]), utc=False).strftime("%Y-%m-%d")

    def get_diagnostic_pdf_base64(self):
        with open(self.pdf, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
    def get_config(self):
        return self.db.get_all_config()
    
    def get_checker_warnings(self):
        warnings = checker(psr_dir=self.source_dir, db_hdl=self.db).check()

        warnings_formatted = []
        for checker_module in warnings.keys():
            for key in warnings[checker_module].keys():
                if warnings[checker_module][key]["level"] > 0:
                    warnings_formatted.append(warnings[checker_module][key])
                
        return warnings_formatted
    
    def get_processing_log(self, to_json=True):
        if not os.path.exists(self.logfile):
            if to_json:
                return '["No log file available."]'
            return "No log file available."
        
        with open(self.logfile, "r") as f:
            if to_json:
                return json.dumps(f.readlines())
            return f.read()
        
    def get_dealias_log(self, to_json=True):
        if not os.path.exists(self.dealias_logfile):
            if to_json:
                return '["No log file available."]'
            return "No log file available."
        
        with open(self.dealias_logfile, "r") as f:
            if to_json:
                return json.dumps(f.readlines())
            return f.read()
        
    def get_profile_mjds(self):
        mjds = {}
        obs_mjds = self.last_timing_info["obs_mjds"]
        obs_filenames = self.last_timing_info["files"]

        for i, mjd in enumerate(self.last_timing_info["obs_mjds"]):
            if utils.mjd_to_datetime(mjd, utc=False).strftime("%Y.%m") not in mjds:
                mjds[utils.mjd_to_datetime(mjd, utc=False).strftime("%Y.%m")] = []

            mjds[utils.mjd_to_datetime(mjd, utc=False).strftime("%Y.%m")].append({
                "mjd": mjd,
                "day": utils.mjd_to_datetime(mjd, utc=False).strftime("%d"),
                "filename": obs_filenames[i]
            })

            # sort by mjd 
            mjds[utils.mjd_to_datetime(mjd, utc=False).strftime("%Y.%m")] = sorted(
                mjds[utils.mjd_to_datetime(mjd, utc=False).strftime("%Y.%m")], 
                key=lambda x: x["mjd"], 
                reverse=False
            )

        # sort by month lastest first
        mjds = {k: mjds[k] for k in sorted(mjds.keys(), reverse=True)}

        return mjds
    
    def get_profile_data(self, filename):
        return self.db.get_archive_info_by_filename(filename)

    def update_checker(self):
        this_db_md5 = self.get_db_md5()
        if this_db_md5 != self.db_md5:
            self.db.close()
            self.db_md5 = this_db_md5
            self.connect_db()
            print(f"Database {self.psr_id} updated")

    def get_source_position_error(self):
        raj_err = 0.5
        decj_err = 0.5 

        parfile = self.get_parfile()
        for line in parfile.split("\n"):
            if line.startswith("RAJ"):
                if len(line.split()) == 4:
                    raj_err = float(line.split()[3]) / 3600
            if line.startswith("DECJ"):
                if len(line.split()) == 4:
                    decj_err = float(line.split()[3]) / 3600

        return {"RAJ": raj_err, "DECJ": decj_err}

    def simbad_query(self, ra, dec, radius=0.5):
        def parse_coord(coord):
            # split coord
            coord = coord.split(" ")
            ra_h = coord[0]
            ra_m = coord[1]
            ra_s = coord[2]
            dec_d = coord[3]
            dec_m = coord[4]
            dec_s = coord[5]

            # convert to deg
            ra = (float(ra_h) + float(ra_m)/60 + float(ra_s)/3600) * 15
            dec = float(dec_d) + float(dec_m)/60 + float(dec_s)/3600

            return ra, dec

        api = "https://simbad.cds.unistra.fr/simbad/sim-coo?output.format=ASCII&Coord=%RA%+%DEC%&Radius=%RADIUS%&Radius.unit=deg"
        api = api.replace("%RA%", str(ra)).replace("%DEC%", str(dec)).replace("%RADIUS%", str(radius))
        table_raw = requests.get(api).text
        
        table = {}
        for line in table_raw.split("\n"):
            if " # |" in line:
                for key in line.split("|"):
                    table[key.strip()] = []
            elif "|" in line and "----" not in line and table != {}:
                for i, value in enumerate(line.split("|")):
                    table[list(table.keys())[i]].append(value.strip())
        

        if table == {}:
            return []
        
        table_formatted = []
        for i in range(len(table[list(table.keys())[0]])):
            row = {}
            for key in table.keys():
                if "coord" in key.lower():
                    row["coord"] = table[key][i]
                if "ident" in key.lower():
                    row["ident"] = table[key][i]
                if "dist" in key.lower():
                    row["dist"] = table[key][i]
                if "typ" in key.lower():
                    row["type"] = table[key][i]
                
            name = row["ident"]
            ra, dec = parse_coord(row["coord"])
            radius = 0.01
            notes = f"Type: {row['type']}"
            dist = float(row["dist"]) / 3600 # convert to deg

            table_formatted.append({
                'name': name,
                'ra': ra,
                'dec': dec,
                'radius': radius,
                'notes': notes,
                'catalog': 'SIMBAD Query', 
                "distance": dist
            })

            return table_formatted

    def source_coincidence_query(self, ra, dec, radius=0.5, data=None, simbad=True):
        def distance(ra1, dec1, ra2, dec2):
            ra1 = np.radians(ra1)
            dec1 = np.radians(dec1)
            ra2 = np.radians(ra2)
            dec2 = np.radians(dec2)

            return float(np.degrees(np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))))
        
        sources = []
        self.source_coincidences_catalogs = []

        if data is None:
            data = json.loads(
                open(os.path.dirname(__file__) + "/../data/source_coincidence_data.json", "r").read()
            )
        self.source_coincidences_catalogs = data["catalogs"]

        # initialize kd-tree
        tree = KDTree(data["kdt_index"])

        # query all
        idx = tree.query_ball_point([ra, dec], radius)

        print(f"Found {len(idx)} sources within {radius} deg of ({ra}, {dec})")
        for i in idx:
            sources.append(data["sources"][i])
            sources[-1]["distance"] = distance(ra, dec, sources[-1]["ra"], sources[-1]["dec"])

        # simbad query
        if simbad:
            sources += self.simbad_query(ra, dec, radius)
            self.source_coincidences_catalogs.append("SIMBAD Query")
            print(f"Found {len(sources)} sources in total with SIMBAD Query")

        # sort by distance
        sources = sorted(sources, key=lambda x: x["distance"])

        # # get most frequent catalog name
        # source_counts = {}
        # most_frequent_catalog = "SIMBAD Query"
        # for source in sources:
        #     if source["catalog"] not in source_counts:
        #         source_counts[source["catalog"]] = 0
        #     source_counts[source["catalog"]] += 1
            
        # if source_counts != {}:
        #     most_frequent_catalog = max(source_counts, key=source_counts.get)
        # print(f"Most frequent catalog: {most_frequent_catalog}")

        # return sources, most_frequent_catalog

        # get nearest source
        nearest_source_catalog = "SIMBAD Query"
        if len(sources) > 0:
            nearest_source_catalog = sources[0]["catalog"]

        return sources, nearest_source_catalog
