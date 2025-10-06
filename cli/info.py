import time
import datetime

class PSRInfo:
    def __init__(self, psr, masterdb, sourcedb):
        self.psr = psr
        self.masterdb = masterdb
        self.sourcedb = sourcedb

    def get_info(self):
        return {
            "masterdb": self.get_masterdb_info(),
            "sourcedb": self.get_sourcedb_info()
        }

    def get_masterdb_info(self):
        # Raw data info
        raw_data = self.masterdb.get_raw_data(self.psr)
        raw_data_types = {}
        for status in self.masterdb.allowed_status:
            raw_data_types["n_" + status] = len(
                [d for d in raw_data if d["status"] == status]
            )
        for status in self.masterdb.allowed_formats:
            raw_data_types["n_" + status] = len(
                [d for d in raw_data if d["format"] == status]
            )
        raw_data_types["n_total"] = len(raw_data)

        # Timing summary
        timing_info = self.masterdb.get_timing(psr_id=self.psr)[0]
        timing_summary = {
            "timing_dir": timing_info["timing_dir"],
            "last_status": timing_info["last_status"],
            "last_updated": self.format_timestamp(timing_info["last_updated"]),
        }

        return {
            "raw_data_summary": raw_data_types,
            "timing_summary": timing_summary
        }

    def get_sourcedb_info(self):
        # Get number of loaded files
        n_loaded_files = len(
            self.sourcedb.get_all_archive_info()
        )

        # Get number of generated TOAs
        n_generated_toas = len(
            self.sourcedb.get_all_toas()
        )

        # Get number of iterations of timing runs
        n_timing_runs = len(
            self.sourcedb.get_all_timing_info()
        )

        # Get latest timing info
        latest_timing = {"timestamp": 0, "n_toas": 0, "n_bad_toas": 0, "n_params": 0, "chi2r": 0, "mjd_start": 0, "mjd_finish": 0}
        if n_timing_runs > 0:
            latest_timing_entry = self.sourcedb.get_last_timing_info()
            latest_timing["timestamp"] = self.format_timestamp(latest_timing_entry["timestamp"])
            latest_timing["n_toas"] = len(latest_timing_entry["obs_mjds"])
            latest_timing["n_bad_toas"] = len(latest_timing_entry["notes"]["bad_toa_mjds"])
            latest_timing["n_params"] = len(latest_timing_entry["unfreeze_params"])

            if "CHI2R" in latest_timing_entry["fitted_params"]: # older version of pint may not have CHI2R by default
                latest_timing["chi2r"] = latest_timing_entry["fitted_params"]["CHI2R"]

            if len(latest_timing_entry["obs_mjds"]) > 0:
                latest_timing["mjd_start"] = min(latest_timing_entry["obs_mjds"])
                latest_timing["mjd_finish"] = max(latest_timing_entry["obs_mjds"])
        else:
            latest_timing["notes"] = "No timing runs yet."


        # Get dealias info
        n_dealiased = len(
            self.sourcedb.get_all_dealias_history()
        )
        latest_dealiased_info = {}
        if n_dealiased > 0:
            latest_dealias_info = self.sourcedb.get_last_dealias_history()
            latest_dealiased_info["timestamp"] = self.format_timestamp(latest_dealias_info["timestamp"])
            latest_dealiased_info["alias_factor"] = latest_dealias_info["alias_factor"]
            latest_dealiased_info["n_stacked"] = latest_dealias_info["n_stacked"]
            latest_dealiased_info["finishing_remark"] = latest_dealias_info["notes"]["remark"]

        else:
            latest_dealiased_info["notes"] = "No dealiasing yet."

        return {
            "summary": {
                "n_loaded_files": n_loaded_files,
                "n_generated_toas": n_generated_toas,
                "n_timing_runs": n_timing_runs,
            }, 
            "latest_timing": latest_timing,
            "dealiasing": {
                "n_dealiased": n_dealiased,
                "latest_dealiasing": latest_dealiased_info
            }
        }

    def format_timestamp(self, ts):
        # never run
        if ts <= 0:
            return "Never"

        # calculate time offset
        time_offset = time.time() - ts
        offset_str = ""
        if time_offset < 60:
            offset_str = f"{ts}, {int(time_offset)} seconds ago"
        if time_offset < 24 * 3600:
            offset_str = f"{ts}, {int(time_offset // 3600)} hours ago"
        elif time_offset < 7 * 24 * 3600:
            offset_str = f"{ts}, {int(time_offset // (24 * 3600))} days ago"
        elif time_offset < 30 * 24 * 3600:
            offset_str = f"{ts}, {int(time_offset // (7 * 24 * 3600))} weeks ago"
        elif time_offset < 365 * 24 * 3600:
            offset_str = f"{ts}, {int(time_offset // (30 * 24 * 3600))} months ago"
        else:
            offset_str = f"{ts}, {int(time_offset // (365 * 24 * 3600))} years ago"

        return str(datetime.datetime.fromtimestamp(ts).isoformat()) + f" ({offset_str})"
