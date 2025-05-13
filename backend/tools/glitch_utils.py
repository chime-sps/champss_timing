import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import time
import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation

from ..datastores.database import database
from ..utils.logger import logger
from ..utils.utils import utils

class glitch_utils():
    def __init__(self, db_hdl=None, db_path=None, logger=logger()):
        if db_hdl is None and db_path is None:
            raise ValueError("Either db_hdl or db_path must be provided")

        self.db_hdl = db_hdl
        self.db_path = db_path
        if db_hdl is None:
            self.db_hdl = database(db_path, readonly=True)
        else:
            self.db_path = None

        self.timing_info_day0 = None
        self.timing_info_day1 = None
        self.profile_day0 = None
        self.profile_day1 = None

        self.logger = logger

    def initialize(self):
        # timing info
        all_timing_info = self.db_hdl.get_all_timing_info()
        self.timing_info_day0 = all_timing_info[-2]
        self.timing_info_day1 = all_timing_info[-1]

        # profile
        all_profiles_filename_idxed = {}
        for archive_info in self.db_hdl.get_all_archive_info():
            all_profiles_filename_idxed[archive_info['filename']] = archive_info
        self.profile_day0 = all_profiles_filename_idxed[self.timing_info_day0['files'][-1]]
        self.profile_day1 = all_profiles_filename_idxed[self.timing_info_day1['files'][-1]]
    
    def calc_glitch(self, P, mjd_day0, mjd_day_1, resid_day0, resid_day1, additional_phase_wrap=0, anti_glitch=False, sideral_day=86164.1, resid_in_phase=False):
        """
        Calculate glitch/anti-glitch strength
        ---
        P: float, period in seconds
        mjd_day0: float, MJD of day 0
        mjd_day_1: float, MJD of day 1
        resid_day0: float, residuals of day 0 in seconds or phase
        resid_day1: float, residuals of day 1 in seconds or phase
        additional_phase_wrap: int, additional phase wrap. For example, if the phase wrap between day 0 and day 1 is lager than 1 but smaller than 2, additional_phase_wrap=1. 
        anti_glitch: bool, if True, calculate anti-glitch
        sideral_day (optional): float, sideral day in seconds
        resid_in_phase (optional): bool, if True, residuals are in phase unit. Otherwise, residuals are in seconds.
        """
        if not resid_in_phase:
            resid_day0 = resid_day0 / P
            resid_day1 = resid_day1 / P

        Delta_phase = resid_day0 - resid_day1
        if anti_glitch: # delta_glitch is negative
            if Delta_phase > 0:
                Delta_phase -= 1
            Delta_phase -= additional_phase_wrap
        else: # delta_glitch is positive
            if Delta_phase < 0:
                Delta_phase += 1
            Delta_phase += additional_phase_wrap
        
        Delta_phase = Delta_phase / (mjd_day_1 - mjd_day0)
        self.logger.debug(f"Delta_phase: {Delta_phase}, anti_glitch: {anti_glitch}, additional_phase_wrap: {additional_phase_wrap}")
            
        N_phaseperday = sideral_day / P
        P_new = sideral_day / (N_phaseperday + Delta_phase)
        Delta_P = P_new - P
        
        return {"dP": Delta_P, "dPP": Delta_P / P}

    def calc_transit_time(self, ra, dec, date, latitude=49.320751, longitude=-119.620811, out_timezone_offset=+8): # default to DRAO
        # Define observer location
        observer_location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)

        # Get source coordinates
        source_coord = SkyCoord.from_name(f"{ra} {dec}")

        # Define time range
        obs_time = Time(f"{date} 00:00:00")  # Start of the given day
        times = obs_time + np.linspace(0, 1, 1000) * u.day  # Generate times throughout the day

        # Convert to AltAz frame
        altaz_frame = AltAz(obstime=times, location=observer_location)
        source_altaz = source_coord.transform_to(altaz_frame)

        # Find the time when altitude is maximized (transit)
        max_alt_index = np.argmax(source_altaz.alt)
        transit_time = times[max_alt_index]
        
        # plt.figure(figsize=(10, 5))
        # plt.plot_date(times.plot_date, source_altaz.alt, fmt="-")
        
        # Convert to output timezone
        transit_time = transit_time - out_timezone_offset * u.hour

        return transit_time
    
    def estimate_glitch(self, savefig=None):
        # Period
        P = 1 / self.timing_info_day0["fitted_params"]["F0"]

        # get_residuals
        mjds = []
        residuals_mjd_idxed = {}
        for i, mjd in enumerate(self.timing_info_day1["notes"]["fitted_mjds"]):
            mjds.append(mjd)
            residuals_mjd_idxed[mjd] = {
                "val": self.timing_info_day1["residuals"]["val"][i],
                "err": self.timing_info_day1["residuals"]["err"][i]
            }
        mjds.sort()

        # Day 0 info
        mjd_day0 = mjds[-2]
        resid_val_day0 = residuals_mjd_idxed[mjd_day0]["val"]
        resid_err_day0 = residuals_mjd_idxed[mjd_day0]["err"]
        self.logger.debug(f"Day 0: MJD={mjd_day0}, Residuals={resid_val_day0} +/- {resid_err_day0}")

        # Day 1 info
        mjd_day1 = mjds[-1]
        resid_val_day1 = residuals_mjd_idxed[mjd_day1]["val"]
        resid_err_day1 = residuals_mjd_idxed[mjd_day1]["err"]
        self.logger.debug(f"Day 1: MJD={mjd_day1}, Residuals={resid_val_day1} +/- {resid_err_day1}")

        # Calculate glitch
        glitch_info = {
            "glitch": {
                "phase_wrap=0": self.calc_glitch(
                    P = P,
                    mjd_day0 = mjd_day0,
                    mjd_day_1 = mjd_day1,
                    resid_day0 = resid_val_day0 * 1e-6, # us to s
                    resid_day1 = resid_val_day1 * 1e-6, # us to s
                    additional_phase_wrap = 0,
                    anti_glitch = False,
                    resid_in_phase = False
                ), 
                "phase_wrap=1": self.calc_glitch(
                    P = P,
                    mjd_day0 = mjd_day0,
                    mjd_day_1 = mjd_day1,
                    resid_day0 = resid_val_day0 * 1e-6, # us to s
                    resid_day1 = resid_val_day1 * 1e-6, # us to s
                    additional_phase_wrap = 1,
                    anti_glitch = False,
                    resid_in_phase = False
                ),
            }, 
            "anti_glitch": {
                "phase_wrap=0": self.calc_glitch(
                    P = P,
                    mjd_day0 = mjd_day0,
                    mjd_day_1 = mjd_day1,
                    resid_day0 = resid_val_day0 * 1e-6, # us to s
                    resid_day1 = resid_val_day1 * 1e-6, # us to s
                    additional_phase_wrap = 0,
                    anti_glitch = True,
                    resid_in_phase = False
                ), 
                "phase_wrap=1": self.calc_glitch(
                    P = P,
                    mjd_day0 = mjd_day0,
                    mjd_day_1 = mjd_day1,
                    resid_day0 = resid_val_day0 * 1e-6, # us to s
                    resid_day1 = resid_val_day1 * 1e-6, # us to s
                    additional_phase_wrap = 1,
                    anti_glitch = True,
                    resid_in_phase = False
                ),
            }
        }

        # print glitch info
        self.logger.info(f"Glitch info:")
        for key in glitch_info:
            self.logger.info(f"  {key}:")
            for subkey in glitch_info[key]:
                self.logger.info(f"    {subkey}:")
                for subsubkey in glitch_info[key][subkey]:
                    self.logger.info(f"      {subsubkey}: {glitch_info[key][subkey][subsubkey]}")

        # Calculate next transit time at DRAO
        next_transit_time = self.calc_transit_time(
            ra = self.timing_info_day0["fitted_params"]["RAJ"],
            dec = self.timing_info_day0["fitted_params"]["DECJ"],
            date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        )

        # Plot diagnostic
        self.plot_diagnostic(
            resid_day0 = {"mjd": mjd_day0, "val": resid_val_day0, "err": resid_err_day0},
            resid_day1 = {"mjd": mjd_day1, "val": resid_val_day1, "err": resid_err_day1}, 
            glitch_info = glitch_info, 
            next_transit_time = next_transit_time, 
            savefig = savefig
        )
    
    def plot_diagnostic(self, resid_day0, resid_day1, glitch_info, next_transit_time, savefig=None):
        fig, ax = plt.subplots(2, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1, 2]})
        P = 1 / self.timing_info_day1["fitted_params"]["F0"]

        # normalize amps
        day0_amps = np.tile(self.normalize_amps(self.profile_day0["psr_amps"]), 2)
        day1_amps = np.tile(self.normalize_amps(self.profile_day1["psr_amps"]), 2)

        # resample amps
        n_samples = np.max([len(day0_amps), len(day1_amps)])
        day0_amps = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(day0_amps)), day0_amps)
        day1_amps = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(day1_amps)), day1_amps)

        # calculate toa shift
        toa_shift = round(
            np.abs(resid_day0["val"] * 1e-6 / P - resid_day1["val"] * 1e-6 / P) * n_samples / 2, 
            2
        )

        # Plot pulse profiles for day 0
        ax[0, 0].plot(day0_amps, color="k", lw=0.5)
        ax[0, 0].plot(day1_amps, color="k", lw=0.25, alpha=0.25)
        ax[0, 0].hlines(0.9, 0, toa_shift, color="k")
        ax[0, 0].text(0, 0.915, f"↓ Size of Observed Phase Shift", ha="left", va="bottom")
        ax[0, 0].set_title(f"Pulse Profile on Day 0 (MJD={int(resid_day0['mjd'])})")
        ax[0, 0].axis("off")

        # Plot pulse profiles for day 1
        ax[1, 0].plot(day0_amps, color="k", lw=0.25, alpha=0.25)
        ax[1, 0].plot(day1_amps, color="k", lw=0.5)
        ax[1, 0].hlines(0.9, 0, toa_shift, color="k")
        ax[1, 0].text(0, 0.915, f"↓ Size of Observed Phase Shift", ha="left", va="bottom")
        ax[1, 0].set_title(f"Pulse Profile on Day 1 (MJD={int(resid_day1['mjd'])})")
        ax[1, 0].axis("off")
        
        # Plot residuals
        ax[0, 1].errorbar(
            np.array(self.timing_info_day1["notes"]["fitted_mjds"])[-30:], 
            np.array(self.timing_info_day1["residuals"]["val"])[-30:] * 1e-6 / P, 
            yerr=np.array(self.timing_info_day1["residuals"]["err"])[-30:] * 1e-6 / P,  
            fmt="x", 
            color="k"
        )
        ax[0, 1].errorbar(
            resid_day0["mjd"],
            resid_day0["val"] * 1e-6 / P,
            yerr=resid_day0["err"] * 1e-6 / P,
            fmt="x", 
            color="blue", 
            label="Day 0 (MJD={:.5f}, residual={:.3f} +/- {:.3f})".format(resid_day0["mjd"], resid_day0["val"] * 1e-6 / P, resid_day0["err"] * 1e-6 / P)
        )
        ax[0, 1].axvline(resid_day0["mjd"], color="blue", linestyle="-", alpha=0.5)
        ax[0, 1].axhline(resid_day0["val"] * 1e-6 / P, color="blue", linestyle="-", alpha=0.5)
        ax[0, 1].errorbar(
            resid_day1["mjd"],
            resid_day1["val"] * 1e-6 / P,
            yerr=resid_day1["err"] * 1e-6 / P,
            fmt="x", 
            color="red", 
            label="Day 1 (MJD={:.5f}, residual={:.3f} +/- {:.3f})".format(resid_day1["mjd"], resid_day1["val"] * 1e-6 / P, resid_day1["err"] * 1e-6 / P)
        )
        ax[0, 1].axvline(resid_day1["mjd"], color="red", linestyle="-", alpha=0.5)
        ax[0, 1].axhline(resid_day1["val"] * 1e-6 / P, color="red", linestyle="-", alpha=0.5)
        ax[0, 1].set_xlabel("MJD")
        ax[0, 1].set_ylabel("Residuals (phase)")
        ax[0, 1].legend(frameon=False)
        ax[0, 1].set_title("Timing Residuals")

        # Plot glitch info
        ax[1, 1].text(0.5, 1 - 1 * 0.07, f"Glitch Information", ha="center", va="center", weight="bold")
        ax[1, 1].text(0, 1 - 2 * 0.07, f"If this is a glitch:", ha="left", va="center")
        ax[1, 1].text(0, 1 - 3 * 0.07, f"  $\\Delta$P > {glitch_info['glitch']['phase_wrap=0']['dP']:.3e} s, $\\Delta$P/P > {glitch_info['glitch']['phase_wrap=0']['dPP']:.3e} (phase_wrap < 1/day)", ha="left", va="center")
        ax[1, 1].text(0, 1 - 4 * 0.07, f"  $\\Delta$P > {glitch_info['glitch']['phase_wrap=1']['dP']:.3e} s, $\\Delta$P/P > {glitch_info['glitch']['phase_wrap=1']['dPP']:.3e} (1 < phase_wrap < 2/day)", ha="left", va="center")
        ax[1, 1].text(0.5, 1 - 2 * 0.07, f"If this is an anti-glitch:", ha="left", va="center")
        ax[1, 1].text(0.5, 1 - 3 * 0.07, f"  $\\Delta$P > {glitch_info['anti_glitch']['phase_wrap=0']['dP']:.3e} s, $\\Delta$P/P > {glitch_info['anti_glitch']['phase_wrap=0']['dPP']:.3e} (phase_wrap < 1/day)", ha="left", va="center")
        ax[1, 1].text(0.5, 1 - 4 * 0.07, f"  $\\Delta$P > {glitch_info['anti_glitch']['phase_wrap=1']['dP']:.3e} s, $\\Delta$P/P > {glitch_info['anti_glitch']['phase_wrap=1']['dPP']:.3e} (1 < phase_wrap < 2/day)", ha="left", va="center")

        # Plot transit info
        ax[1, 1].text(0.5, 1 - 5.5 * 0.07, f"Observation Information", ha="center", va="center", weight="bold")
        ax[1, 1].text(0, 1 - 6.5 * 0.07, f"Next transit time of the source at DRAO: {next_transit_time} (PST)", ha="left", va="center")

        # Plot instructions
        ax[1, 1].text(0.5, 1 - 8 * 0.07, f"Instructions & Checklists", ha="center", va="center", weight="bold")
        ax[1, 1].text(0, 1 - 9 * 0.07, f"☐ Check if this event was caused by a real glitch/anti-glitch, RFI, or timing noise. ", ha="left", va="center")
        ax[1, 1].text(0, 1 - 10 * 0.07, f"☐ Run pdmp on raw archive data to check whether it was a glitch or anti-glitch. ", ha="left", va="center")
        ax[1, 1].text(0, 1 - 11 * 0.07, f"☐ Run pdmp on raw archive data to check whether the calculated $\\Delta$P/P was an aliased solution. ", ha="left", va="center")
        ax[1, 1].text(0, 1 - 12 * 0.07, f"  → If the lower limit of $\\Delta$P/P was GREATER than 3e-6, trigger a follow-up observation. ", ha="left", va="center")
        ax[1, 1].text(0, 1 - 13 * 0.07, f"  → If the lower limit of $\\Delta$P/P was LESS than 3e-6, wait for the next observation (source transit time see above) and estimate $\\Delta$P/P again. ", ha="left", va="center")

        # remove x and y for glitch info
        ax[1, 1].axis("off")

        # plot pipeline info
        psr_id = "UNKN-OWNX"
        if "PSR" in self.timing_info_day0["fitted_params"]:
            psr_id = self.timing_info_day0["fitted_params"]["PSR"]
        elif "PSRJ" in self.timing_info_day0["fitted_params"]:
            psr_id = self.timing_info_day0["fitted_params"]["PSRJ"]
        fig.text(0.001, 0, f"CHAMPSS Timing Pipeline ({utils.get_version_hash()}) glitch_utils | PSR {psr_id} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, ha="left", va="bottom", family="monospace")
        
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()

    def normalize_amps(self, amps):
        amps = np.array(amps)
        amps -= np.min(amps)
        amps /= np.max(amps)
        return amps

    def cleanup(self):
        if self.db_path is not None:
            self.db_hdl.close()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False
        