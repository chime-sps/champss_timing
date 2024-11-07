from .database import database
from .utils import utils

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import  astropy.units as u
import datetime

class plot:
    def __init__(self, db_path=None, db_hdl=None):
        self.db_path = db_path
        self.db_hdl = db_hdl
        self.db_loaded = False
        self.timing_info = []
        self.archive_info = []
        self.archive_info_file_inxed = {}

        self.initialize()

    def initialize(self):
        if self.db_hdl is None:
            self.db_hdl = database(self.db_path, readonly=True)
            self.db_hdl.initialize()
            self.db_loaded = True
        
        self.timing_info = self.db_hdl.get_all_timing_info()
        self.archive_info = self.db_hdl.get_all_archive_info()

        if self.db_loaded:
            self.db_hdl.close()

        # toas_file_idxed = {}
        # for this_toa in toas:
        #     toas_file_idxed[this_toa["filename"]] = this_toa

        self.archive_info_file_inxed = {}
        for this_archive in self.archive_info:
            self.archive_info_file_inxed[this_archive["filename"]] = this_archive

    def get_plot_data(self):
        plot_data = {
            # MJDs
            "mjds": [], 
            "bad_toa_mjds": [], 

            # Fitted Residuals
            "resid_mjds": [], 
            "resids": [], 
            "resids_phase": [], 
            "resids_err": [], 
            "resids_err_phase": [], 
            "rms": [], 

            # Bad Residuals
            "bad_resids": [], 
            "bad_resids_phase": [], 
            "bad_resids_err": [],
            "bad_resids_err_phase": [],

            # Diagnostic Data
            "chi2r": [], 
            "snr": [], 
            "n_params": [], 
            "params_x": [], 
            "params_y": [], 

            # Amplitudes
            "amps": [], 
            "amps_normalized": [], 
            "stacked_amps": [], 

            # Other
            "fitted_params": {}, 
            "unfreeze_params": [],
            "mjd_gaps": []
        }

        last_mjd = 0
        for this_timing in self.timing_info:
            i_range = np.where(np.array(this_timing["obs_mjds"]) > last_mjd)[0]
            for i in i_range:
                plot_data["mjds"].append(this_timing["obs_mjds"][i])
                plot_data["chi2r"].append(this_timing["chi2_reduced"])
                plot_data["snr"].append(self.archive_info_file_inxed[this_timing["files"][i]]["psr_snr"])

                plot_data["n_params"].append(len(this_timing["unfreeze_params"]))
                for this_param in this_timing["unfreeze_params"]:
                    plot_data["params_x"].append(this_timing["obs_mjds"][i])
                    plot_data["params_y"].append(this_param)

                # Find mjd gaps and add blanks to amplitude for no observation days
                if len(plot_data["mjds"]) > 1:
                    d_mjd = plot_data["mjds"][-1] - plot_data["mjds"][-2]
                    if d_mjd > 1:
                        for _ in range(int(d_mjd - 1)):
                            plot_data["amps"].append(np.zeros_like(plot_data["amps"][-1]))
                            plot_data["amps_normalized"].append(np.zeros_like(plot_data["amps_normalized"][-1]))
                        plot_data["mjd_gaps"].append([plot_data["mjds"][-2] + 0.75, plot_data["mjds"][-1] - 0.75])

                # Get Amplitude
                plot_data["amps"].append(self.archive_info_file_inxed[this_timing["files"][i]]["psr_amps"])
                
                # Normalize amplitude
                plot_data["amps_normalized"].append(plot_data["amps"][-1])
                plot_data["amps_normalized"][-1] = np.array(plot_data["amps_normalized"][-1]) / max(plot_data["amps_normalized"][-1])
                plot_data["amps_normalized"][-1] = plot_data["amps_normalized"][-1] - min(plot_data["amps_normalized"][-1])

                # get rms
                plot_data["rms"].append(np.sqrt(np.mean(np.array(this_timing["residuals"]["val"])**2)))
            last_mjd = max(this_timing["obs_mjds"])
        
        # get residuals
        plot_data["resids"] = self.timing_info[-1]["residuals"]["val"]
        plot_data["resids_err"] = self.timing_info[-1]["residuals"]["err"]
        plot_data["bad_resids"] = self.timing_info[-1]["notes"]["bad_toa_residuals"]["val"]
        plot_data["bad_resids_err"] = self.timing_info[-1]["notes"]["bad_toa_residuals"]["err"]

        # get bad toa mjds
        if "bad_toa_mjds" in self.timing_info[-1]["notes"]:
            plot_data["bad_toa_mjds"] = self.timing_info[-1]["notes"]["bad_toa_mjds"]

        # get residuals mjds
        if "fitted_mjds" in self.timing_info[-1]["notes"]:
            plot_data["resid_mjds"] = self.timing_info[-1]["notes"]["fitted_mjds"]
        else:
            utils.print_warning("No fitted mjds found in the timing info. Using the original mjds.")
            utils.print_warning("Set the fitted mjds to a time series starting from 0 and incrementing by 1.")
            plot_data["resid_mjds"] = np.arange(len(plot_data["resids"]))

        # Get residuals in phase
        this_F0 = ((1/self.timing_info[-1]["fitted_params"]["F0"]) * u.s).to(u.us).value
        plot_data["resids_phase"] = plot_data["resids"] / this_F0
        plot_data["resids_err_phase"] = plot_data["resids_err"] / this_F0
        plot_data["bad_resids_phase"] = plot_data["bad_resids"] / this_F0
        plot_data["bad_resids_err_phase"] = plot_data["bad_resids_err"] / this_F0
  
        # stack absolute amplitudes
        plot_data["stacked_amps"] = np.sum(plot_data["amps"], axis=0)

        # parameters
        plot_data["fitted_params"] = self.timing_info[-1]["fitted_params"]
        plot_data["unfreeze_params"] = self.timing_info[-1]["unfreeze_params"]

        return plot_data
    
    def _round_axis(self, ticks, n=2):
        rounded_ticks = []
        
        for this_tick in ticks:
            rounded_ticks.append(round(this_tick, n))

        return rounded_ticks

    def _format_y_axis(self, ax):
        for label in ax.get_ymajorticklabels():
            label.set_rotation(90)
            label.set_verticalalignment("center")
    
    def diagnostic(self, savefig=None):
        plot_data = self.get_plot_data()

        # create 4x5 grid of plots
        fig, axs = plt.subplots(4, 5, figsize=(20, 11.25))

        # plot stacked amps
        axs[0, 0].plot(np.tile(plot_data["stacked_amps"], 2), color="k", lw=0.5)
        # axs[0, 0].set_title("Stacked Profile")
        # axs[0, 0].set_xlabel("Phase")
        # axs[0, 0].set_ylabel("Amplitude")
        # axs[0, 0].set_yticklabels(axs[0, 0].get_yticks(), rotation=90, fontdict={"verticalalignment": "center"})
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 0].set_xlim(0, len(plot_data["stacked_amps"]) * 2)
        axs[0, 0].set_title("Stacked Profile")

        # plot amps vertically across 3 grids
        ## remove the underlying Axes
        axs[1, 0].remove()
        axs[2, 0].remove()
        axs[3, 0].remove()
        ## combine the 3 grids
        amps_gs = axs[1, 0].get_gridspec()
        axs_amps = fig.add_subplot(amps_gs[1:, 0])
        axs_amps.matshow(np.tile(plot_data["amps_normalized"], (1, 2)), cmap="gray_r", aspect="auto")
        # axs_amps.set_title("Amplitudes")
        axs_amps.set_xlabel("Phase")
        axs_amps.set_ylabel("MJD")
        # axs_amps.set_yticklabels(self._round_axis(axs_amps.get_yticks(), 1), rotation=90, fontdict={"verticalalignment": "center"})
        y_ticks_loc = np.linspace(1, max(plot_data["mjds"]) - min(plot_data["mjds"]), 7)
        y_ticks_lab = y_ticks_loc + min(plot_data["mjds"])
        axs_amps.set_yticks(y_ticks_loc)
        axs_amps.set_yticklabels(self._round_axis(y_ticks_lab, 1), rotation=90, fontdict={"verticalalignment": "center"})
        axs_amps.set_title("Amplitudes of single observations")
        ## set right axis to mark bad toa mjds
        if len(plot_data["bad_toa_mjds"]) > 0:
            ax_right = axs_amps.twinx()
            ax_right.set_ylim(axs_amps.get_ylim())
            ax_right.set_yticks(np.array(plot_data["bad_toa_mjds"]) - min(plot_data["mjds"]))
            ax_right.set_yticklabels(["←"] * len(plot_data["bad_toa_mjds"]), fontdict={"verticalalignment": "center"}, color="r")

        # plot residuals horizontally across 2 grids
        ## remove the underlying Axes
        axs[0, 1].remove()
        axs[0, 2].remove()
        axs[0, 3].remove()
        ## combine the 2 grids
        resids_gs = axs[0, 1].get_gridspec()
        axs_resids = fig.add_subplot(resids_gs[0, 1:4])
        ## plot residuals
        axs_resids.errorbar(plot_data["resid_mjds"], plot_data["resids_phase"], plot_data["resids_err_phase"], fmt="x", c="k", capsize=3, label="Fitted")
        ## set axis limits
        lim_0, lim_1 = axs_resids.get_ylim()
        lim_0, lim_1 = (-max([np.abs(lim_0), np.abs(lim_1)]), max([np.abs(lim_0), np.abs(lim_1)]))
        if lim_1 < 0.001: lim_0, lim_1 = (-0.001, 0.001)
        axs_resids.set_ylim(lim_0, lim_1)
        ## plot residuals for bad toas
        # if len(plot_data["bad_toa_mjds"]) > 0:
            # axs_resids.errorbar(plot_data["bad_toa_mjds"], plot_data["bad_resids_phase"], plot_data["bad_resids_err_phase"], fmt="x", c="r", capsize=3, label="Bad TOAs")
        for i in range(len(plot_data["bad_toa_mjds"])):
            if plot_data["bad_resids_phase"][i] > 0:
                axs_resids.text(plot_data["bad_toa_mjds"][i], lim_1, f"{round(plot_data['bad_resids_phase'][i], 2)} → ", c="red", label="Bad TOAs", horizontalalignment="center", verticalalignment="top", rotation=90)
            else:
                axs_resids.text(plot_data["bad_toa_mjds"][i], lim_0, f" ← {round(plot_data['bad_resids_phase'][i], 2)}", c = "red", label="Bad TOAs", horizontalalignment="center", verticalalignment="bottom", rotation=90)
        axs_resids.axhline(y=0, color="k", linestyle="--", alpha=0.25)
        axs_resids.set_xlabel("MJD")
        axs_resids.set_ylabel("Timing Residuals (phase)")
        self._format_y_axis(axs_resids)
        axs_resids.set_title("Diagnostic Plots")
        ## fill mjd gaps
        for this_gap in plot_data["mjd_gaps"]:
            axs_resids.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10, label="No Observation")
        # self.__legend_without_duplicate_labels(axs_resids)

        # plot chi2r horizontally across 2 grids
        ## remove the underlying Axes
        axs[1, 1].remove()
        axs[1, 2].remove()
        axs[1, 3].remove()
        ## combine the 2 grids
        chi2r_gs = axs[1, 1].get_gridspec()
        axs_chi2r = fig.add_subplot(chi2r_gs[1, 1:4])
        axs_chi2r.axhline(y=1, color="k", linestyle="--", alpha=0.25)
        axs_chi2r.plot(plot_data["mjds"], plot_data["chi2r"], "kx", label="CHI2R")
        axs_chi2r.set_yscale("log")
        # axs_chi2r.set_title("Reduced Chi2")
        axs_chi2r.set_xlabel("MJD")
        axs_chi2r.set_ylabel("Reduced Chi2")
        self._format_y_axis(axs_chi2r)
        ## fill mjd gaps
        lim_0, lim_1 = axs_chi2r.get_ylim()
        axs_chi2r.set_ylim(lim_0, lim_1)
        for this_gap in plot_data["mjd_gaps"]:
            axs_chi2r.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10)
        # self.__legend_without_duplicate_labels(axs_chi2r)

        # # plot n_params horizontally across 2 grids
        # ## remove the underlying Axes
        # axs[2, 1].remove()
        # axs[2, 2].remove()
        # axs[2, 3].remove()
        # ## combine the 2 grids
        # n_params_gs = axs[2, 1].get_gridspec()
        # axs_n_params = fig.add_subplot(n_params_gs[2, 1:4])
        # axs_n_params.plot(plot_data["mjds"], plot_data["n_params"], "kx", label="Nparams")
        # # axs_n_params.set_title("N Params")
        # axs_n_params.set_xlabel("MJD")
        # axs_n_params.set_ylabel("Number of Parameters Fitted")
        # axs_n_params.set_yticklabels(self._round_axis(axs_n_params.get_yticks(), 1), rotation=90, fontdict={"verticalalignment": "center"})
        # ## fill mjd gaps
        # lim_0, lim_1 = axs_n_params.get_ylim()
        # axs_n_params.set_ylim(lim_0, lim_1)
        # for this_gap in plot_data["mjd_gaps"]:
        #     axs_n_params.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10)
        # # self.__legend_without_duplicate_labels(axs_n_params)

        # plot n_params horizontally across 2 grids
        ## remove the underlying Axes
        axs[2, 1].remove()
        axs[2, 2].remove()
        axs[2, 3].remove()
        ## combine the 2 grids
        n_params_gs = axs[2, 1].get_gridspec()
        axs_n_params = fig.add_subplot(n_params_gs[2, 1:4])
        axs_n_params.plot(plot_data["params_x"], plot_data["params_y"], "kx", label="Nparams")
        # axs_n_params.set_title("N Params")
        axs_n_params.set_xlabel("MJD")
        axs_n_params.set_ylabel("Number of Parameters Fitted")
        # axs_n_params.set_yticklabels(self._round_axis(axs_n_params.get_yticks(), 1), rotation=90, fontdict={"verticalalignment": "center"})
        self._format_y_axis(axs_n_params)
        ## fill mjd gaps
        lim_0, lim_1 = axs_n_params.get_ylim()
        axs_n_params.set_ylim(lim_0, lim_1)
        for this_gap in plot_data["mjd_gaps"]:
            axs_n_params.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10)
        # self.__legend_without_duplicate_labels(axs_n_params)

        # # plot rms horizontally across 2 grids
        # axs[3, 1].remove()
        # axs[3, 2].remove()
        # axs[3, 3].remove()
        # ## combine the 2 grids
        # rms_gs = axs[3, 1].get_gridspec()
        # axs_rms = fig.add_subplot(rms_gs[3, 1:4])
        # axs_rms.plot(plot_data["mjds"], plot_data["rms"], "kx", label="RMS")
        # # axs_rms.set_title("RMS")
        # axs_rms.set_xlabel("MJD")
        # axs_rms.set_ylabel("RMS")
        # self._format_y_axis(axs_rms)
        # ## fill mjd gaps
        # lim_0, lim_1 = axs_rms.get_ylim()
        # axs_rms.set_ylim(lim_0, lim_1)
        # for this_gap in plot_data["mjd_gaps"]:
        #     axs_rms.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10)
        # # self.__legend_without_duplicate_labels(axs_rms)
        
        # plot snr horizontally across 2 grids
        ## remove the underlying Axes
        axs[3, 1].remove()
        axs[3, 2].remove()
        axs[3, 3].remove()
        ## combine the 2 grids
        snr_gs = axs[3, 1].get_gridspec()
        axs_snr = fig.add_subplot(snr_gs[3, 1:4])
        axs_snr.plot(plot_data["mjds"], plot_data["snr"], "kx", label="SNR")
        # axs_snr.set_title("SNR")
        axs_snr.set_xlabel("MJD")
        axs_snr.set_ylabel("Signal to Noise Ratio")
        self._format_y_axis(axs_snr)
        ## fill mjd gaps
        lim_0, lim_1 = axs_snr.get_ylim()
        axs_snr.set_ylim(lim_0, lim_1)
        for this_gap in plot_data["mjd_gaps"]:
            axs_snr.fill_between(this_gap, lim_0, lim_1, color="gray", alpha=0.10)
        # self.__legend_without_duplicate_labels(axs_snr)

        # Plot fitted_params as table
        table_data = []
        for key, value in plot_data["fitted_params"].items():
            if key in plot_data["unfreeze_params"]:
                table_data.append(["◼︎ " + key, value])
            else:
                table_data.append(["◻︎ " + key, value])
        axs[0, 4].remove()
        axs[1, 4].remove()
        axs[2, 4].remove()
        axs[3, 4].remove()
        table_gs = axs[0, 3].get_gridspec()
        axs_table = fig.add_subplot(table_gs[:, 4])
        axs_table.axis("off")
        axs_table_hdl = axs_table.table(cellText=table_data, colLabels=["Parameter", "Value"], loc="center", cellLoc="left", colLoc="left")
        axs_table_hdl.auto_set_font_size(False)
        axs_table_hdl.set_fontsize(8)
        axs_table_hdl.scale(1.3, 1.3)
        axs_table.set_title("Fitted Model Parameters")

        # Add date to the right bottom corner
        fig.text(0.999, 0.001, f"PSR {plot_data['fitted_params']['PSR']} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, ha="right", va="bottom")

        # Tight layout
        fig.tight_layout()     

        # Save the figure
        if savefig is not None:
            print(f"  > Diagnostic plot > {savefig}")
            plt.savefig(savefig)   
        else:
            plt.show()

    def __legend_without_duplicate_labels(self, figure):
        # Ref: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys(), loc='lower right')
