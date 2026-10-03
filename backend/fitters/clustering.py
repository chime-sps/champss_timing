import copy
import tqdm
import numpy as np
import astropy.units as u
from pint import logging

from .downhillwls import LenientDownhillWLSFitter

# Mute logging from PINT to avoid flushing terminal output with too many messages
logging.setup(level="ERROR")

class ClusteredTOAs:
    def __init__(self, toas):
        # Get toa clusters
        self.cluster_0, self.cluster_1 = self.get_clusters(toas)

    def get_clusters(self, toas):
        # Get the MJDs from the TOAs
        mjds = toas.get_mjds()

        # Calculate diff between consecutive MJDs
        diffs = np.diff(mjds)
        
        # Find out where the data has unusually large gaps
        gap_limit = np.mean(diffs) + 2 * np.std(diffs)
        
        # Get clusters labels
        cluster_labels = toas.get_clusters(gap_limit=gap_limit)

        # Split toas into two clusters
        cluster_0 = toas[cluster_labels != max(cluster_labels)]
        cluster_1 = toas[cluster_labels == max(cluster_labels)]

        return cluster_0, cluster_1

    def fit_with_phase_offset(self, model, phase_offset=0):
        # Make copy of m and t
        m = copy.deepcopy(model)

        # Remove PHOFF component in model
        if hasattr(m, "PHOFF"):
            m.remove_component("PhaseOffset")

        # Fit t1 iteratively
        for i in range(len(self.cluster_1)):
            # Get this t1 after truncating
            this_t0, this_t1 = copy.deepcopy(self.cluster_0), copy.deepcopy(self.cluster_1)[:i+1]

            # Compute pulse number for the truncated t1
            this_t0.compute_pulse_numbers(m)
            this_t1.compute_pulse_numbers(m)

            # Add phase offset to the truncated t1 on the first iteration
            if i == 0:
                this_t1.table["delta_pulse_number"] += phase_offset

            # Initialize fitter with combined TOAs
            f = LenientDownhillWLSFitter(this_t0 + this_t1, m, track_mode="use_pulse_numbers")

            # Fit the model
            try:
                f.fit_toas(maxiter=10)
            except Exception as e:
                print(f"Fitting failed at phase offset {phase_offset}. Error: {e}")
                continue

            # Overwrite the model with the newly fitted model
            m = f.model

        return f

    def get_all_mjds(self):
        return (self.cluster_0 + self.cluster_1).get_mjds()

    def get_cluster_baselines(self):
        return self.cluster_0.get_mjds().max().value - self.cluster_0.get_mjds().min().value, self.cluster_1.get_mjds().max().value - self.cluster_1.get_mjds().min().value
    
    def get_gap_size(self):
        return self.cluster_1.get_mjds().min().value - self.cluster_0.get_mjds().max().value

    def get_chi2rs(self, model):
        def get_chi2r(toas, model):
            # f = WLSFitter(toas, model)
            f = LenientDownhillWLSFitter(toas, model)
            f.fit_toas(maxiter=10)
            return f.model.CHI2R.value

        # Get chi2r values
        chi2r_c0 = get_chi2r(self.cluster_0, model)
        chi2r_c1 = get_chi2r(self.cluster_1, model)
        chi2r_c0c1 = get_chi2r(self.cluster_0 + self.cluster_1, model)

        return chi2r_c0, chi2r_c1, chi2r_c0c1

    def has_valid_clusters(self):
        return len(self.cluster_0) > 5 and len(self.cluster_1) > 0

class ClusteringFitter:
    def __init__(self, toas, model, verbose=False): 
        self.clustered_toas = ClusteredTOAs(toas)
        self.model = model
        self.verbose = verbose

    def fit_toas(self, maxiter=10, n_offsets=25, center_offset=0, *args, **kwargs):
        # Initialize offsets
        trial_phase_offsets = np.arange(center_offset - n_offsets, center_offset + n_offsets + 1)
        
        # Initialize results
        postfit_fitters_states = []
        postfit_chi2rs = []

        # Loop over all trial phase offsets
        for phase_offset in tqdm.tqdm(trial_phase_offsets, desc="Running clustering fitter trials"):
            if self.verbose:
                print(f"Trying phase offset: {phase_offset}")

            # Fit the model with the current phase offset
            fitter_state = self.clustered_toas.fit_with_phase_offset(self.model, phase_offset=phase_offset)

            # Store the results
            postfit_fitters_states.append(fitter_state)
            postfit_chi2rs.append(fitter_state.model.CHI2R.value)

            if self.verbose:
                print(f"Phase offset: {phase_offset}, CHI2R: {fitter_state.model.CHI2R.value}")

        # Find the model with lowest chi2r
        best_index = np.argmin(postfit_chi2rs)
        best_fitter_state = postfit_fitters_states[best_index]

        # Check if the best state is the first or the last trial
        if best_index == 0 or best_index == len(trial_phase_offsets) - 1:
            if maxiter > 1:
                # Retry with the best phase offset as the new center
                return self.fit_toas(
                    maxiter=maxiter-1, 
                    n_offsets=n_offsets, 
                    center_offset=trial_phase_offsets[best_index], 
                    *args, 
                    **kwargs
                )

        # Plot for verbose
        if self.verbose:
            import matplotlib.pyplot as plt

            # Get mjds
            mjds = self.clustered_toas.get_all_mjds()

            # Get the fitter state without any offsets
            no_offset_fitter_state = self.clustered_toas.fit_with_phase_offset(self.model, phase_offset=0)

            # Get residuals without phase offset
            no_offset_r_vals = no_offset_fitter_state.resids.time_resids.to(u.s).value
            no_offset_r_errs = no_offset_fitter_state.resids.get_data_error(scaled=True).to(u.s).value

            # Get residuals at the best state
            best_state_r_vals = best_fitter_state.resids.time_resids.to(u.s).value
            best_state_r_errs = best_fitter_state.resids.get_data_error(scaled=True).to(u.s).value

            # Plot diagnostics
            _, ax = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
            ax[0].errorbar(mjds, no_offset_r_vals, yerr=no_offset_r_errs, fmt='x', capsize=3)
            ax[0].set_xlabel("MJD")
            ax[0].set_ylabel("Residuals (s)")
            ax[0].set_title("Residuals without Phase Offset")
            ax[1].errorbar(mjds, best_state_r_vals, yerr=best_state_r_errs, fmt='x', capsize=3)
            ax[1].set_xlabel("MJD")
            ax[1].set_ylabel("Residuals (s)")
            ax[1].set_title("Residuals with Best Phase Offset")
            ax[2].plot(trial_phase_offsets, postfit_chi2rs, marker='o')
            ax[2].axvline(trial_phase_offsets[best_index], color='r', linestyle='--', label='Best Phase Offset')
            ax[2].set_yscale('log') 
            ax[2].legend()
            ax[2].set_xlabel("Phase Offset")
            ax[2].set_ylabel("CHI2R")
            ax[2].set_title("CHI2R vs Phase Offset")

            # import time
            # plt.savefig(f"clustering_fit_{int(time.time())}.png")

            plt.show()

        return best_fitter_state
