import copy
import tqdm
import numpy as np
import astropy.units as u
from pint import logging

from .downhillwls import LenientDownhillWLSFitter

# Mute logging from PINT to avoid flushing terminal output with too many messages
logging.setup(level="ERROR")

class ClusteredTOAs:
    def __init__(self, toas, verbose=False):
        # Get toa clusters
        self.cluster_0, self.cluster_1 = self.get_clusters(toas)
        self.verbose = verbose

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

    def fit_with_phase_offset(self, model, phase_offset=0, maxiter=100, miniter=30, fit_iteratively=False):
        """
        Fit the model to the TOAs in cluster_1 iteratively, adding a phase offset on the first iteration.
        
        Parameters:
        model : astropy.modeling.Model
            The timing model to fit.
        phase_offset : float, optional
            The phase offset to apply to the first iteration of cluster_1, by default 0.
        maxiter : int, optional
            The maximum number of iterations for the fitter, by default 100.
            If maxiter is equal to or less than 0, the fitting will be skipped and return prefit state. 
        
        Returns:
        LenientDownhillWLSFitter
            The fitter after the final iteration.
        """

        # Make copy of m and t
        m = copy.deepcopy(model)

        if fit_iteratively:
            # Fit t1 iteratively
            for i in range(len(self.cluster_1)):
                # Get this t1 after truncating
                this_t0, this_t1 = copy.deepcopy(self.cluster_0), copy.deepcopy(self.cluster_1)[:i+1]

                # Make a copy of model
                this_model = copy.deepcopy(m)

                # Compute pulse number for the truncated t1
                this_t0.compute_pulse_numbers(this_model)
                this_t1.compute_pulse_numbers(this_model)

                # Remove PHOFF component in model
                if hasattr(this_model, "PHOFF"):
                    this_model.remove_component("PhaseOffset")

                # Add phase offset to the truncated t1 on the first iteration
                if i == 0:
                    this_t1.table["delta_pulse_number"] += phase_offset

                # Initialize fitter with combined TOAs
                f = LenientDownhillWLSFitter(this_t0 + this_t1, this_model, track_mode="use_pulse_numbers")

                # Check if maxiter is valid
                if maxiter <= 0:
                    continue # continue without fitting

                # Fit the model
                try:
                    f.fit_toas(maxiter=maxiter, miniter=miniter)
                except Exception as e:
                    print(f"Fitting failed at phase offset {phase_offset}. Error: {e}")
                    continue

                # Overwrite the model with the newly fitted model
                m = f.model
        else:
            # Get this t0 and t1
            this_t0, this_t1 = copy.deepcopy(self.cluster_0), copy.deepcopy(self.cluster_1)

            # Make a copy of model
            this_model = copy.deepcopy(m)
            
            # Compute pulse number for the truncated t1
            this_t0.compute_pulse_numbers(this_model)
            this_t1.compute_pulse_numbers(this_model)

            # Remove PHOFF component in model
            if hasattr(this_model, "PHOFF"):
                this_model.remove_component("PhaseOffset")

            # Add phase offset to t1
            this_t1.table["delta_pulse_number"] += phase_offset

            f = LenientDownhillWLSFitter(this_t0, this_model, track_mode="use_pulse_numbers")
            f.fit_toas(maxiter=maxiter, miniter=miniter)
            this_model = f.model

            # Initialize fitter with combined TOAs
            f = LenientDownhillWLSFitter(this_t0 + this_t1, this_model, track_mode="use_pulse_numbers")

            # Check if maxiter is valid
            if maxiter > 0:
                try:
                    f.fit_toas(maxiter=maxiter, miniter=miniter)
                except Exception as e:
                    print(f"Fitting failed at phase offset {phase_offset}. Error: {e}")

        if self.verbose:
            import matplotlib.pyplot as plt

            # Get mjds
            mjds, idx0, idx1 = self.get_all_mjds()

            # Get residuals
            r_vals = f.resids.time_resids.to(u.s).value
            r_errs = f.resids.get_data_error(scaled=True).to(u.s).value

            plt.figure(figsize=(10, 5))
            plt.errorbar(mjds[idx0], r_vals[idx0], yerr=r_errs[idx0], fmt='kx', label='Cluster 0 Residuals', capsize=3)
            plt.errorbar(mjds[idx1], r_vals[idx1], yerr=r_errs[idx1], fmt='rx', label='Cluster 1 Residuals', capsize=3)
            plt.xlabel('MJD')
            plt.ylabel('Residuals')
            plt.legend()
            plt.show()

        return f

    def get_all_mjds(self):
        return (self.cluster_0 + self.cluster_1).get_mjds(), np.arange(len(self.cluster_0)), len(self.cluster_0) + np.arange(len(self.cluster_1))

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
        self.clustered_toas = ClusteredTOAs(toas, verbose=verbose)
        self.model = model
        self.verbose = verbose

    def fit_toas(self, maxiter=20, miniter=5, maxiter_trials=10, n_offsets=10, center_offset=0, prev_iter_res=None, *args, **kwargs):
        """
        Fit the TOAs with multiple phase offsets and return the best fitting state. 
        The function will ignore any extra arguments and keyword arguments passed to it.

        Parameters:
        maxiter (int): Maximum number of iterations for LenientDownhillWLSFitter.
        miniter (int): Minimum number of iterations for LenientDownhillWLSFitter.
        maxiter_trials (int): Maximum number of trials for re-centering the phase offset.
        n_offsets (int): Number of phase offsets to try on each side of the center.
        center_offset (int): The central phase offset to start trials from.
        prev_iter_res (dict or None): Dictionary containing results from the previous iteration, including 'trial_phase_offsets', 'postfit_fitters_states', and 'postfit_chi2rs'. If None, a new set of results will be initialized.
        
        Returns:
        ClusteringFitter: The best fitting state after trying all phase offsets.
        """
        
        # Initialize results
        trial_phase_offsets = []
        postfit_fitters_states = []
        postfit_chi2rs = []

        if prev_iter_res is not None:
            trial_phase_offsets = prev_iter_res.get('trial_phase_offsets', trial_phase_offsets)
            postfit_fitters_states = prev_iter_res.get('postfit_fitters_states', postfit_fitters_states)
            postfit_chi2rs = prev_iter_res.get('postfit_chi2rs', postfit_chi2rs)

        # Loop over all trial phase offsets
        for phase_offset in tqdm.tqdm(
            list(np.arange(center_offset - n_offsets, center_offset + n_offsets + 1)), 
            desc="Running clustering fitter trials"
        ):
            if self.verbose:
                print(f"Trying phase offset: {phase_offset}")

            if phase_offset in trial_phase_offsets:
                if self.verbose:
                    print(f"Phase offset {phase_offset} already tried. Skipping.")
                continue

            # Fit the model with the current phase offset
            fitter_state = self.clustered_toas.fit_with_phase_offset(
                self.model, 
                phase_offset=phase_offset, 
                maxiter=maxiter, 
                miniter=miniter
            )

            # Store the results
            trial_phase_offsets.append(phase_offset)
            postfit_fitters_states.append(fitter_state)
            postfit_chi2rs.append(fitter_state.model.CHI2R.value)

            if self.verbose:
                print(f"Phase offset: {phase_offset}, CHI2R: {fitter_state.model.CHI2R.value}")

        # Find the model with lowest chi2r
        best_index = np.argmin(postfit_chi2rs)
        best_fitter_state = postfit_fitters_states[best_index]

        # Check if there are more than one best chi2r
        best_chi2r = postfit_chi2rs[best_index]
        best_indices = [i for i, chi2r in enumerate(postfit_chi2rs) if chi2r == best_chi2r]
        if len(best_indices) > 1:
            if self.verbose:
                print(f"Multiple best chi2r found at indices: {best_indices}. Returning prefit state.")
            return self.clustered_toas.fit_with_phase_offset(self.model, phase_offset=0, maxiter=0)

        # Check if the best state is the first or the last trial
        if best_index == 0 or best_index == len(trial_phase_offsets) - 1:
            if maxiter_trials > 1:
                # Retry with the best phase offset as the new center
                return self.fit_toas(
                    maxiter=maxiter, 
                    miniter=miniter,
                    maxiter_trials=maxiter_trials-1,
                    n_offsets=n_offsets, 
                    center_offset=trial_phase_offsets[best_index], 
                    prev_iter_res={
                        "trial_phase_offsets": trial_phase_offsets,
                        "postfit_fitters_states": postfit_fitters_states,
                        "postfit_chi2rs": postfit_chi2rs
                    },
                    *args, 
                    **kwargs
                )

        # Plot for verbose
        if self.verbose:
            import matplotlib.pyplot as plt

            # Get mjds
            mjds, i_c0, i_c1 = self.clustered_toas.get_all_mjds()

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
            ax[0].errorbar(mjds[i_c0], no_offset_r_vals[i_c0], yerr=no_offset_r_errs[i_c0], fmt='kx', capsize=3)
            ax[0].errorbar(mjds[i_c1], no_offset_r_vals[i_c1], yerr=no_offset_r_errs[i_c1], fmt='rx', capsize=3)
            ax[0].set_xlabel("MJD")
            ax[0].set_ylabel("Residuals (s)")
            ax[0].set_title("Residuals without Phase Offset")
            ax[1].errorbar(mjds[i_c0], best_state_r_vals[i_c0], yerr=best_state_r_errs[i_c0], fmt='kx', capsize=3)
            ax[1].errorbar(mjds[i_c1], best_state_r_vals[i_c1], yerr=best_state_r_errs[i_c1], fmt='rx', capsize=3)
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
