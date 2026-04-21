import corner
import numpy as np
import uncertainties
import matplotlib.pyplot as plt


class PosteriorSamples:
    def __init__(self, sampler, labels, burn_in=1000, thin=15):
        self.sampler = sampler
        self.labels = labels
        self.burn_in = burn_in
        self.thin = thin

    def plot_chain(self):
        fig, ax = plt.subplots(len(self.labels), 1, figsize=(10, 6), sharex=True, squeeze=False)
            
        # Get samples without burn-in and thinning
        samples = self.sampler.get_chain(discard=0, thin=1)

        for i in range(len(self.labels)):
            ax[i].plot(samples[:, :, i], alpha=0.5, color='k')
            ax[i].set_ylabel(self.labels[i])
            ax[i].axvline(self.burn_in, color='r', linestyle='--', label='Burn-in')
            ax[i].legend(loc='upper right')
        ax[-1].set_xlabel("Step")   

    def plot_corner(self):
        fig = corner.corner(
            self.sampler.get_chain(flat=True, discard=self.burn_in, thin=self.thin), 
            labels=self.labels, 
            show_titles=True
        )
        plt.show()

    def get_posterior_statistics(self, uncertainty_percentile=(0.16, 0.84)):
        samples = self.sampler.get_chain(flat=True, discard=self.burn_in, thin=self.thin)
        medians = np.median(samples, axis=0)
        lower = np.percentile(samples, uncertainty_percentile[0]*100, axis=0)
        upper = np.percentile(samples, uncertainty_percentile[1]*100, axis=0)
        return medians, lower, upper
    
    def __getattr__(self, name):
        if name in self.labels:
            idx = self.labels.index(name)
            results = self.get_posterior_statistics()
            return uncertainties.ufloat(results[0][idx], (results[2][idx] - results[1][idx]) / 2)
        else:
            raise AttributeError(f"'PosteriorSamples' object has no attribute '{name}'")