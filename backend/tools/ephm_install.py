import matplotlib.pyplot as plt
import astropy.units as u
import copy
import numpy as np
import pint.toa
from pint.models import get_model
from pint.polycos import Polycos
from ..utils.correlation import fourier_shifts

class EphmInstall:
    def __init__(self, amps, freq, epoch, site):
        # set data
        self.amps = amps
        self.freq = freq
        self.epoch = epoch
        self.site = site

        # set model variables
        self.model = None
        self.jump_phase = 0.0
        self.amps_model_installed = None

    def calc_phase(self, model, DE_eph, meth="model.phase"):
        '''
        Calculate phase at self.epoch using model and method specified.
        meth: "model.phase" or "polycos"
        '''

        # calculate phase using model.phase or polycos
        if meth == "model.phase":
            return model.phase(
                pint.toa.get_TOAs_array(
                    [self.epoch], 
                    obs=self.site, 
                    freqs=[self.freq]*u.MHz, 
                    ephem=DE_eph
                ), abs_phase=True
            )
        elif meth == "polycos":
            p = Polycos.generate_polycos(model, self.epoch-0.0208333333, self.epoch+0.0208333333, self.site, 3599.9999999999998, 12, self.freq)
            return p.eval_abs_phase(self.epoch)
        else:
            raise ValueError("meth must be 'model.phase' or 'polycos'")
    
    def roll_profile_by_phase(self, profile, phase):
        '''
        Roll profile by phase.
        '''

        # if no roll needed
        if phase == 0:
            return profile

        # roll profile with phase
        bins = len(profile)
        shift = float(phase * bins)
        return fourier_shifts.roll(profile, shift)
    
    def install_model(self, model, DE_eph="DE440"):
        '''
        Install model to profile by calculating phase at self.epoch and rolling profile.
        '''

        # get amplitudes and calculate phase
        amps = copy.deepcopy(self.amps)
        phase = self.calc_phase(model, DE_eph=DE_eph)

        # roll profile with phase
        amps = self.roll_profile_by_phase(amps, phase.frac[0])

        # commit changes
        self.model = model
        self.amps_model_installed = amps

        return amps
    
    def install_parfile(self, parfile, DE_eph="DE440"):
        '''
        Install parfile to profile by calculating phase at self.epoch and rolling profile.
        '''

        return self.install_model(get_model(parfile), DE_eph=DE_eph)

    def jump_by_phase(self, phase):
        '''
        Jump parfile installed profile by phase.
        '''

        # get copy of amplitudes
        amps = copy.deepcopy(self.amps_model_installed)
        if amps is None:
            raise ValueError("Model not installed yet. Please install model before jumping.")

        # get amplitudes and roll profile with phase
        amps = self.roll_profile_by_phase(amps, phase)

        # commit changes
        self.jump_phase += phase
        self.amps_model_installed = amps

        return amps

    def jump_by_time_given_model(self, model, time):
        '''
        Jump parfile installed profile by time given a PINT model.
        time: time in seconds
        '''

        # get spin frequency from model
        f0 = model.F0.value  # in Hz

        # calculate phase change
        phase = time * f0

        return self.jump_by_phase(phase)

    def jump_by_time_given_parfile(self, parfile, time):
        '''
        Jump parfile installed profile by time given a parfile.
        time: time in seconds
        '''

        return self.jump_by_time_given_model(get_model(parfile), time)

    def get_init_amps(self):
        return copy.deepcopy(self.amps)

    def get_model_installed_amps(self):
        return copy.deepcopy(self.amps_model_installed)

    def plot(self, savepath=None):
        '''
        Plot profile before and after (if any) model installation.
        '''

        # pre-install
        plt.plot(self.amps, c="k", lw=1)
        plt.title(f"Pulse Profile at {self.freq} MHz")
        plt.xlabel("Phase Bin")
        plt.ylabel("Amplitude")

        # post-install
        if self.amps_model_installed is not None:
            plt.plot(self.amps_model_installed, c="r", lw=1, ls="--", label="Model Installed")
            plt.legend()

        # save figure
        if savepath:
            plt.savefig(savepath)
        plt.show()