import pint.fitter

import numpy as np
import copy
import contextlib
from loguru import logger as log

###################################################################
# Overwrite pint.fitter.WLSState to add constraints on RA and DEC #
###################################################################

class WLSState(pint.fitter.WLSState):
    def take_step_model(self, step, lambda_=1):
        """Make a new model reflecting the new parameters."""
        # log.debug(f"Taking step {lambda_} * {list(zip(self.params, step))}")
        new_model = copy.deepcopy(self.model)
        for p, s in zip(self.params, step * lambda_):
            # Set bounds for RA and DEC. 
            # This is for CHIME/CHAMPSS to avoid alias solution. 
            # By testing, s for RA and Dec here are in deg.
            if p == "RA" or p == "RAJ":
                if np.abs(s) > 0.5:
                    log.warning(f"RA hits the upper/lower bound of CHIME beamsize (abs[{s}] > 0.5 deg). Lowering to 0.5 deg.")
                    if s > 0:
                        s = 0.5
                    if s < 0:
                        s = -0.5
            if p == "DEC" or p == "DECJ":
                if np.abs(s) > 0.5:
                    log.warning(f"DEC hits the upper/lower bound of CHIME beamsize (abs[{s}] > 0.5 deg). Lowering to 0.5 deg.")
                    if s > 0:
                        s = 0.5
                    if s < 0:
                        s = -0.5
                    
            try:
                with contextlib.suppress(ValueError):
                    log.trace(f"Adjusting {getattr(self.model, p)} by {s}")
                pm = getattr(new_model, p)
                if pm.value is None:
                    pm.value = 0
                pm.value += s
                # getattr(new_model, p).value = getattr(self.model, p).value + s
                # getattr(self.model, p) + s
                # getattr(new_model, p).value = s
            except AttributeError:
                if p != "Offset":
                    log.warning(f"Unexpected parameter {p}")
        return new_model

# Overwrite the WLSState class in PINT
pint.fitter.WLSState = WLSState

class WLSFitter(pint.fitter.WLSFitter):
    def __init__(self, *args, **kwargs):
        log.trace(f"Using WLSFitter (CHAMPSS ver.)")
        super().__init__(*args, **kwargs)
