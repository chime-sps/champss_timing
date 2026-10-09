import pint
import copy
import pint.fitter
from pint.fitter import fit_wls_svd
from pint.pint_matrix import (
    CorrelationMatrix,
    CovarianceMatrix
)

import numpy as np
import copy
import contextlib
import astropy.units as u
from loguru import logger as log
from typing import Optional


####################################################################
# Overwrite pint.fitter.WLSFitter to add constraints on RA and DEC #
####################################################################

class WLSFitter(pint.fitter.WLSFitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_toas(self, *args, **kwargs):
        '''
        Overwrite the fit_toas method to add additional sanity checks on the fitted parameters. 
        '''

        # Make a copy of prefit state
        prefit_state = copy.deepcopy(self)

        # Fit the TOAs using the parent class method
        res = super().fit_toas(*args, **kwargs)

        # Check for convergence
        if self.resids.chi2 >= prefit_state.resids.chi2:
            raise ValueError(f"Fitting failed. Post-fit CHI2R is not better than pre-fit CHI2R (post={self.resids.chi2}, pre={prefit_state.resids.chi2}). ")

        # Check if the output model has physical parameters 
        raj = self.model.RAJ.quantity.to(u.deg).value
        decj = self.model.DECJ.quantity.to(u.deg).value
        if raj < 0 or raj >= 360:
            raise ValueError(f"Fitting failed. Postfit RAJ is not physical (RAJ={raj}). ")
        if decj < -90 or decj > 90:
            raise ValueError(f"Fitting failed. Postfit DECJ is not physical (DECJ={decj}). ")

        return res