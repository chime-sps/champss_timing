import pint
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
        log.trace(f"Using WLSFitter (CHAMPSS Timing Pipeline version)")
        super().__init__(*args, **kwargs)

    def fit_toas(
        self, maxiter: int = 1, threshold: Optional[float] = None, debug: bool = False
    ) -> float:
        """Run a linear weighted least-squared fitting method.

        Parameters
        ----------
        maxiter: int
            Repeat the least-squares fitting up to this many times if necessary.
        threshold : float or None
            Discard singular values smaller than ``threshold`` times the largest
            singular value. If None, use a value based on floating-point epsilon
            and the matrix sizes.
        """
        # check that params of timing model have necessary components
        self.model.validate()
        self.model.validate_toas(self.toas)
        chi2 = 0
        for _ in range(maxiter):
            fitp = self.model.get_params_dict("free", "quantity")
            fitpv = self.model.get_params_dict("free", "num")
            fitperrs = self.model.get_params_dict("free", "uncertainty")
            # Define the linear system
            M, params, units = self.get_designmatrix()
            # Get residuals and TOA uncertainties in seconds
            self.update_resids()
            residuals = self.resids.time_resids.to(u.s).value
            sigma = self.model.scaled_toa_uncertainty(self.toas).to(u.s).value

            dpars, Sigma, norm, _ = fit_wls_svd(
                residuals,
                sigma,
                M,
                params,
                (threshold if threshold is not None else 1e-14 * max(M.shape)),
            )

            # errs = np.sqrt(np.diag(Sigma)) / fac

            errors = np.sqrt(np.diag(Sigma))

            # covariance matrix stuff (for randomized models in pintk)
            # sigma_var = (Sigma / fac).T / fac
            # errors = np.sqrt(np.diag(sigma_var))
            Sigma_cov = (Sigma / errors).T / errors
            # covariance matrix = variances in diagonal, used for gaussian random models
            covariance_matrix = Sigma
            covariance_matrix_labels = {
                param: (i, i + 1, unit)
                for i, (param, unit) in enumerate(zip(params, units))
            }
            # covariance matrix is 2D and symmetric
            covariance_matrix_labels = [
                covariance_matrix_labels
            ] * covariance_matrix.ndim
            self.parameter_covariance_matrix = CovarianceMatrix(
                covariance_matrix, covariance_matrix_labels
            )

            # correlation matrix = 1s in diagonal, use for comparison to tempo/tempo2 cov matrix
            self.parameter_correlation_matrix = CorrelationMatrix(
                Sigma_cov, covariance_matrix_labels
            )
            self.fac = norm
            self.errors = errors

            # The delta-parameter values
            #   dpars = V s^-1 U^T r
            # Scaling by fac recovers original units
            # dpars = np.dot(Vt.T, np.dot(U.T, residuals) / s) / fac

            # for pn in fitp.keys():
            #     uind = params.index(pn)  # Index of designmatrix
            #     un = 1.0 / (units[uind])  # Unit in designmatrix
            #     un *= u.s
            #     pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
            #     fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
            #     # NOTE We need some way to use the parameter limits.
            #     fitperrs[pn] = errors[uind]

            for pn in fitp.keys():
                uind = params.index(pn)
                un = 1.0 / (units[uind]) * u.s
                pv = fitpv[pn] * fitp[pn].units
                dpv = dpars[uind] * un

                # Apply constraint to avoid unphysical jumps in RA and DEC. 
                # This is a bit hacky, but it prevents the fit from diverging when some bad outliers presents in the data. 
                if pn in ("RA", "RAJ", "DEC", "DECJ"):
                    dpv_deg = dpv.to(u.deg).value
                    if abs(dpv_deg) > 0.5: 
                        log.warning(f"{pn} step {dpv_deg:.4f} deg exceeds beam limit, clamping...")
                        dpv = np.sign(dpv_deg) * 0.5 * u.deg
                        dpv = dpv.to(fitp[pn].units)

                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                fitperrs[pn] = errors[uind]

            chi2 = self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

        self.update_model(chi2)

        return chi2