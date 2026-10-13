from pint.fitter import (
    DownhillWLSFitter,
    ConvergenceFailure,
    MaxiterReached,
    StepProblem,
)
from .wls import GuardedWLSFitter

class LenientDownhillWLSFitter(DownhillWLSFitter):
    def fit_toas(self, *args, **kwargs):

        miniter = 0
        if "miniter" in kwargs:
            miniter = kwargs.pop("miniter")

        if miniter > 0:
            try:
                f = GuardedWLSFitter(self.toas, self.model)
                f.fit_toas(maxiter=miniter)
            except ConvergenceFailure as e:
                return False

            self.model = f.model

        try:
            super().fit_toas(*args, **kwargs)
        except ConvergenceFailure as e:
            return False
        except MaxiterReached as e:
            return False
        except StepProblem as e:
            return False

        return True