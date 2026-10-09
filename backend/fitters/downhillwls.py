from pint.fitter import (
    DownhillWLSFitter,
    ConvergenceFailure,
    MaxiterReached,
    StepProblem,
)

class LenientDownhillWLSFitter(DownhillWLSFitter):
    def fit_toas(self, *args, **kwargs):
        try:
            super().fit_toas(*args, **kwargs)
        except ConvergenceFailure as e:
            return False
        except MaxiterReached as e:
            return False
        except StepProblem as e:
            return False

        return True