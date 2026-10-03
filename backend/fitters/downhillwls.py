from pint.fitter import DownhillWLSFitter, ConvergenceFailure

class LenientDownhillWLSFitter(DownhillWLSFitter):
    def fit_toas(self, *args, **kwargs):
        try:
            return super().fit_toas(*args, **kwargs)
        except ConvergenceFailure as e:
            return None