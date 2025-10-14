import io
from astropy import units as u
from pint.models import get_model_and_toas
from pint.fitter import WLSFitter
from pint.residuals import Residuals

class PintAPI:
    def __init__(self, parfile, timfile):
        self.m, self.t = get_model_and_toas(io.StringIO(parfile), io.StringIO(timfile))
        self.f = WLSFitter(self.t, self.m)

    def get_resids(self):
        # get residuals
        r = Residuals(self.t, self.m)

        # Format results
        result = {
            "model": self.m.as_parfile(),
            "toas": [float(v) for v in self.t.get_mjds().to(u.us).value],
            "resid_val": [float(v) for v in r.time_resids.to(u.us).value],
            "resid_err": [float(v) for v in self.t.get_errors().to(u.us).value], 
            "ids": [v["name"] for v in self.t.table["flags"]]
        }

        return result

    def fit(self):
        # Fit TOAs
        self.f.fit_toas()
        
        # Format results
        result = {
            "model": self.f.model.as_parfile(),
            "toas": [float(v) for v in self.t.get_mjds().to(u.us).value],
            "resid_val": [float(v) for v in self.f.resids.time_resids.to(u.us).value],
            "resid_err": [float(v) for v in self.t.get_errors().to(u.us).value],
            "ids": [v["name"] for v in self.t.table["flags"]]
        }
        
        return result