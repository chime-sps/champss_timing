import sigpyproc
from sigpyproc.readers import FilReader
from astropy import units
from astropy.coordinates import SkyCoord

def parse_radec(src_raj: float, src_dej: float) -> SkyCoord:
    """Parse Sigproc format RADEC float as Astropy SkyCoord.

    Parameters
    ----------
    src_raj : float
        Sigproc style HHMMSS.SSSS right ascension
    src_dej : float
        Sigproc style DDMMSS.SSSS declination

    Returns
    -------
    :class:`~astropy.coordinates.SkyCoord`
        Astropy coordinate class
    """
    ho, mi = divmod(src_raj, 10000)
    mi, se = divmod(mi, 100)

    sign = -1 if src_dej < 0 else 1
    de, ami = divmod(abs(src_dej), 10000)
    ami, ase = divmod(ami, 100)

    radec_str = f"{int(ho)} {int(mi)} {se} {sign * int(de)} {int(ami)} {ase}"
    try:
        return SkyCoord(radec_str, unit=(units.hourangle, units.deg))
    except Exception as e:
        # Having this to fix an issue of parsing CHIME/Pulsar filterbank data header
        # Since we actually don't use the RADEC information (at least at the moment), 
        # we can just return a placeholder I think...
        print(f"Cannot parse coordinate [{radec_str}]. Returnning (0, 0) as placeholder")
        return SkyCoord("0 0", unit=(units.hourangle, units.deg))
        
sigpyproc.io.sigproc.parse_radec = parse_radec

class FilterbankReader:
    def __init__(self, filterbank):
        # Initialize sigpyproc object
        self.fil = sigpyproc.readers.FilReader(filterbank)

    def get_obs_duration_in_days(self):
        return self.fil.header.tsamp * self.fil.header.nsamples / 86400

    def get_data(self, dm):
        return self.fil.dedisperse(dm).data

    def get_snr(self, dm):
        data = self.get_data(dm)
        return data.max() / data.std()

    def get_mjd(self):
        return self.fil.header.tstart
    
    def get_header(self):
        return self.fil.header.to_dict()
    
    def get_metadata(self):
        header_info = self.get_header()
        header_info["coord"] = [header_info["coord"].ra.deg, header_info["coord"].dec.deg]
        header_info["azimuth"] = header_info["azimuth"].deg
        header_info["zenith"] = header_info["zenith"].deg

        del header_info["dtype"]
        del header_info["chan_freqs"]

        return header_info
    
# fu = filterbank_utils("/Users/wenky/Downloads/S00314_1.fil")
# print(fu.get_obs_duration_in_days())
# print(fu.get_data(0))
# print(fu.get_snr(0))
# print(fu.get_mjd())
# print(fu.get_header())
# print(fu.get_metadata())
