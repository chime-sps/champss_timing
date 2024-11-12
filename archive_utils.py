import psrchive
import numpy as np

class archive_utils:
    def __init__(self, archive):
        self.archive = psrchive.Archive.load(archive)
        self.data = self.archive.get_data
        self.subint = self.archive.get_Integration(0)
        self.prof = self.subint.get_Profile(0,0)
    
    def get_amps(self, tolist=True):
        if tolist:
            return self.prof.get_amps().tolist()
        
        return self.prof.get_amps()
    
    def get_snr(self):
        return self.prof.snr()

    def get_bad_channels(self, output_format="list"): # works similar to get_bad_channel_list.py on Cedar, but without aquiring data on site.
        bad_chans = []

        for i in range(self.subint.get_nchan()):
            this_pow = np.round(self.subint.get_Profile(0, i).get_amps(), 8)
            if (this_pow == this_pow[0]).all():
                bad_chans.append(i)

        if output_format == "clfd":
            return "\n".join(bad_chans)

        return bad_chans