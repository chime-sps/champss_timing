import psrchive
import numpy as np


class archive_utils:
    def __init__(self, archive):
        self.archive = psrchive.Archive.load(archive)
        self.data = self.archive.get_data
        self.subint = self.archive.get_Integration(0)
        self.prof = self.subint.get_Profile(0, 0)

    def get_amps(self, tolist=True):
        if tolist:
            return self.prof.get_amps().tolist()

        return self.prof.get_amps()

    def get_snr(self):
        return self.prof.snr()

    def get_bad_channels(self, output_format="list"):  # works similar to get_bad_channel_list.py on Cedar, but without aquiring data on site.
        bad_chans = []

        for i in range(self.subint.get_nchan()):
            for j in range(self.subint.get_npol()):
                if np.std(self.subint.get_Profile(j, i).get_amps()) < 1e-9:
                    bad_chans.append(i)
                    break

        #     this_pow = np.round(self.subint.get_Profile(0, i).get_amps(), 6)
        #     if (this_pow == this_pow[0]).all() and this_pow[0] < 0.0005:
        #         bad_chans.append(i)

        bad_percentage = len(bad_chans) / self.subint.get_nchan()

        if output_format == "clfd":
            return "\n".join([str(i) for i in bad_chans]), bad_percentage

        return bad_chans, bad_percentage