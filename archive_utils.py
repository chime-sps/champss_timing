import psrchive

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