import psrchive
import traceback
import numpy as np

class ArchiveReader:
    def __init__(self, archive, dedisperse=True, retries=3):
        # Initialize archive object
        while retries > 0: 
            # retry 3 times to avoid loading error of network file system issue (sometimes it fails to load on Narval)
            try:
                self.archive = psrchive.Archive.load(archive)
                break
            except:
                retries -= 1
                traceback.print_exc()

        if retries == 0:
            raise Exception(f"Failed to load archive: {archive}")

        # Dedisperse
        if dedisperse:
            self.archive.dedisperse()

        # Get data, subint, and profile objects
        self.subint = self.archive.get_Integration(0)
        self.prof = self.subint.get_Profile(0, 0)

    def get_amps(self, tolist=True):
        if tolist:
            return self.prof.get_amps().tolist()

        return self.prof.get_amps()

    def get_obs_duration_in_days(self):
        return (self.archive.end_time() - self.archive.start_time()).in_days()

    def get_data(self):
        return self.archive.get_data()

    def get_snr(self):
        return self.prof.snr()

    def get_mjd(self):
        return self.subint.get_start_time().in_days()

    def get_dm(self):
        return self.archive.get_dispersion_measure()

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
    
    def get_metadata(self):
        return {
            "source_name": self.archive.get_source(),
            "receiver_name": self.archive.get_receiver_name(),
            "backend_name": self.archive.get_backend_name(),
            "backend_delay": self.archive.get_backend_delay(),
            "get_ant_xyz": self.archive.get_ant_xyz(),
            "nbin": self.archive.get_nbin(),
            "nchan": self.archive.get_nchan(),
            "npol": self.archive.get_npol(),
            "nsubint": self.archive.get_nsubint(),
            "first_subint": {
                "start_time": self.subint.get_start_time().in_days(),
                "end_time": self.subint.get_end_time().in_days(), 
                "duration": self.subint.get_duration(),
                "telescope": self.subint.get_telescope(),
                "ra_deg": self.subint.get_coordinates().ra().getDegrees(), 
                "dec_deg": self.subint.get_coordinates().dec().getDegrees(),
                "centre_frequency": self.subint.get_centre_frequency(),
                "bandwidth": self.subint.get_bandwidth(),
                "dispersion_measure": self.subint.get_dispersion_measure(),
                "rotation_measure": self.subint.get_rotation_measure(),
                "basis": self.subint.get_basis(),
                "state": self.subint.get_state(),
                "effective_dispersion_measure": self.subint.get_effective_dispersion_measure(),
                "effective_rotation_measure": self.subint.get_effective_rotation_measure(),
                "telescope_zenith": self.subint.get_telescope_zenith(),
                "telescope_azimuth": self.subint.get_telescope_azimuth(),
                "parallactic_angle": self.subint.get_parallactic_angle(),
                "position_angle": self.subint.get_position_angle(),
                "galactic_latitude": self.subint.get_galactic_latitude(),
                "galactic_longitude": self.subint.get_galactic_longitude(),
                "local_sidereal_time": self.subint.get_local_sidereal_time(),
                "doppler_factor": self.subint.get_doppler_factor(),
                "auxiliary_dispersion_corrected": self.subint.get_auxiliary_dispersion_corrected(),
                "auxiliary_birefringence_corrected": self.subint.get_auxiliary_birefringence_corrected(),
                "faraday_corrected": self.subint.get_faraday_corrected(),
            }
        }