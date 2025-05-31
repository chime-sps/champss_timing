import threading
import shutil
import copy
import time

class PublicAPI:
    def __init__(self, app):
        self.app = app

    def update_psrdir(self):
        if self.app.update != None:
            # for source in self.app.sources:
            #     source.cleanup()
            # self.app.update()
            # for source in self.app.sources:
            #     source.initialize()
            # self.app.sources.get_heatmap()

            # Get the new directory into a new psrdir
            psr_dir_temp = self.app.sources.psr_dir + "_temp" # Avoid conflicts with existing psr_dir
            self.app.update(dir=psr_dir_temp)

            # Clean up the existing source handlers
            self.app.sources.cleanup()

            # Remove the existing psr_dir and replace it with the new one
            if shutil.os.path.exists(self.app.sources.psr_dir):
                shutil.rmtree(self.app.sources.psr_dir)
            shutil.move(psr_dir_temp, self.app.sources.psr_dir)

            # Reinitialize the sources
            self.app.sources.initialize()

            return "Updated."

        return "Update handler is not set."

    def handle(self, endpoint, request):
        if endpoint == "update_psrdir":
            return self.update_psrdir()
        if endpoint == "heartbeat":
            return "ok"
        return "Invalid endpoint."


class PrivateAPI:
    def __init__(self, app):
        self.app = app

    def get_notes_by_psr(self, psr, group_by_date):
        notes = self.app.notes.fetch(tag=psr, group_by_date=group_by_date)
        return notes

    def handle(self, endpoint, request):
        if endpoint == "get_notes_by_psr":
            psr = request.args.get("psr")
            group_by_date = request.args.get("group_by_date", "true").lower() == "true"
            if psr is None:
                return "PSR not provided."
            return self.get_notes_by_psr(psr, group_by_date)
        return "Invalid endpoint."