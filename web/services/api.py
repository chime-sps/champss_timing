import threading

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
            self.app.sources.cleanup()
            self.app.update()
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