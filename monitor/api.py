import threading

class api:
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
        return "Invalid endpoint."