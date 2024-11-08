class api:
    def __init__(self, app):
        self.app = app

    def update_psrdir(self):
        if self.app.update != None:
            self.app.update()
            return "Update started."

        return "Update handler is not set."

    def handle(self, endpoint, request):
        if endpoint == "update_psrdir":
            return self.update_psrdir()
        return "Invalid endpoint."