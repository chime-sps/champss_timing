class login:
    def __init__(self, session, password):
        self.session = session
        self.password = password

    def checker(self, password):
        if self.password == False:
            return True

        if password == self.password:
            self.session['logged_in'] = True
            return True
        return False

    def has_logged_in(self):
        if self.password == False:
            return True

        return 'logged_in' in self.session