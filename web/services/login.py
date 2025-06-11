import importlib

class login:
    def __init__(self, session, authenticator):
        self.session = session
        self.auth_hdl = self.load_authenticator(authenticator)

    def load_authenticator(self, authenticator):
        """
        Load the authenticator module based on the provided name.
        """
        try:
            module = importlib.import_module(f"..authenticators.{authenticator}", package=__package__).Main()
        except ImportError as e:
            raise ImportError(f"Authenticator '{authenticator}' not found: {e}")
        
        # Make sure the authenticator has all required attributes
        if not hasattr(module, 'AUTH_FIELDS'):
            raise AttributeError(f"Authenticator '{authenticator}' does not have an AUTH_FIELDS attribute.")
        if not hasattr(module, 'NAME'):
            raise AttributeError(f"Authenticator '{authenticator}' does not have an HINTS attribute.")
        if not hasattr(module, 'authenticate'):
            raise AttributeError(f"Authenticator '{authenticator}' does not have an authenticate method.")
        
        return module


    def is_needed(self):
        """
        Check if authentication is needed.
        Returns
        -------
        bool
            True if authentication is needed, False otherwise.
        """
        return self.auth_hdl.AUTH_FIELDS != []

    def checker(self, **kwargs):
        """
        Authenticate the user with the provided credentials.
        Parameters
        ----------
        kwargs : dict
            The keyword arguments containing the authentication fields.
        Returns
        -------
        bool
            True if authentication is successful, False otherwise.
        """

        if not self.auth_hdl.authenticate(**kwargs):
            return False

        self.session['logged_in'] = True
        return True

    def has_logged_in(self):
        if not self.is_needed():
            return True

        return 'logged_in' in self.session
    
    def get_auth_name(self):
        """
        Get the title of the authenticator.
        Returns
        -------
        str
            The title of the authenticator.
        """
        return self.auth_hdl.NAME
    
    def get_fields(self):
        """
        Get the authentication fields required by the authenticator.
        Returns
        -------
        list
            A list of authentication fields.
        """
        return self.auth_hdl.AUTH_FIELDS