import pam

class Main:
    """
    Example authenticator for the CHAMPSS Timing web application.
    This class handles user authentication with a password stored in an environment variable.
    """
    
    NAME = "CHIME Account"
    AUTH_FIELDS = [
        {
            "name": "Username",
            "id": "username",
            "type": "default"
        }, 
        {
            "name": "Password",
            "id": "password",
            "type": "hidden"
        }
    ]

    def __init__(self):
        """
        Initialize the authenticator.
        """
        
        # Initialize the PAM module for authentication
        self.pam = pam.pam()

    def authenticate(self, **kwargs):
        """
        Authenticate the user based on the provided credentials.
        Parameters
        ----------
        kwargs : dict
            The keyword arguments containing the authentication fields.
        Returns
        -------
        bool
            True if authentication is successful, False otherwise.
        """
        
        return self.pam.authenticate(
            kwargs.get("username"), 
            kwargs.get("password")
        )