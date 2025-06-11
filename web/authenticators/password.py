import os

class Main:
    """
    Example authenticator for the CHAMPSS Timing web application.
    This class handles user authentication with a password stored in an environment variable.
    """
    
    NAME = "CHAMPSS Timing Password"
    AUTH_FIELDS = [
        {
            "name": "Password",
            "id": "password",
            "type": "hidden"
        }
    ]

    def __init__(self):
        """
        Initialize the authenticator.
        This method can be extended to include additional initialization logic if needed.
        """
        
        # Get password from environment variable
        self.password = os.getenv("CHAMPSS_TIMING_WEB_PASSWORD")

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
        
        # Check if the password matches the environment variable
        return kwargs.get("password") == self.password if self.password else False