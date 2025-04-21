# Import all the modules in the package
from .backend import *

# Shortcuts for modules that are frequently been used (backwards compatible for some old scripts)
try:
    from .backend.datastores.database import database
    from .backend.datastores.tmg_master import tmg_master
except ImportError:
    print("Database utilities (.backend.datastores) are disabled. Some required modules may not be available.")

try:
    from .backend.utils.notification import notification
except ImportError:
    print("Notification utilities (.backend.utils.notification) are disabled. Some required modules may not be available (could be slackbolt?).")

try:
    from .backend.pipecore.plot import plot
except ImportError:
    print("Plot utilities (.backend.pipecore.plot) are disabled. Some required modules may not be available (could be matplotlib, numpy?).")

try:
    from .backend.utils.utils import utils
except ImportError:
    print("Pipeline utility module (.backend.utils.utils) is disabled. Some required modules may not be available.")

try:
    from .backend.utils.logger import logger
except ImportError:
    print("Logging utilities (.backend.utils.logger) is disabled. Some required modules may not be available.")



