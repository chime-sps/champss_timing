import glob
import os
import shutil
import argparse
import sys
import time
import readline

from cli.config import CLIConfig
from cli.search import CLIInitialTimingSolutionSearch
from cli.masterdb import CLIMasterDBHandler
from cli.add import CLINewPulsar

TIMING_SOURCES_PATH = "./timing_sources"
TMG_MASTER_DB_PATH = "./timing_sources/TMGMaster.sqlite3.db"

##########################################
# Helper functions
##########################################

# Custom exception for setup abortion
class SetupAborted(Exception):
    pass

def ask_options(prompt, options):
    """
    Ask the user to choose from a list of options.

    prompt: The message to display to the user.
    options: A dictionary of options where keys are option identifiers and values are descriptions.
    """

    # Print option
    print(prompt)
    for key, description in options.items():
        print(f"  {key}: {description}")

    # Ask for user choice
    while True:
        choice = input("Enter your choice: ").strip().lower()
        if choice in options:
            return choice
        print("Invalid choice. Please try again.")


def clean_path(raw):
    """Helper function to clean and expand a file path."""
    return os.path.expanduser(raw.strip().strip("'\""))


def prompt_until_valid(function, prompt):
    """Prompt the user until the provided function returns a truthy value."""
    while True:
        value = input(prompt).strip()
        if function(value):
            return value
        print("Please try again.")


def retry_until_success(function, *args, description="This step"):
    """Retry the given function until it succeeds or the user aborts."""
    while not function(*args):
        choice = ask_options(
            f"{description} failed. What would you like to do?",
            {
                "retry": "Try again",
                "abort": "Abort the setup",
            },
        )
        if choice == "abort":
            raise SetupAborted(f"{description} failed.")


def ask_archive_files():
    """Ask for a glob pattern until it matches at least one file."""
    while True:
        pattern = clean_path(
            input("Enter the path to the archive files with glob patterns: ")
        )
        files = sorted(
            f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)
        )
        if files:
            print(f"Found {len(files)} archive file(s).")
            return files
        print(f"No files matched '{pattern}'. Please try again.")

def wait_and_confirm(prompt, time_in_seconds=3):
    """Wait for a specified time and then ask the user to confirm."""
    while time_in_seconds > 0:
        print(f"{prompt} (Continuing in {time_in_seconds} seconds...)", end="\r")
        time.sleep(1)
        time_in_seconds -= 1
        
    input(f"{prompt} (Press Enter to continue...)")

def complete_path(text, state):
    line = readline.get_line_buffer()[:readline.get_endidx()]
    expanded = os.path.expanduser(line)
    matches = glob.glob(expanded + '*')
    matches = [m + os.sep if os.path.isdir(m) else m for m in matches]
    cut = len(expanded) - len(text)            # the part readline won't replace
    matches = [m[cut:] for m in matches]
    return matches[state] if state < len(matches) else None

readline.set_completer_delims(' \t\n;')
if readline.__doc__ and 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")
readline.set_completer(complete_path)


##########################################
# Setup steps
##########################################

def run_phase_coherent_search(new_pulsar):
    """Run the search (repeatably) and use its results for the par and std files."""
    psrdir = new_pulsar.get_psrdir()

    while True:
        # Get input archive files
        archive_files = ask_archive_files()

        # Make sure at least one archive file was found
        if len(archive_files) == 0:
            print("No archive files found. Please try again.")
            continue

        # Run the phase coherent search
        print("Running phase coherent search...")
        cli_pcs = CLIInitialTimingSolutionSearch(archive_files=archive_files)
        cli_pcs.optimize(
            output_dir=psrdir,
            make_directory=False,
            preview=True,
        )

        # Ask user to inspect the result
        choice = ask_options(
            "Is the phase coherent search result satisfactory?",
            {
                "yes": "Use this result",
                "rerun": "Run the search again with different archive files",
                "abort": "Abort the setup",
            },
        )
        if choice == "yes":
            break
        if choice == "abort":
            raise SetupAborted("Phase coherent search result rejected.")

    # Use phase coherent search results to set the initial par and std files
    search_par = os.path.join(psrdir, "search.par")
    if not new_pulsar.add_psr_parfile(search_par):
        raise FileNotFoundError(
            f"Phase coherent search did not produce the initial pulsar parfile ({search_par})."
        )

    # Ask user to fit a paas template
    while True:
        input("Press Enter to create the pulsar standard profile template from the stacked profile...")
        retry_until_success(
            new_pulsar.create_psr_stdfile,
            os.path.join(psrdir, "search.ar"),
            description="Creating the standard profile template",
        )
        choice = ask_options(
            "Is the standard profile just created satisfactory?",
            {
                "yes": "Use this template",
                "rerun": "Create the template again",
                "abort": "Abort the setup",
            },
        )
        if choice == "yes":
            break
        if choice == "abort":
            raise SetupAborted("Standard profile template creation aborted.")


def setup_manually(new_pulsar):
    input("Press Enter to create the pulsar parfile...")
    retry_until_success(
        new_pulsar.create_psr_parfile,
        description="Creating the pulsar parfile",
    )

    prompt_until_valid(
        lambda path: new_pulsar.add_psr_stdfile(clean_path(path)),
        "Enter the path to the pulsar standard profile template: ",
    )


def configure(new_pulsar):
    config_option = ask_options(
        "How would you like to configure the new pulsar?",
        {
            "default": "No configuration (use default settings)",
            "existing": "Use a configuration from an existing pulsar",
            "new": "Create a new configuration",
        },
    )
    if config_option == "existing":
        prompt_until_valid(
            new_pulsar.load_from_existing_pulsar,
            "Enter the name of the existing pulsar to use its configuration: ",
        )
    elif config_option == "new":
        print("Preparing to create a new configuration for the pulsar...")
        retry_until_success(
            new_pulsar.create_psr_config,
            description="Creating the pulsar configuration",
        )
    else:
        print("Using default configuration for the pulsar.") # No need to setup anything


def setup(new_pulsar):
    # prompt_until_valid(new_pulsar.add_psr_name, "Enter the pulsar name: ")

    # Create the initial parfile and standard profile template
    setup_option = ask_options(
        "How would you like to set up the initial parfile (pulsar.par) "
        "and standard profile template (paas.std)?",
        {
            "pcs": "Phase coherent search",
            "manual": "Manual input",
        },
    )
    if setup_option == "manual":
        setup_manually(new_pulsar)
    else:
        run_phase_coherent_search(new_pulsar)

    # Configure the new pulsar
    configure(new_pulsar)

    # Commit the new pulsar setup
    print("Committing the new pulsar setup...")
    new_pulsar.commit()
    print("New pulsar setup committed successfully.")

def finalize():
    """Finalize the setup of the new pulsar."""
    finalize_option = ask_options(
        "How would you like to finalize the setup of the new pulsar?",
        {
            "time": "Insert raw data to TMGMaster and time the pulsar",
            "insert": "Insert raw data to TMGMaster only",
            "skip": "Skip finalization",
        },
    )

    # Insert raw data
    if finalize_option in ["time", "insert"]:
        CLIMasterDBHandler(
            db_path=TMG_MASTER_DB_PATH,
            backends=cli_config.config["backends"], 
            fast_mode_mem_gb=1
        ).insert_data(
            placeholder_if_corrupted=False, 
            psr=args.psr
        )

    # Start timing the pulsar if requested
    if finalize_option == "time":
        # Just import the pipelin script... this is not elegant, but as long as it works... 
        import pipeline

def cleanup(new_pulsar):
    """Clean up the partially created pulsar directory if the setup did not finish."""
    # Check if the pulsar directory exists
    try:
        psrdir = new_pulsar.get_psrdir()
    except Exception:
        return  

    # If the pulsar directory does not exist, nothing to clean up
    if not psrdir or not os.path.isdir(psrdir):
        return

    # Ensure the target directory is within the timing sources path before removing
    root = os.path.realpath(TIMING_SOURCES_PATH)
    target = os.path.realpath(psrdir)
    if target == root or os.path.commonpath([root, target]) != root:
        print(f"Partial setup left in {psrdir} (outside {TIMING_SOURCES_PATH}; not removing).")
        return

    # Ask the user whether to remove the partially created directory
    try:
        choice = ask_options(
            f"The setup did not finish. Remove the partially created directory {psrdir}?",
            {
                "yes": "Remove it",
                "no": "Keep it for debugging",
            },
        )
    except (KeyboardInterrupt, EOFError):
        print(f"\nKeeping {psrdir}.")
        return

    # Clean up the partially created directory based on user choice
    if choice == "yes":
        wait_and_confirm(f"About to remove the partially created directory {psrdir}")
        shutil.rmtree(target)
        print(f"Removed {psrdir}.")
    else:
        print(f"Kept {psrdir}.")



##########################################
# Main loop
##########################################

# Initialize config
cli_config = CLIConfig()

# Initialize parser
parser = argparse.ArgumentParser(description="Add a new pulsar to the timing sources.")
parser.add_argument("--psr", help="The name of the new pulsar.")
args = parser.parse_args()

# Initializer a new pulsar handler
new_pulsar = CLINewPulsar(TIMING_SOURCES_PATH)

# Check if new pulsar name is valid. 
if not new_pulsar.add_psr_name(args.psr):
    print(f"Pulsar name {args.psr} is not valid. Please try a different name. ")
    exit(1)

try:
    print(f"Setting up timing pipeline for {args.psr}")
    setup(new_pulsar)
except KeyboardInterrupt:
    print("\nSetup interrupted.")
    cleanup(new_pulsar)
    sys.exit(130)
except SetupAborted as e:
    print(f"Setup aborted: {e}")
    cleanup(new_pulsar)
    sys.exit(1)
except Exception:
    print("Setup failed with an error.")
    cleanup(new_pulsar)
    raise

try:
    finalize()
except Exception as e:
    print(f"Finalization failed with an error: {e}")
    raise