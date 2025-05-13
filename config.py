import os
import json
import argparse
from cli.config import CLIConfig

# Initialize parser
parser = argparse.ArgumentParser(description="CHAMPSS Timing Pipeline Configuration")
parser.add_argument(
    "-e"
    "--edit",
    action="store_true",
    dest="edit",
    help="Edit the configuration file. If the configuration file does not exist, it will be created.",
)
parser.add_argument(
    "-w"
    "--where",
    action="store_true",
    dest="where",
    help="Show the location of the configuration file.",
)
parser.add_argument(
    "-t",
    "--test",
    action="store_true",
    dest="test",
    help="Test the configuration file. If the configuration file does not exist, it will be created.",
)
parser.add_argument(
    "--reset",
    action="store_true",
    dest="reset",
    help="Reset the configuration file to default settings.",
)

# Parse arguments
args = parser.parse_args()

if args.edit:
    # Load configuration
    config = CLIConfig(load_error=False)
    # Prompt vim or nano to edit the configuration file
    editor = "vim" if os.system("which vim") == 0 else "nano"
    os.system(f"{editor} {config.config_file}")
elif args.where:
    # Load configuration
    config = CLIConfig()
    # Show the location of the configuration file
    print(f"Configuration file location: {config.config_file}")
elif args.test:
    # Load configuration
    try:
        config = CLIConfig(load_error=True)
        print("Loaded configuration:")
        print(json.dumps(config.get_config(), indent=4))
        print("Configuration file is valid.")
    except Exception as e:
        print(f"Configuration file is invalid: {e}.")
        print("Please edit the configuration file to fix the errors.")
        exit(1)
elif args.reset:
    response = input("Are you sure you want to reset the configuration file to default settings? (y/n): ")

    if response == "y":
        # Remove the existing configuration file. 
        if os.path.exists(config.config_file):
            os.remove(config.config_file)
            print(f"Configuration file {config.config_file} removed.")

        # It will be recreated with default settings
        config = CLIConfig()
        print(f"Configuration file {config.config_file} reset to default settings.")
    else:
        print("Configuration file reset cancelled.")
        exit(0)
else:
    # Show help message
    parser.print_help()
    exit(0)