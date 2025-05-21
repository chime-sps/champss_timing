import os
import sys
import argparse
import pkgutil
import importlib
from backend.utils.utils import utils

def main(module_avail):
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        # No module given or help asked before subcommand
        parser = argparse.ArgumentParser(
            description=f"CHAMPSS Timing Pipeline ({utils.get_version_hash()})"
        )
        subparsers = parser.add_subparsers(dest="module", required=True)
        for module_name, module_desc in module_avail.items():
            subparsers.add_parser(module_name, help=module_desc)
        parser.print_help()
        sys.exit()

    selected_module = sys.argv[1]
    if selected_module not in module_avail:
        print(f"Unknown module: {selected_module}")
        print("Available modules:")
        for name, desc in module_avail.items():
            print(f"  {name:<12} {desc}")
        sys.exit(1)

    # Reconstruct argv for the target module
    sys.argv = [selected_module] + sys.argv[2:]
    importlib.import_module(selected_module)

if __name__ == "__main__":
    module_avail = {
        "pipeline": "Run the main pipeline.",
        "server": "Run the pipeline web server.",
        "dealias": "Resolve dealiasing in pulsar period solution.",
        "template": "Generate a template for the pulsar.",
        "truncate": "Truncate source database.",
        "masterdb": "Run master database utilities.",
        "config": "Show or edit the configuration file.", 
        "misc": "Run user-defined miscellaneous scripts."
    }
    main(module_avail)
