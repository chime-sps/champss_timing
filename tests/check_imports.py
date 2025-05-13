########################################################################
# This script checks if all Python files in the                        #
# champss_timing directory can be imported without errors.             #
# It also checks the top-level scripts in the current directory.       #
# It prints a summary of successful and failed imports.                #
########################################################################

import os
import traceback
import importlib.util

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))  # Current dir
FAILURES = []

def try_import_file(py_path):
    rel_path = os.path.relpath(py_path, ROOT_DIR)
    module_name = rel_path.replace("/", ".").replace("\\", ".").rstrip(".py")

    try:
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ Imported: {rel_path}")
    except Exception as e:
        print(f"❌ Failed: {rel_path}")
        traceback.print_exc(limit=1)
        FAILURES.append((rel_path, str(e)))


def scan_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                full_path = os.path.join(root, file)
                try_import_file(full_path)


if __name__ == "__main__":
    print(f"🔍 Scanning for Python files in {ROOT_DIR}")
    scan_directory(os.path.join(ROOT_DIR, "backend"))

    # Also check top-level scripts
    # for fname in os.listdir(ROOT_DIR):
    #     if fname.endswith(".py") and not fname.startswith("__"):
    #         try_import_file(os.path.join(ROOT_DIR, fname))

    print("\n--- Summary ---")
    print(f"✅ Successful imports: {len(os.listdir(ROOT_DIR)) - len(FAILURES)}")
    print(f"❌ Failed imports: {len(FAILURES)}")
    if FAILURES:
        print("\nFailed files:")
        for path, err in FAILURES:
            print(f" - {path}: {err}")
