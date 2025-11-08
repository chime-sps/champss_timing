import glob
import shutil
import os
import argparse

copy_files = [
    "champss_timing.sqlite3.db", 
    "champss_diagnostic.pdf", 
    "pulsar.par", 
    "champss_timing.log", 
    "timing_summary.txt", 
    "dealias_info.ecsv", 
    "diagnostic.png", 
    "pulsar.dealias.pdf", 
    "pulsar.dealiased.par", 
    "dealias", 
    "TMGMaster.sqlite3.db"
]

parser = argparse.ArgumentParser(description="Copy necessary files to web server.")
parser.add_argument("--psr", type=str, help="Pulsar name.", default="*")
parser.add_argument("--src", type=str, help="Source directory.", default="timing_sources")
parser.add_argument("--dest", type=str, help="Destination directory.", default="champss_timing_sources")
args = parser.parse_args()

for file in glob.glob(f"{args.src}/{args.psr}"):
    # Check if is directory
    if os.path.isdir(file):
        # Create directory in args.dest
        new_dir = file.replace(args.src, args.dest)
        os.makedirs(new_dir, exist_ok=True)
        print(f"Copying {file} to {new_dir}")

        # Copy files in directory
        for f in glob.glob(f"{file}/*"):
            if f.split("/")[-1] in copy_files:
                # if directory, copy all files in directory
                if os.path.isdir(f):
                    # Create directory in args.dest
                    sub_new_dir = f.replace(args.src, args.dest)
                    os.makedirs(sub_new_dir, exist_ok=True)
                    print(f" Copying {f} to {sub_new_dir}")

                    # Copy files in directory
                    for ff in glob.glob(f"{f}/*"):
                        if ff.split("/")[-1] in copy_files:
                            shutil.copy(ff, sub_new_dir)
                            print(f"  Copying {ff} to {sub_new_dir}")
                else:
                    shutil.copy(f, new_dir)
                    print("", f"Copying {f} to {new_dir}")
    else:
        if file.split("/")[-1] in copy_files:
            shutil.copy(file, args.dest)
            print(f"Copying {file} to {args.dest}")