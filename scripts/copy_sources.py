import glob
import shutil
import os

src_dir = "timing_sources"
dest_dir = "champss_timing_sources"
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

for file in glob.glob(f"{src_dir}/*"):
    # Check if is directory
    if os.path.isdir(file):
        # Create directory in dest_dir
        new_dir = file.replace(src_dir, dest_dir)
        os.makedirs(new_dir, exist_ok=True)
        print(f"Copying {file} to {new_dir}")

        # Copy files in directory
        for f in glob.glob(f"{file}/*"):
            if f.split("/")[-1] in copy_files:
                # if directory, copy all files in directory
                if os.path.isdir(f):
                    # Create directory in dest_dir
                    sub_new_dir = f.replace(src_dir, dest_dir)
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
            shutil.copy(file, dest_dir)
            print(f"Copying {file} to {dest_dir}")