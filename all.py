import os

batch_sizes = [32]
epochs_list = [100,200,300,500]
script_path = "/home/siat_dy/Learn/LearnPy/ROBIO2025/"
for batch in batch_sizes:
    for ep in epochs_list:
        cmd = f"python {script_path}fusion.py  --batch_size {batch} --epochs {ep}"
        os.system(cmd)
        cmd = f"python {script_path}eeg.py  --batch_size {batch} --epochs {ep}"
        os.system(cmd)
        # cmd = f"python {script_path}fnirs.py  --batch_size {batch} --epochs {ep}"
        print(f"Running: {cmd}")
        # os.system(cmd)