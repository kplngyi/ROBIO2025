# %% [markdown]
# Fusion Training Notebook Launcher
#
# Run this file in VSCode's Jupyter mode. Adjust the parameters below, then
# execute the cells in order.

# %%
from pathlib import Path
import os
import shlex
import subprocess
import sys


# %%
def detect_project_root():
    candidates = [
        Path.cwd(),
        Path("/content/ROBIO2025"),
        Path("/Users/hpyi/SAYGB/26Codes/ROBIO2025"),
    ]
    for candidate in candidates:
        if (candidate / "fusion.py").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = detect_project_root()
OUTPUT_ROOT = (
    Path("/content/drive/MyDrive/ROBIO2025_runs").expanduser()
    if PROJECT_ROOT.as_posix().startswith("/content/")
    else PROJECT_ROOT
)
DATA_DIR = (
    Path("/content/drive/MyDrive/ROBIO2025_data/FusionEEG-fNIRS").expanduser()
    if PROJECT_ROOT.as_posix().startswith("/content/")
    else PROJECT_ROOT / "FusionEEG-fNIRS"
)
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

BATCH_SIZE = 32
EPOCHS = 30
TOP_K_STEP = 2
MIN_TOP_K = None
FILES_LIMIT = None


# %%
def build_command():
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "fusion.py"),
        "--project_root",
        str(PROJECT_ROOT),
        "--output_root",
        str(OUTPUT_ROOT),
        "--data_dir",
        str(DATA_DIR),
        "--config_path",
        str(CONFIG_PATH),
        "--batch_size",
        str(BATCH_SIZE),
        "--epochs",
        str(EPOCHS),
        "--top_k_step",
        str(TOP_K_STEP),
    ]
    if MIN_TOP_K is not None:
        cmd.extend(["--min_top_k", str(MIN_TOP_K)])
    if FILES_LIMIT is not None:
        cmd.extend(["--files_limit", str(FILES_LIMIT)])
    return cmd


def print_command(cmd):
    print("Command:")
    print(" ".join(shlex.quote(part) for part in cmd))


# %%
command = build_command()
print_command(command)


# %%
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.setdefault("XDG_CACHE_HOME", str(OUTPUT_ROOT / ".cache"))
env.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".mplconfig"))
subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
