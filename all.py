import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, batch_size, epochs, extra_args):
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
    ]
    cmd.extend(extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train fusion/eeg/fnirs pipelines in sequence.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--epochs_list", type=int, nargs="+", default=[300, 100, 200, 500])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir_eeg", type=str, default="PPEEG")
    parser.add_argument("--data_dir_fnirs", type=str, default="PPfNIRS")
    parser.add_argument("--data_dir_fusion", type=str, default="FusionEEG-fNIRS")
    parser.add_argument("--files_limit", type=int, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    print(f"Script directory: {script_dir}")
    train_order = [
        ("fusion.py", args.data_dir_fusion),
        ("eeg.py", args.data_dir_eeg),
        ("fnirs.py", args.data_dir_fnirs),
    ]

    for batch in args.batch_sizes:
        for ep in args.epochs_list:
            for script_name, data_dir in train_order:
                script_path = script_dir / script_name
                extra_args = ["--device", args.device, "--data_dir", data_dir]
                if args.files_limit is not None:
                    extra_args.extend(["--files_limit", str(args.files_limit)])
                run_script(script_path, batch, ep, extra_args)


if __name__ == "__main__":
    main()