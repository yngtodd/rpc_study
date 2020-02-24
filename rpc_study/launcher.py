import argparse
import subprocess
from pathlib import Path, PurePath


def main():
    parser = argparse.ArgumentParser(description='Study launcher')
    parser.add_argument('path', metavar='FILENAME', help='path to study') 
    args = parser.parse_args()

    launch_path = Path(__file__).resolve().parent
    run_script = PurePath(launch_path).joinpath('run.sh')

    # Use Pytorch's distributed launch
    subprocess.Popen(["/bin/bash", run_script, args.path])


if __name__=='__main__':
    main()
