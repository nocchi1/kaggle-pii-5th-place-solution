import argparse
import subprocess
import zipfile
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--comp_name', required=True)
parser.add_argument('-p', '--dir_path', required=True)
parser.add_argument('-u', '--unzip', required=True, action='store_true')
parser.add_argument('-f', '--force', action='store_true', default=False)

args = parser.parse_args()
comp_name = args.comp_name
dir_path = args.dir_path
unzip = args.unzip
force = args.force

command = f'kaggle competitions download -c {comp_name} -p {dir_path}'
if force:
    command += ' -o'
subprocess.run(command.split())
print('DownLoad Complete!')

if args.unzip:
    with zipfile.ZipFile(Path(dir_path) / f'{comp_name}.zip') as f:
        f.extractall(Path(dir_path))
        print('Unzip Complete!')
    Path(dir_path).joinpath(f'{comp_name}.zip').unlink()