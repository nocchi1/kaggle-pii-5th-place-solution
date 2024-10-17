import argparse
import json
import subprocess
from pathlib import Path

from slugify import slugify

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dir_path", required=True)
parser.add_argument("-t", "--title", required=True)
parser.add_argument("-r", "--dir_mode", required=False, default="tar")  # skip, tar, zip
parser.add_argument("--public", action="store_true")

args = parser.parse_args()
dir_path = args.dir_path
title = args.title
dir_mode = args.dir_mode
public = args.public
user_id = "ryotak12"

command = f"kaggle datasets init -p {dir_path}"
subprocess.run(command.split(), check=False)

metadata = json.load(open(Path(dir_path) / "dataset-metadata.json"))
metadata["title"] = title
metadata["id"] = user_id + "/" + slugify(title, replacements=[["&", "and"]])
metadata["licenses"] = [{"nameNullable": "unknown", "name": "unknown", "hasName": True}]
with open(Path(dir_path) / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

command = f"kaggle datasets create -p {dir_path} -r {dir_mode}"
if public:
    command += " --public"
subprocess.run(command.split(), check=False)
