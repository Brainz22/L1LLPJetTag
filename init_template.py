#!/usr/bin/env python3
# use: python init_template.py --name "MyCoolProject" --author "Daniel Diaz"

import argparse
import os
import shutil


def replace_in_file(filepath, replacements):
    with open(filepath, "r") as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)


def walk_and_replace(base_dir, replacements):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith((".py", ".md", ".toml", ".txt")):
                replace_in_file(os.path.join(root, file), replacements)


def rename_project_folder(old_name, new_name):
    old_path = os.path.join("src", old_name)
    new_path = os.path.join("src", new_name)
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        print(f"Renamed {old_path} → {new_path}")
    else:
        print(f"WARNING: {old_path} does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize the project from the template"
    )
    parser.add_argument(
        "--name", required=True, help="New project name (e.g. MyProject)"
    )
    parser.add_argument(
        "--author", required=True, help="Author name (e.g. Daniel Diaz)"
    )
    args = parser.parse_args()

    replacements = {
        "SampleProject": args.name,
        "{{ author }}": args.author,
    }

    walk_and_replace(".", replacements)
    rename_project_folder("SampleProject", args.name)

    print(
        f"✅ Template initialized with project '{args.name}' and author '{args.author}'."
    )
