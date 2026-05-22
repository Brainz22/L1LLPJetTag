"""
Sample command
python scripts/run_dataForge.py -i jetTuple_extended_5.root -s sample=HiddenGluGluH_mH125_Phi15_ctau1000_cccc_PU200 -d Test -t outnano/Jets -o /uscms/home/ddiaz/nobackup/L1LLPJetTagger/data/
"""

from L1LLPJetTagger.dataForge import forge_h5
from L1LLPJetTagger.utils.config import config

import argparse
from pathlib import Path


def parse_args():
    prj = config.PROJECT_ROOT
    parser = argparse.ArgumentParser(prog="run_dataForge")
    parser.add_argument(
        "-i", "--input", type=Path, default=prj / "DY_1k.root", help="Input ROOT file"
    )
    parser.add_argument(
        "-s", "--sample", type=Path, help="Sample type folder where files are, e.g. HiddenGluGluH_mH125_Phi15_ctau1000_cccc_PU200."
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=prj / "data/", help="Output directory"
    )
    parser.add_argument(
        "-t", "--tree", type=str, default="outnano/Jets", help="Tree name in ROOT file"
    )
    parser.add_argument(
        "-d", "--data_type", type=str, default="DY", help="Data type: BkgProc or Sig"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input ROOT file {args.input} does not exist.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"\nRunning dataForge on: {args.input}/{args.sample}, outputting to: {args.out}, tree: {args.tree}, data_type: {args.data_type}"
    )
    events = forge_h5(
        root_path=str(args.input),
        sample_flavor=str(args.sample),
        out_path=str(args.out),
        tree_name=str(args.tree),
        data_type=str(args.data_type),
    )
    return events


if __name__ == "__main__":
    main()
