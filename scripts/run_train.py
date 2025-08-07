import argparse
import os
from L1LLPJetTagger.utils.config import config
from L1LLPJetTagger.utils.utils import load_h5_data


def main():
    sig_path = os.path.join(config.PROJECT_ROOT, "data/Sig_train.h5")
    bkg_path = os.path.join(config.PROJECT_ROOT, "data/DY_train.h5")
    parser = argparse.ArgumentParser(description="Load HDF5 data")

    parser.add_argument(
        "--signal", type=str, default=sig_path, help="Path to the signal data file"
    )
    parser.add_argument(
        "--background",
        type=str,
        default=bkg_path,
        help="Path to the background data file",
    )
    args = parser.parse_args()

    signal_data = load_h5_data(args.signal, "jet_constituents")
    background_data = load_h5_data(args.background, "jet_constituents")

    print("Signal data shape:", signal_data.shape)
    print("Background data shape:", background_data.shape)


if __name__ == "__main__":
    main()
