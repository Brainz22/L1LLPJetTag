import argparse
import os
from L1LLPJetTagger.utils.config import config
from L1LLPJetTagger.utils.loader import load_data
from L1LLPJetTagger.utils.process_data_llp import combine_and_shuffle_data


def main():
    sig_path = os.path.join(config.PROJECT_ROOT, "data/Sig_train.h5")
    bkg_path = os.path.join(config.PROJECT_ROOT, "data/DY_train.h5")
    sig_jet_path = os.path.join(config.PROJECT_ROOT, "data/Sig_jet_data.h5")
    bkg_jet_path = os.path.join(config.PROJECT_ROOT, "data/DY_jet_data.h5")
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
    parser.add_argument(
        "--signal_jet",
        type=str,
        default=sig_jet_path,
        help="Path to the signal jet data file",
    )
    parser.add_argument(
        "--background_jet",
        type=str,
        default=bkg_jet_path,
        help="Path to the background jet data file",
    )
    args = parser.parse_args()

    signal, bkg, sig_meta, bkg_meta = load_data(
        args.signal, args.background, args.signal_jet, args.background_jet
    )

    print("Signal data shape:", signal.shape)
    print("Signal jet data shape:", sig_meta.shape)
    print("Background data shape:", bkg.shape)
    print("Background jet data shape:", bkg_meta.shape)

    # Combine and shuffle data
    X, y, combined_jet_data = combine_and_shuffle_data(signal, sig_meta, bkg, bkg_meta)

    print("Combined data shape:", X.shape)
    print("Combined labels shape:", y.shape)
    print("Combined jet data shape:", combined_jet_data.shape)


if __name__ == "__main__":
    main()
