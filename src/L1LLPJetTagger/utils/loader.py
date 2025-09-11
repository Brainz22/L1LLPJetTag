import numpy as np
from L1LLPJetTagger.utils.utils import load_h5_data


def load_data(
    signal_file: str,
    background_file: str,
    signal_jet_file: str,
    background_jet_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load signal and background data and jet files from HDF5 files.
    """

    signal = load_h5_data(signal_file, "jet_constituents")
    background = load_h5_data(background_file, "jet_constituents")
    signal_jets = load_h5_data(signal_jet_file, "jet_data")
    background_jets = load_h5_data(background_jet_file, "jet_data")

    return signal, background, signal_jets, background_jets
