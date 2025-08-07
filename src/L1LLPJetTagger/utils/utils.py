import h5py
import numpy as np
import awkward as ak

known_ids = [11, -11, 13, -13, 22, 130, 211, -211]


def one_hot_encode_pdgid(pdg_ids: np.ndarray) -> np.ndarray:
    """
    One-hot encodes an array of PDG IDs.

    Args:
        pdg_ids (np.ndarray): Array of PDG ID integers.
        known_ids (list[int]): List of PDG IDs to encode as one-hot.

    Returns:
        np.ndarray: One-hot encoded array of shape (len(pdg_ids), len(known_ids))
    """
    mapping = {pid: i for i, pid in enumerate(known_ids)}
    one_hot = np.zeros((len(pdg_ids), len(known_ids)), dtype=np.float32)

    for i, pid in enumerate(pdg_ids):
        if pid in mapping:
            one_hot[i, mapping[pid]] = 1.0
    return one_hot


def pad_and_fill(array, target_len, fill_value=0.0):
    """
    Pads each sublist in `array` to `target_len` and fills None with `fill_value`.

    Parameters:
    - array: awkward array
    - target_len: int, number of elements to pad to
    - fill_value: value to replace None with (default: 0.0)

    Returns:
    - processed awkward array
    """
    return ak.fill_none(ak.pad_none(array, target_len, clip=True), fill_value)


def load_h5_data(file_path, dataset_name):
    with h5py.File(file_path, "r") as f:
        data = f[dataset_name][:]
    return data
