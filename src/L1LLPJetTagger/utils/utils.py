import numpy as np

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
