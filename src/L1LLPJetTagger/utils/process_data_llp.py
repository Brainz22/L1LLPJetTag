from L1LLPJetTagger.utils.config import config
import numpy as np


def combine_and_shuffle_data(
    dataset_sig, dataset_jet_data_sig, dataset_bkg, dataset_jet_data_bkg
):
    """
    Combines signal and background datasets, shuffles them, and creates labels.
    """
    # Stack signal and background datasets
    combined_data = np.concatenate((dataset_sig, dataset_bkg), axis=0)
    combined_jet_data = np.concatenate(
        (dataset_jet_data_sig, dataset_jet_data_bkg), axis=0
    )

    # randomize
    n = combined_data.shape[0]

    indices = np.arange(n)
    np.random.shuffle(indices)

    combined_data = combined_data[indices]
    combined_jet_data = combined_jet_data[indices]

    # create inputs and outputs
    X = combined_data[:, 0 : len(combined_data[0]) - 1]
    y = combined_data[:, -1]
    # back to input matrix shape
    X = X.reshape((X.shape[0], config.N_CAND, config.N_FEAT))
    # combined_jet_data = combined_jet_data.reshape( (combined_jet_data.shape[0], config.N_CAND, config.N_FEAT) )

    return X, y, combined_jet_data
