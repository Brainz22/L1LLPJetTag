from L1LLPJetTagger import delta_phi
import numpy as np
from L1LLPJetTagger.utils.utils import one_hot_encode_pdgid
from L1LLPJetTagger.utils.utils import pad_and_fill
from L1LLPJetTagger.utils.utils import known_ids
from L1LLPJetTagger.utils.config import config
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import awkward as ak
import h5py
import os, glob, sys
from pathlib import Path

DEBUG = False


def debug_print(events, num_jets=5):

    print("\nLoaded fields:")
    for field in events.fields:
        print(f" - {field}: {ak.type(events[field])}")

    print("\nFirst {} jets:".format(num_jets))
    for i in range(min(num_jets, len(events["jet_pt"]))):
        pt = ak.to_list(events["jet_pt"][i])
        eta = ak.to_list(events["jet_eta"][i])
        phi = ak.to_list(events["jet_phi"][i])
        pid = ak.to_list(events["jet_pfcand_id"][i])
        cand_pt = ak.to_list(events["jet_pfcand_pt_phys"][i])
        cand_eta = ak.to_list(events["jet_pfcand_eta_phys"][i])
        cand_phi = ak.to_list(events["jet_pfcand_phi_phys"][i])
        vx = ak.to_list(events["jet_pfcand_track_vx"][i])
        vy = ak.to_list(events["jet_pfcand_track_vy"][i])
        vz = ak.to_list(events["jet_pfcand_track_vz"][i])
        dxy = ak.to_list(events["jet_pfcand_dxy"][i])

        print(f"\nJet {i}:")
        print(f"  pt:  {pt:.2f}")
        print(f"  eta: {eta:.2f}")
        print(f"  phi: {phi:.2f}")
        print(f"  nPFCand: {len(vx)}")
        print(f"  pfCand_ID: {pid[:]}")
        print(f"  pfCand_pt: {cand_pt[:]}")
        print(f"  pfCand_eta: {cand_eta[:]}")
        print(f"  pfCand_phi: {cand_phi[:]}")
        print(f"  track_vx: {vx[:]}")
        print(f"  track_vy: {vy[:]}")
        print(f"  track_vz: {vz[:]}")
        print(f"  track_dxy: {dxy[:]}")

    print("\nConfig paths:")
    print(f"  pkgdir: {config.PKG_ROOT}")
    print(f"  projdir: {config.PROJECT_ROOT}")


def forge_h5(
    root_path: str,
    sample_flavor: str,
    out_path: str,
    tree_name: str,
    data_type: str = "UNKNOWN",
    num_constituents: int = 10,
    train_split: float = 0.6,
):
    """
    Processes a ROOT file containing jet and constituent information and saves the output in HDF5 format.

    This function loads jet-level constituent data from a ROOT TTree, pads each jet
    to a fixed number of constituents, and stores the processed data as a flat array in an HDF5 file.

    Args:
        root_path (str): Path to the input ROOT file.
        sample_flavor (str): Folder name where the sample roots files exist.
        out_path (str): Path to the output HDF5 file.
        tree_name (str): Name of the TTree in the ROOT file to process.

    Returns:
        ak.Array: Awkward Array of events loaded from the ROOT file using BaseSchema.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Assign label: 1 for signal, 0 for background
    if data_type.lower().startswith("sig"):
        label = 1
    else:
        label = 0

    fpaths = glob.glob(os.path.join(root_path, sample_flavor, "**/*.root"), recursive=True)
    print(f"========================================================")
    print(f"Check first path: {fpaths[0]}")
    print(f"\nFound {len(fpaths)} files")
    print(f"=========================================================")

    #sys.exit(0)
    # Load NanoEvents using BaseSchema for custom structure
    # events = NanoEventsFactory.from_root(
    #     f"{root_path}:{tree_name}", schemaclass=BaseSchema
    # ).events()

    print("Beginning tree reading *******************")

    events = ak.concatenate([NanoEventsFactory.from_root(
                {f: tree_name},
                #entry_stop=1000,
                schemaclass=BaseSchema
            ).events()
        for f in fpaths[:]
    ])

    # Debug print to show first N jets
    if DEBUG:
        debug_print(events, 2)

    # Pad to fixed length num_constituents per jet using 0 (NJets, NConstituents)
    pid = pad_and_fill(events["jet_pfcand_id"], num_constituents)
    pt = pad_and_fill(events["jet_pfcand_pt_phys"], num_constituents)
    eta = pad_and_fill(events["jet_pfcand_eta_phys"], num_constituents)
    phi = pad_and_fill(events["jet_pfcand_phi_phys"], num_constituents)
    vx = pad_and_fill(events["jet_pfcand_track_vx"], num_constituents)
    vy = pad_and_fill(events["jet_pfcand_track_vy"], num_constituents)
    vz = pad_and_fill(events["jet_pfcand_track_vz"], num_constituents)

    # keep track of dummy entries
    mask = np.abs(pt) > 1e-6

    # Normalize to jet-level quantities
    rel_pt = pt / events["jet_pt_phys"][..., None]
    rel_eta = eta - events["jet_eta_phys"][..., None]
    rel_phi = delta_phi(phi, events["jet_phi_phys"][..., None])
    rel_eta = ak.where(mask, rel_eta, 0.0)
    rel_phi = ak.where(mask, rel_phi, 0.0)

    # massage the pid array to be flat and do the one-hot encoding
    flat_pid = ak.to_numpy(pid).reshape(-1)
    one_hot = one_hot_encode_pdgid(flat_pid)
    one_hot = one_hot.reshape(len(pid), num_constituents, len(known_ids))

    # shape: (n_jets, n_cands,n_features)
    constituents = ak.concatenate(
        [
            one_hot,
            vz[..., None],
            vx[..., None],
            vy[..., None],
            rel_pt[..., None],
            rel_eta[..., None],
            rel_phi[..., None]
        ],
        axis=-1,
    )
    # Reshape to (n_jets, n_cands * n_features)
    flat = ak.to_numpy(constituents).reshape(len(vx), -1)

    # Append label as the last column
    labels = np.full((flat.shape[0], 1), label, dtype=np.float32)
    flat = np.concatenate((flat, labels), axis=1)
    jetData = ak.concatenate((events["jet_pt_phys"][...,None], events["jet_eta_phys"][..., None], \
                events["jet_phi_phys"][..., None], events["jet_mass"][..., None] ), axis=-1)
    jetData = ak.to_numpy(jetData)

    #select only jets with LLP_daughter matched to an sc jet
    if data_type.lower().startswith("sig"):
        print("======================================")
        print("Removing background jets from signal sample and applying eta cut (< abs(2.4))" )
        print("Initial shape: ", flat.shape)
        mask = (events["jet_doesmatch_genLLPDecay"][...]) & (np.abs(events["jet_eta_phys"][...]) < 2.4)
        flat = flat[mask]
        jetData = jetData[mask]
        print("\n********* Removing done ********* \n")
        print("Final shape: ", flat.shape)
        print("======================================")
    else: 
        print("===================================")
        print("Working with background sample. Only applying eta cut (< abs(2.4))")
        mask = np.abs(events["jet_eta_phys"][...]) < 2.4
        flat = flat[mask]
        jetData = jetData[mask]
        print("\n********* Removing done ********* \n")
        print("Final shape: ", flat.shape)
        print("===================================")
    

    # split dataset into a training and a testing set randomly
    # Shuffle the data for randomness
    n_total = flat.shape[0]
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Split into training and testing sets
    train_size = int(train_split * n_total)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = flat[train_indices]
    train_jetData = jetData[train_indices]

    test_data = flat[test_indices]
    test_jetData = jetData[test_indices]

    print(f"\nFinal array shape before splitting: {flat.shape}")
    print(f"Final array shape for test data: {test_data.shape}")
    print(f"Final array shape for train data: {train_data.shape}")
    print(f"Final array shape for jet-level train data: {train_jetData.shape}")
    print(f"Final array shape for jet-level test data: {test_jetData.shape}")
    print(f"Total jets: {flat.shape[0]}")
    
    ## append final number of jets to file
    log_path = Path(__file__).parent.parent.parent / "data" / "forge_log.txt"
    with open(log_path, "a") as f:
        f.write(f"{sample_flavor} Total jets: {flat.shape[0]} \n")
        print("Logged.\n")


    created_files = []
    # Save to HDF5
    out_path_h5 = os.path.join(out_path, data_type + "_jet_data.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_data", data=flat)
    f.close()
    created_files.append(out_path_h5)

    out_path_h5 = os.path.join(out_path, data_type + "_trainJets.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("train_jet_data", data=train_jetData)
    f.close()
    created_files.append(out_path_h5)

    out_path_h5 = os.path.join(out_path, data_type + "_train.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_constituents", data=train_data)
    f.close()
    created_files.append(out_path_h5)

    out_path_h5 = os.path.join(out_path, data_type + "_testJets.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("test_jet_data", data=test_jetData)
    f.close()
    created_files.append(out_path_h5)

    out_path_h5 = os.path.join(out_path, data_type + "_test.h5")
    with h5py.File(out_path_h5, "w") as f:
        f.create_dataset("jet_constituents", data=test_data)
    f.close()
    created_files.append(out_path_h5)

    for f in created_files:
        print(f"created {f}")
    return events
