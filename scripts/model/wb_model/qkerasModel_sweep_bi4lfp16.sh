#python3 qkerasModel.py "/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/4b/M_LLP_30_ctau_1000/newTrainDataPT20_Signal_Only.h5" "/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/M_LLP_30_ctau_100/trainingDataQCD30.h5" "/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/4b/M_LLP_30_ctau_1000/newSampleDataPT20_Signal_Only.h5" "/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/M_LLP_30_ctau_100/sampleDataQCD30.h5" "4b/M_LLP_30_ctau_1000"
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
WANDB_MODE=offline
python3 qkerasModel_sweep_bi4lfp16.py "/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/train_merged/merged_trainPart.h5" \
"/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/QCD_Pt15To3000_Flat_PU200/Bkg_train.h5" \
"/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/train_merged/merged_trainJet.h5" \
"/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5"
