#!/bin/bash

run_sample() {
    local sample=$1
    python scripts/run_dataForge.py \
    -i /ceph/cms/store/group/LLPs/russelld/fp_ntuples/ \
    -s ${sample} \
    -d Signal \
    -t outnano/Jets \
    -o /home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/${sample}
}

input=${1:-samples.txt}

if [[ -f "$input" ]]; then
    while read -r sample; do
        run_sample "$sample"
    done < "$input"
else
    run_sample "$input"
fi