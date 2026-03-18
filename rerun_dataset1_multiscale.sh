#!/bin/bash
# Rerun multiscale inference for dataset1 only (all 3 shots × 3 scales).
# dataset1.yaml has been fixed to correct 10 marine-animal classes.

set -u

DIR=/NTIRE2026/runs/C22_CrossDomain_FS_OD/GLIP-CDFSOD
CONFIG=${DIR}/configs/pretrain/glip_Swin_L.yaml
CKPT_DIR=${DIR}/checkpoints

export DATASET=/NTIRE2026/C22_CrossDomain_FS_OD/testphase_datasets

SCALES=(640 800 1000)
IOU_THRESH=0.55

MS_ROOT=${DIR}/OUTPUT/ms_predictions
RESULT_DIR=${DIR}/GLIP_result_multiscale
mkdir -p "${MS_ROOT}" "${RESULT_DIR}"

COMMON_ARGS="TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
TEST.EVAL_TASK detection \
DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
DATASETS.USE_OVERRIDE_CATEGORY True \
DATASETS.USE_CAPTION_PROMPT True"

run_combo() {
    local dataset_id=$1
    local shot=$2
    local ckpt=$3
    local task_cfg=$4
    local add_linear=$5

    local combo_dir="${MS_ROOT}/dataset${dataset_id}_${shot}shot"
    mkdir -p "${combo_dir}"

    local ckpt_stem
    ckpt_stem=$(basename "${ckpt}" .pth)

    local input_files=()

    echo ""
    echo "============================================================"
    echo "Dataset ${dataset_id} - ${shot} shot"
    echo "Checkpoint: ${ckpt}"
    echo "============================================================"

    for scale in "${SCALES[@]}"; do
        local max_size=$((scale * 5 / 3))
        local outdir="${DIR}/OUTPUT/ms_runs/d${dataset_id}_s${shot}_scale${scale}"
        local bbox_file="${outdir}/eval/${ckpt_stem}/inference/test/bbox.json"
        local scale_out="${combo_dir}/scale_${scale}.json"

        echo "[Scale ${scale}] MIN_SIZE_TEST=${scale} MAX_SIZE_TEST=${max_size}"

        if [ "${add_linear}" = "1" ]; then
            python tools/test_grounding_net.py \
                --config-file "${CONFIG}" \
                --weight "${ckpt}" \
                --task_config "${task_cfg}" \
                ${COMMON_ARGS} \
                MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER True \
                INPUT.MIN_SIZE_TEST "${scale}" \
                INPUT.MAX_SIZE_TEST "${max_size}" \
                OUTPUT_DIR "${outdir}"
        else
            python tools/test_grounding_net.py \
                --config-file "${CONFIG}" \
                --weight "${ckpt}" \
                --task_config "${task_cfg}" \
                ${COMMON_ARGS} \
                INPUT.MIN_SIZE_TEST "${scale}" \
                INPUT.MAX_SIZE_TEST "${max_size}" \
                OUTPUT_DIR "${outdir}"
        fi

        if [ ! -f "${bbox_file}" ]; then
            echo "ERROR: Missing output ${bbox_file}"
            return 1
        fi

        cp "${bbox_file}" "${scale_out}"
        input_files+=("${scale_out}")
        echo "Saved scale output: ${scale_out}"
    done

    local fused_out="${RESULT_DIR}/dataset${dataset_id}_${shot}shot.json"
    python "${DIR}/fuse_multiscale.py" --inputs "${input_files[@]}" --output "${fused_out}" --iou "${IOU_THRESH}"
    echo "Final fused output: ${fused_out}"
}

cd "${DIR}"

echo "========================================"
echo "Rerunning dataset1 multiscale inference"
echo "Config: configs/cdfsod/dataset1.yaml"
echo "========================================"

run_combo 1 1  "${CKPT_DIR}/data1_1s_full_pse0.4-0.5_1333.pth"    "${DIR}/configs/cdfsod/dataset1.yaml" 0 || exit 1
run_combo 1 5  "${CKPT_DIR}/data1_5s_seed2_full_ps0.4-0.5_234.pth" "${DIR}/configs/cdfsod/dataset1.yaml" 0 || exit 1
run_combo 1 10 "${CKPT_DIR}/data1_10s_full_pse0.4-0.5_233.pth"     "${DIR}/configs/cdfsod/dataset1.yaml" 0 || exit 1

echo ""
echo "Done. Dataset1 fused outputs in: ${RESULT_DIR}"
ls -lh "${RESULT_DIR}"/dataset1_*.json
