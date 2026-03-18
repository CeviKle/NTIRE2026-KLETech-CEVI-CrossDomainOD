#!/bin/bash
# =============================================================================
# GLIP-CDFSOD Inference Script
# Runs inference on all 9 dataset/shot combinations using downloaded checkpoints
# Results are saved to GLIP_result/ for ensemble
# =============================================================================

# Note: not using set -e so script continues on individual failures
set -u

# Ensure PyTorch shared libs are found
TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH:-}
fi

DIR=/NTIRE2026/runs/C22_CrossDomain_FS_OD/GLIP-CDFSOD
CONFIG=${DIR}/configs/pretrain/glip_Swin_L.yaml
CKPT_DIR=${DIR}/checkpoints

# Set DATASET env var so paths_catalog.py can find testphase datasets
export DATASET=/NTIRE2026/C22_CrossDomain_FS_OD/testphase_datasets

# Output directory for GLIP results
RESULT_DIR=${DIR}/GLIP_result
mkdir -p ${RESULT_DIR}

# Common inference arguments
COMMON_ARGS="TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
    TEST.EVAL_TASK detection \
    DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
    DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
    DATASETS.USE_OVERRIDE_CATEGORY True \
    DATASETS.USE_CAPTION_PROMPT True"

# Keep multiscale inference enabled.
SCALES=(640 800 1000)
FUSE_IOU=0.55

TMP_SCALE_DIR=${DIR}/OUTPUT/multiscale_preds
mkdir -p "${TMP_SCALE_DIR}"

run_one_scale() {
    local weight=$1
    local task_cfg=$2
    local min_size=$3
    local add_linear=${4:-False}
    local marker_file
    marker_file=$(mktemp)

    touch "${marker_file}"
    if [ "${add_linear}" = "True" ]; then
        python tools/test_grounding_net.py \
            --config-file "${CONFIG}" \
            --weight "${weight}" \
            --task_config "${task_cfg}" \
            ${COMMON_ARGS} \
            MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER True \
            INPUT.MIN_SIZE_TEST "${min_size}"
    else
        python tools/test_grounding_net.py \
            --config-file "${CONFIG}" \
            --weight "${weight}" \
            --task_config "${task_cfg}" \
            ${COMMON_ARGS} \
            INPUT.MIN_SIZE_TEST "${min_size}"
    fi

    local bbox_file
    bbox_file=$(find "${DIR}/OUTPUT/eval" -type f -name "bbox.json" -newer "${marker_file}" 2>/dev/null | sort | tail -n 1)
    rm -f "${marker_file}"
    echo "${bbox_file}"
}

fuse_scales_to_output() {
    local output_json=$1
    shift
    local inputs=("$@")
    if [ ${#inputs[@]} -eq 0 ]; then
        echo "[WARN] No scale outputs available for ${output_json}"
        return 1
    fi

    python3 - "${output_json}" "${FUSE_IOU}" "${inputs[@]}" << 'PY'
import json
import sys
from collections import defaultdict

out_path = sys.argv[1]
iou_thr = float(sys.argv[2])
in_files = sys.argv[3:]

def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

preds = []
for fp in in_files:
    with open(fp, "r") as f:
        preds.extend(json.load(f))

groups = defaultdict(list)
for p in preds:
    groups[(p["image_id"], p["category_id"])].append(p)

fused = []
for _, dets in groups.items():
    dets = sorted(dets, key=lambda x: x.get("score", 0.0), reverse=True)
    while dets:
        best = dets.pop(0)
        fused.append(best)
        keep = []
        for d in dets:
            if iou_xywh(best["bbox"], d["bbox"]) < iou_thr:
                keep.append(d)
        dets = keep

with open(out_path, "w") as f:
    json.dump(fused, f)
print(f"Saved fused output: {out_path} (count={len(fused)})")
PY
}

run_multiscale_job() {
    local tag=$1
    local weight=$2
    local task_cfg=$3
    local add_linear=${4:-False}

    local scale_files=()
    for scale in "${SCALES[@]}"; do
        echo "    - scale ${scale}"
        local bbox
        bbox=$(run_one_scale "${weight}" "${task_cfg}" "${scale}" "${add_linear}")
        if [ -z "${bbox}" ] || [ ! -f "${bbox}" ]; then
            echo "[WARN] Missing bbox output for ${tag} at scale ${scale}"
            continue
        fi
        local out_scale="${TMP_SCALE_DIR}/${tag}_s${scale}.json"
        cp "${bbox}" "${out_scale}"
        scale_files+=("${out_scale}")
    done

    fuse_scales_to_output "${RESULT_DIR}/${tag}.json" "${scale_files[@]}"
}

echo "============================================"
echo "Starting GLIP-CDFSOD Inference"
echo "Working directory: ${DIR}"
echo "Dataset root: ${DATASET}"
echo "============================================"

# -------------------------------------------
# Dataset 1 (Fruit detection, 8 classes)
# -------------------------------------------

echo ""
echo ">>> Dataset 1 - 1 shot"
run_multiscale_job "dataset1_1shot" "${CKPT_DIR}/data1_1s_full_pse0.4-0.5_1333.pth" "${DIR}/configs/cdfsod/dataset1.yaml"

echo ""
echo ">>> Dataset 1 - 5 shot"
run_multiscale_job "dataset1_5shot" "${CKPT_DIR}/data1_5s_seed2_full_ps0.4-0.5_234.pth" "${DIR}/configs/cdfsod/dataset1.yaml"

echo ""
echo ">>> Dataset 1 - 10 shot"
run_multiscale_job "dataset1_10shot" "${CKPT_DIR}/data1_10s_full_pse0.4-0.5_233.pth" "${DIR}/configs/cdfsod/dataset1.yaml"

# -------------------------------------------
# Dataset 2 (Car detection, 2 classes)
# -------------------------------------------

echo ""
echo ">>> Dataset 2 - 1 shot"
run_multiscale_job "dataset2_1shot" "${CKPT_DIR}/data2_1s_model_0000014.pth" "${DIR}/configs/cdfsod/dataset2.yaml"

echo ""
echo ">>> Dataset 2 - 5 shot"
run_multiscale_job "dataset2_5shot" "${CKPT_DIR}/data2_5s_model_0000008.pth" "${DIR}/configs/cdfsod/dataset2.yaml"

echo ""
echo ">>> Dataset 2 - 10 shot"
run_multiscale_job "dataset2_10shot" "${CKPT_DIR}/data2_10s_model_0000004.pth" "${DIR}/configs/cdfsod/dataset2.yaml"

# -------------------------------------------
# Dataset 3 (Car damage detection, 7 classes)
# NOTE: 1-shot and 5-shot need ADD_LINEAR_LAYER True
# -------------------------------------------

echo ""
echo ">>> Dataset 3 - 1 shot (ADD_LINEAR_LAYER=True)"
run_multiscale_job "dataset3_1shot" "${CKPT_DIR}/data3_1s_seed3_tv2_25iter.pth" "${DIR}/configs/cdfsod/dataset3.yaml" True

echo ""
echo ">>> Dataset 3 - 5 shot (ADD_LINEAR_LAYER=True)"
run_multiscale_job "dataset3_5shot" "${CKPT_DIR}/data3_5s_seed0_tv2_40iter.pth" "${DIR}/configs/cdfsod/dataset3.yaml" True

echo ""
echo ">>> Dataset 3 - 10 shot"
run_multiscale_job "dataset3_10shot" "${CKPT_DIR}/data3_10s_tv1.pth" "${DIR}/configs/cdfsod/dataset3.yaml"

echo ""
echo "ID audit against test annotations"
python3 - << 'PY'
import glob
import json
import os

result_dir = '/NTIRE2026/runs/C22_CrossDomain_FS_OD/GLIP-CDFSOD/GLIP_result'
dataset_root = '/NTIRE2026/C22_CrossDomain_FS_OD/testphase_datasets'

for ds in ('dataset1', 'dataset2', 'dataset3'):
    ann_path = os.path.join(dataset_root, ds, 'annotations', 'test.json')
    if not os.path.exists(ann_path):
        print(f'{ds}: missing annotation file {ann_path}')
        continue
    ann = json.load(open(ann_path, 'r'))
    valid_images = {x['id'] for x in ann.get('images', [])}
    valid_categories = {x['id'] for x in ann.get('categories', [])}

    for pred_file in sorted(glob.glob(os.path.join(result_dir, f'{ds}_*shot.json'))):
        preds = json.load(open(pred_file, 'r'))
        bad_img = sum(1 for p in preds if p.get('image_id') not in valid_images)
        bad_cat = sum(1 for p in preds if p.get('category_id') not in valid_categories)
        print(f"{os.path.basename(pred_file)} total={len(preds)} bad_image_id={bad_img} bad_category_id={bad_cat}")
PY

echo ""
echo "============================================"
echo "All inference complete!"
echo "Results saved in: ${RESULT_DIR}/"
ls -la ${RESULT_DIR}/
echo "============================================"
echo ""
echo "To run ensemble with DINO results, place DINO JSONs in DINO_result/ and run:"
echo "  python ensemble.py"
