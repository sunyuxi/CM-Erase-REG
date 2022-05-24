

GPU_ID=1 #$1
DATASET='rsvg' #$2
SPLIT='test' #$3

ID="rsvg_erase"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.5 \
                --id ${ID}

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.25 \
                --id ${ID}

SPLIT='val' #$3
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.5 \
                --id ${ID}

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.25 \
                --id ${ID}