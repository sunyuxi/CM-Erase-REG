GPU_ID=1 #$1
DATASET='rsvg' #$2
SPLIT='test' #$3

ID="rsvg_erase"
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
                --id ${ID}

SPLIT='val' #$3
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
                --id ${ID}