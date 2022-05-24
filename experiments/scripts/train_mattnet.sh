GPU_ID=0 #$1
DATASET='rsvg' #$2

IMDB="dota_v1_0"
TAG="RoITransformer"
NET="res50"
ID="dota_pretrain_roitrans"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --id ${ID} \
    --max_epochs 15 \
    --learning_rate 4e-4 \
    --erase_train 0 \
    --batch_size 28
