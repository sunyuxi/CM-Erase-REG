GPU_ID=0 #$1

IMDB="dota_v1_0"
TAG="RoITransformer"
NET="res50"
DATASET="rsvg"
ID=rsvg_erase

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --id ${ID} \
    --learning_rate 4e-4 \
    --learning_rate_decay_start 6 \
    --learning_rate_decay_every 6 \
    --max_epochs 30 \
    --erase_size_visual 2 \
    --erase_lang_weight 1 \
    --erase_allvisual_weight 1 \
    --erase_train 1 \
    --batch_size 20 \
    --start_from dota_pretrain_roitrans
