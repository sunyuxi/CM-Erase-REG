#rm -f cache/feats/rsvg/refnms_res50_dota_v1_0_RoITransformer_det_feats.h5
#python tools/extract_mrcn_det_feats_refnms.py
#sh experiments/scripts/eval_dets_refnms.sh

GPU_ID=1 #$1
DATASET='rsvg' #$2
SPLIT='test' #$3
ID="rsvg_erase"

refnmsdet_dirpath=hbb_obb_features_refnms_det_selected256 # changed
refnmsdet_feats_suffix=hbb_det_res50_dota_v1_0_RoITransformer.hdf5
wholeimg_feats_dirpath=hbb_obb_features_wholeimg
wholeimg_feats_suffix=hbb_img_res50_dota_v1_0_RoITransformer.hdf5
refnmsdet_jsonpath="../ref-nms/output/matt_dets_att_vanilla_refnms256_rsvg_0.json" # changed
refnmsdet_meanpools_feats_path="../MAttNet/cache/feats/rsvg/refnms256_res50_dota_v1_0_RoITransformer_det_feats.h5" # changed

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets_refnms.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.5 \
                --id ${ID} \
                --refnmsdet_dirpath ${refnmsdet_dirpath} \
                --refnmsdet_feats_suffix ${refnmsdet_feats_suffix} \
                --wholeimg_feats_dirpath ${wholeimg_feats_dirpath} \
                --wholeimg_feats_suffix ${wholeimg_feats_suffix} \
                --refnmsdet_jsonpath ${refnmsdet_jsonpath} \
                --refnmsdet_meanpools_feats_path ${refnmsdet_meanpools_feats_path}

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets_refnms.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.25 \
                --id ${ID} \
                --refnmsdet_dirpath ${refnmsdet_dirpath} \
                --refnmsdet_feats_suffix ${refnmsdet_feats_suffix} \
                --wholeimg_feats_dirpath ${wholeimg_feats_dirpath} \
                --wholeimg_feats_suffix ${wholeimg_feats_suffix} \
                --refnmsdet_jsonpath ${refnmsdet_jsonpath} \
                --refnmsdet_meanpools_feats_path ${refnmsdet_meanpools_feats_path}

SPLIT='val' #$3
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets_refnms.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.5 \
                --id ${ID} \
                --refnmsdet_dirpath ${refnmsdet_dirpath} \
                --refnmsdet_feats_suffix ${refnmsdet_feats_suffix} \
                --wholeimg_feats_dirpath ${wholeimg_feats_dirpath} \
                --wholeimg_feats_suffix ${wholeimg_feats_suffix} \
                --refnmsdet_jsonpath ${refnmsdet_jsonpath} \
                --refnmsdet_meanpools_feats_path ${refnmsdet_meanpools_feats_path}

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets_refnms.py \
                --dataset ${DATASET} \
                --split ${SPLIT} \
		        --iou_threshold 0.25 \
                --id ${ID} \
                --refnmsdet_dirpath ${refnmsdet_dirpath} \
                --refnmsdet_feats_suffix ${refnmsdet_feats_suffix} \
                --wholeimg_feats_dirpath ${wholeimg_feats_dirpath} \
                --wholeimg_feats_suffix ${wholeimg_feats_suffix} \
                --refnmsdet_jsonpath ${refnmsdet_jsonpath} \
                --refnmsdet_meanpools_feats_path ${refnmsdet_meanpools_feats_path}
