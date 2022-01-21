
# 11.12, IP89, Tiny R18
sleep 5400s

CUDA_VISIBLE_DEVICES=0,1 PORT=120000 bash tools/dist_train.sh configs/classification/tiny_imagenet/automixV2/v0824_1011/r18/CE_q_soft_BCE_k/r18_v0824_1110_l2_dp0_CE_use_soft_neg0_01_k0_50_mask_adj0__min_lr5e_2_ep400.py 2 &
CUDA_VISIBLE_DEVICES=2,3 PORT=120001 bash tools/dist_train.sh configs/classification/tiny_imagenet/automixV2/v0824_1011/r18/CE_q_soft_BCE_k/r18_v0824_1110_l2_dp0_CE_use_soft_neg0__k0_25_mask_adj0__min_lr5e_2_ep400.py 2 &
CUDA_VISIBLE_DEVICES=4,5 PORT=120002 bash tools/dist_train.sh configs/classification/tiny_imagenet/automixV2/v0824_1011/r18/CE_q_soft_BCE_k/r18_v0824_1110_l2_dp0_CE_use_soft_neg0_001_k0_25_mask_adj0__min_lr5e_2_ep400.py 2 &
CUDA_VISIBLE_DEVICES=6,7 PORT=120003 bash tools/dist_train.sh configs/classification/tiny_imagenet/automixV2/v0824_1011/r18/CE_q_soft_BCE_k/r18_v0824_1110_l2_dp0_CE_use_soft_neg0_50_k0_25_mask_adj0__min_lr5e_2_ep400.py 2


echo "finished, soft BCE Tiny"
