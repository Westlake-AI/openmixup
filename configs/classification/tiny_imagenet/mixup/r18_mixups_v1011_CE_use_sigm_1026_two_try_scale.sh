
# 10.26, 89
##### Finished #######
# # two_hotTrue two_hot_modenone mix_modemixup
# CUDA_VISIBLE_DEVICES=0 PORT=100000 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100001 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100002 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100003 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100004 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100005 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100006 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100007 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100008 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modenone mix_modecutmix
# CUDA_VISIBLE_DEVICES=1 PORT=100009 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100010 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100011 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100012 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100013 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100014 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100015 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_1

# ###### resume, 10.27 ######
# CUDA_VISIBLE_DEVICES=0 PORT=100016 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100017 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modenone mix_modesaliencymix
# CUDA_VISIBLE_DEVICES=2 PORT=100018 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100019 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100020 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100021 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100022 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100023 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100024 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100025 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100026 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modenone mix_modefmix
# CUDA_VISIBLE_DEVICES=3 PORT=100027 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100028 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100029 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100030 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100031 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_2


# # finished ###### resume, 10.29 ######
# CUDA_VISIBLE_DEVICES=0 PORT=100032 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=120033 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=120034 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=120035 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modenone mix_moderesizemix
# CUDA_VISIBLE_DEVICES=4 PORT=100036 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100037 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100038 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100039 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100040 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100041 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100042 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100043 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100044 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modenone mix_modemanifoldmix
# CUDA_VISIBLE_DEVICES=5 PORT=100045 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100046 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100047 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_3


# resume, 10.30
# CUDA_VISIBLE_DEVICES=0 PORT=100048 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100049 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100050 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100051 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100052 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100053 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# ##############################################
# # two_hotTrue two_hot_modepow mix_modemixup
# CUDA_VISIBLE_DEVICES=6 PORT=100000 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100001 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100002 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100003 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100004 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100005 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100006 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100007 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100008 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemixup/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modepow mix_modecutmix
# CUDA_VISIBLE_DEVICES=7 PORT=100009 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_4


# CUDA_VISIBLE_DEVICES=0 PORT=100010 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100011 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100012 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100013 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100014 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100015 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100016 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100017 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modecutmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# two_hotTrue two_hot_modepow mix_modesaliencymix
# CUDA_VISIBLE_DEVICES=0 PORT=100018 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100019 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100020 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100021 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100022 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100023 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100024 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100025 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_5


# CUDA_VISIBLE_DEVICES=0 PORT=100026 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modepow mix_modefmix
# CUDA_VISIBLE_DEVICES=1 PORT=100027 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100028 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100029 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100030 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100031 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100032 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100033 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100034 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100035 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modepow mix_moderesizemix
# CUDA_VISIBLE_DEVICES=2 PORT=100036 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100037 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100038 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100039 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100040 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100041 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1

# sleep 1000s 
# echo "pause"
# touch /usr/lsy/src/OpenSelfSup_v0705/work_dirs/classification/tiny_imagenet/finished_6


# CUDA_VISIBLE_DEVICES=0 PORT=100042 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100043 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100044 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# # two_hotTrue two_hot_modepow mix_modemanifoldmix
# CUDA_VISIBLE_DEVICES=3 PORT=100045 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=100046 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100047 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100048 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100049 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=0 PORT=100050 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100051 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 


# CUDA_VISIBLE_DEVICES=6 PORT=100052 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=7 PORT=100053 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1


############## TO BE resume #################

# saliencymix, RUNNED
# CUDA_VISIBLE_DEVICES=0 PORT=110026 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modesaliencymix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# resizemix
# CUDA_VISIBLE_DEVICES=1 PORT=110040 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=110041 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &

# CUDA_VISIBLE_DEVICES=3 PORT=110042 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=4 PORT=110043 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=110044 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1 &
# sleep 1s 


# two_hotTrue two_hot_modepow mix_modemanifoldmix
# CUDA_VISIBLE_DEVICES=4 PORT=100045 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=5 PORT=100046 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=6 PORT=100047 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
# sleep 1s 

# CUDA_VISIBLE_DEVICES=0 PORT=100048 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=1 PORT=100049 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=2 PORT=100050 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
# sleep 1s 
# CUDA_VISIBLE_DEVICES=3 PORT=100051 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modepow/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
# sleep 1s 

###############################################

# resume, 11.02

# fmix
# CUDA_VISIBLE_DEVICES=0,1 PORT=130033 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 2 &
CUDA_VISIBLE_DEVICES=2,3 PORT=130034 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 2 &
# CUDA_VISIBLE_DEVICES=4,5 PORT=130035 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modefmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 2 &

# resizemix
CUDA_VISIBLE_DEVICES=6,7 PORT=100041 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 2
echo "pause"
sleep 10s

CUDA_VISIBLE_DEVICES=6 PORT=100043 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_moderesizemix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100051 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &

# manifoldmix
CUDA_VISIBLE_DEVICES=6 PORT=100047 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100045 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_25_ep400.json 1
echo "pause"
sleep 10s

CUDA_VISIBLE_DEVICES=6 PORT=100046 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale0_5_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100047 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha0_2_two_hot_scale1_ep400.json 1 &
CUDA_VISIBLE_DEVICES=6 PORT=100048 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_25_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100049 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale0_5_ep400.json 1
echo "pause"
sleep 10s

CUDA_VISIBLE_DEVICES=6 PORT=100050 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha1_two_hot_scale1_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100051 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_25_ep400.json 1 &
CUDA_VISIBLE_DEVICES=6 PORT=100052 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale0_5_ep400.json 1 &
CUDA_VISIBLE_DEVICES=7 PORT=100053 bash tools/dist_train.sh configs/classification/tiny_imagenet/mixup/r18/two_hotTrue/two_hot_modenone/mix_modemanifoldmix/r18_mixups_v1011_CE_use_sigm_alpha2_two_hot_scale1_ep400.json 1

###############################################

echo "finished"
