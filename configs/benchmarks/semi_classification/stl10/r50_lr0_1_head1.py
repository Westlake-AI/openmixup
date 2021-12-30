_base_ = 'base.py'
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                 paramwise_options={'\Ahead.': dict(lr_mult=1)})

# CUDA_VISIBLE_DEVICES=1 bash benchmarks/dist_train_semi_1gpu.sh configs/benchmarks/semi_classification/stl10/r50_lr0_1_head1.py ${WEIGHT_FILE}
