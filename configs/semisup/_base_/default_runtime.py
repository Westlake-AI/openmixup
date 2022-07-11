train_cfg = {}
test_cfg = {}
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # Bug: remove TensorboardLoggerHook in PyTorch1.10
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# fp16
use_fp16 = False
fp16 = dict(loss_scale='dynamic')

# Sets `find_unused_parameters` in torch.nn.parallel.DistributedDataParallel if necessary
find_unused_parameters = False
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
