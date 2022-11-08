_base_ = './moga_large_sz224_8xb64_accu2_ep300.py'

# additional hooks
custom_hooks = [
    dict(type='PreciseBNHook',
        num_samples=8192,
        update_all_stats=False,
        interval=1,
    ),
]
