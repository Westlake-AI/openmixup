_base_ = './moga_small_sz224_8xb128_ep300.py'

# additional hooks
custom_hooks = [
    dict(type='PreciseBNHook',
        num_samples=8192,
        update_all_stats=False,
        interval=1,
    ),
]
