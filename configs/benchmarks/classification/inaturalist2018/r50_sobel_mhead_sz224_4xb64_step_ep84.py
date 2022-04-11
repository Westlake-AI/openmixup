_base_ = 'r50_mhead_sz224_4b64_step_ep84.py'

# model settings
model = dict(
    with_sobel=True,
    backbone=dict(in_channels=2),
)
