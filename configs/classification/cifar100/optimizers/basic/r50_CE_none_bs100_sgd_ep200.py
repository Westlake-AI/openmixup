_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001)
