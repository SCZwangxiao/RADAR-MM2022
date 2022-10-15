# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='Exp', gamma=1.0001)
total_epochs = 100
