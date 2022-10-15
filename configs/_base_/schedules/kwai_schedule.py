# optimizer
optimizer = dict(
    type='AdamW', lr=0.0005,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 20