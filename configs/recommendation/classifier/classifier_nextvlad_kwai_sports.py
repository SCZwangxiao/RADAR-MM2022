_base_ = [
    '../../_base_/kwai_runtime.py'
]

# model settings
num_classes = 2066
model = dict(
    type='RecognizerHeadOnly',
    backbone=dict(
        type='Indentity'),
    cls_head=dict(
        type='NextVLADHead',
        num_classes=num_classes,
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        in_channels=768,
        cluster_size=128,
        expansion=2,
        groups=8,
        hidden_size=1536,
        gating_reduction=8,
        dropout_ratio=0.4),
    # model training and testing settings
    # train_cfg=dict(aux_info=['vertical']),
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'FrameFeatureDataset'
data_root = './data/kwai/frame_feat'
ann_file_train = './data/kwai/kwai_sports_video_train_list.txt'
ann_file_val = './data/kwai/kwai_sports_video_val_list.txt'
ann_file_test = './data/kwai/kwai_sports_video_test_list.txt'

train_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1024,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        split='train',
        pipeline=train_pipeline,
        total_clips=8,
        num_classes=num_classes),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        split='val',
        pipeline=val_pipeline,
        total_clips=8,
        num_classes=num_classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        split='test',
        pipeline=test_pipeline,
        total_clips=8,
        num_classes=num_classes))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.01, # 0.01 is used for 8 gpus 32 videos/gpu
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# learning policy
lr_config = dict(
    policy='step', 
    step=[2, 4, 6, 8],
    gamma=0.8,
    by_epoch=True)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=1, by_epoch=True)
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])