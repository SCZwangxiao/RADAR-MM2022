_base_ = [
    '../../_base_/models/multimodal.py',
    '../../_base_/default_runtime.py'
]

# model settings
hetu_classes = 284
modality = ['Vision', 'Language']
model = dict(
    type='MultiModalRecognizer2D',
    modality=modality,
    vision_encoder=dict(
        type='FcEncoder', 
        input_dims=768, 
        hidden_dims=768),
    language_encoder=dict(
        type='FcEncoder', 
        input_dims=768, 
        hidden_dims=768),
    reactor=dict(
        type='BasicReactor', 
        operations=['concat']),
    cls_head=dict(
        type='VanillaMultiModalHead',
        num_classes=hetu_classes,
        multi_class=True,
        in_channels=768+768,
        dropout_ratio=0.4,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'MultimodalDataset'
modality = ['RGB', 'Text']
dataset_type_modality = ['VideoDataset', 'SentenceDataset']
data_root = [
    '../data/hetu/video_feature',
    '../data/hetu/sentence_feature']
data_root_val = [
    '../data/hetu/video_feature',
    '../data/hetu/sentence_feature']
ann_file_train = [
    '../data/hetu/hetu_video_train_list.txt',
    '../data/hetu/hetu_text_train_list.txt']
ann_file_val = [
    '../data/hetu/hetu_video_val_list.txt',
    '../data/hetu/hetu_text_val_list.txt']
ann_file_test = [
    '../data/hetu/hetu_video_test_list.txt',
    '../data/hetu/hetu_text_test_list.txt']
ann_class_list = '/home/wangxiao13/annotation/data/hetu/label_map_hetu.txt'


train_pipeline = [
    dict(type='LoadVideoFeature'),
    dict(type='LoadTextFeature'),
    dict(type='Collect', keys=['imgs', 'sents', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'sents', 'label'])
]
val_pipeline = [
    dict(type='LoadVideoFeature'),
    dict(type='LoadTextFeature'),
    dict(type='Collect', keys=['imgs', 'sents', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'sents', 'label'])
]
test_pipeline = [
    dict(type='LoadVideoFeature'),
    dict(type='LoadTextFeature'),
    dict(type='Collect', keys=['imgs', 'sents', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'sents', 'label'])
]


data = dict(
    videos_per_gpu=128,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        types_modality=dataset_type_modality,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        modality=modality,
        multi_class=True,
        num_classes=hetu_classes),
    val=dict(
        type=dataset_type,
        types_modality=dataset_type_modality,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        modality=modality,
        multi_class=True,
        num_classes=hetu_classes),
    test=dict(
        type=dataset_type,
        types_modality=dataset_type_modality,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        modality=modality,
        multi_class=True,
        num_classes=hetu_classes))

evaluation = dict(
    interval=1,
    metrics=['mmit_mean_average_precision', 'top_k_precision',
             'top_k_recall'],  # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3)),
                        top_k_recall=dict(topk=(3, 5))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='Adam', lr=0.0001, weight_decay=0.00001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 5

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])