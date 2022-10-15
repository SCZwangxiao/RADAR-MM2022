_base_ = [
    '../../_base_/default_runtime.py'
]

# Choose vertical
vertical = 'sports'
dataset_root = './data/kwai'
video_emb_dir = 'video_feat'
tag_emb_dir = 'tag_feat_bert'
reload_dataset = False

# model settings
model = dict(
    type='TagGraphRecommender',
    gnn=dict(
        type='DeviseGNN',
        video_in_dim=768,
        video_out_dim=768,
        tag_in_dim=None,
        tag_out_dim=None,
        dropout_ratio=.2),
    linker=dict(
        type='DeviseLinker',
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        label_smooth_eps=0))


# dataset settings
dataset_type = 'KwaiTagRecoDataset'
train_fanouts = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 0,
    ('tag', 'SubTopic', 'tag'): 0,
    ('video', 'HasTag', 'tag')
}]
infer_fanouts = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 0,
    ('tag', 'SubTopic', 'tag'): 0,
    ('video', 'HasTag', 'tag')
}]


data = dict(
    videos_per_gpu=1024, # only one graph
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        vertical=vertical,
        split='train',
        dataset_root=dataset_root,
        video_emb_dir=video_emb_dir,
        tag_emb_dir=tag_emb_dir,
        force_reload=reload_dataset),
    train_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='KwaiNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=train_fanouts)),
    val=dict(
        type=dataset_type,
        vertical=vertical,
        split='val',
        dataset_root=dataset_root,
        video_emb_dir=video_emb_dir,
        tag_emb_dir=tag_emb_dir,
        force_reload=reload_dataset),
    val_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='KwaiNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=infer_fanouts)),
    test=dict(
        type=dataset_type,
        vertical=vertical,
        split='test',
        dataset_root=dataset_root,
        video_emb_dir=video_emb_dir,
        tag_emb_dir=tag_emb_dir,
        force_reload=reload_dataset),
    test_dataloader=dict(
        dataset_type='graph',
        collator=dict(
            type='KwaiNodeCollator'),
        sampler=dict(
            type='MultiLayerNeighborSampler',
            fanouts=infer_fanouts)))

evaluation = dict(
    interval=1,
    metrics=['mmit_mean_average_precision', 
            'top_k_precision', 'top_k_recall'],  # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3)),
                        top_k_recall=dict(topk=(5, 10))),
    save_best='mmit_mean_average_precision')

# optimizer
optimizer = dict(
    type='Adam', lr=0.01, weight_decay=0.001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(  # 注册日志钩子的设置
    interval=10,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

work_dir = './work_dirs/devise_bert_kwai_sports/'