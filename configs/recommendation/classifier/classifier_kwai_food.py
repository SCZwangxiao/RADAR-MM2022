_base_ = [
    '../../_base_/kwai_runtime.py'
]

# Choose vertical
num_tags = 8640
vertical = 'food'
dataset_root = './data/kwai'
video_emb_dir = 'video_feat'
tag_emb_dir = 'tag_feat_bert'
reload_dataset = False

# model settings
model = dict(
    type='TagGraphRecommender',
    gnn=dict(type='ClassifierGNN'),
    linker=dict(
        type='ClassifierLinker',
        num_tags=num_tags,
        feat_dim=768,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.),
        dropout_ratio=0.2,
        label_smooth_eps=0))


# dataset settings
dataset_type = 'KwaiTagRecoDataset'
train_fanouts = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 0,
    ('tag', 'SubTopic', 'tag'): 0,
    ('video', 'HasTag', 'tag'): 0
}]
infer_fanouts = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 0,
    ('tag', 'SubTopic', 'tag'): 0,
    ('video', 'HasTag', 'tag'): 0
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

# optimizer
optimizer = dict(
    type='AdamW', lr=0.01, weight_decay=0.001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 80