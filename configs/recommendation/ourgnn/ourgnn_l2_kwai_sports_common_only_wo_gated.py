_base_ = [
    '../../_base_/kwai_runtime.py',
    '../../_base_/schedules/kwai_schedule.py',
]

# Choose vertical
num_tags = 2066
vertical = 'sports'
dataset_root = './data/kwai'
video_emb_dir = 'video_feat'
tag_emb_dir = 'tag_feat_bert'
reload_dataset = False

# model settings
num_gnn_layers = 2
model = dict(
    type='TagGraphRecommender',
    gnn=dict(
        type='OurGNN',
        num_tags=num_tags,
        video_in_dim=768,
        tag_in_dim=768,
        hidden_dim=768,
        video_emb_layer=2,
        layer_relations=[
            [('video', 'FollowedBy', 'video'),
                ('tag', 'SubTopic', 'tag'),
                ('video', 'HasTag', 'tag')],
            [('video', 'FollowedBy', 'video'),
                ('tag', 'SubTopic', 'tag'),
                ('video', 'HasTag', 'tag')]],
        message_passer=dict(
            type='TransformerMessagePasser',
            in_dim=768,
            out_dim=768,
            num_heads=8),
        aggregator=dict(
            type='StackAggregator'),
        updator=dict(
            type='SumUpdator',
            hidden_dim=768,
            dropout=0.2),
        reducer=dict(
            type='CommonReducer',
            hidden_dim=768,
            dropout=0.2)),
    linker=dict(
        type='DeviseLinker',
        loss_cls=dict(type='BCELossWithLogits',loss_weight=333.),
        dropout_ratio=0.2,
        label_smooth_eps=0))

# dataset settings
dataset_type = 'KwaiTagRecoDataset'
train_fanouts_base = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 4,
    ('tag', 'SubTopic', 'tag'): 4,
    ('video', 'HasTag', 'tag'): 4}]
train_fanouts_final = [{
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 4,
    ('tag', 'SubTopic', 'tag'): 4,
    ('video', 'HasTag', 'tag'): 4}]
infer_fanouts_base = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 4,
    ('tag', 'SubTopic', 'tag'): 4,
    ('video', 'HasTag', 'tag'): 4}]
infer_fanouts_final = [{
    ('tag', 'WhetherHasVideo', 'video'): 0,
    ('tag', 'HasVideo', 'video'): 0,
    ('tag', 'NotHasVideo', 'video'): 0,
    ('video', 'FollowedBy', 'video'): 4,
    ('tag', 'SubTopic', 'tag'): 4,
    ('video', 'HasTag', 'tag'): 4}]
train_fanouts = train_fanouts_base*(num_gnn_layers-1) + train_fanouts_final
infer_fanouts = infer_fanouts_base*(num_gnn_layers-1) + infer_fanouts_final

data = dict(
    videos_per_gpu=1024,
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

# work_dir = './work_dirs/ourgnn_l1_kwai_sports/'