# model settings
model = dict(
    type='BaseMultimodalRecognizer',
    modality=['Vision', 'Language'],
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))