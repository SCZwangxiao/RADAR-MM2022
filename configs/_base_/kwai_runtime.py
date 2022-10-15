# evaluation
evaluation = dict(
    interval=1,
    metrics=['mmit_mean_average_precision', 'mean_average_precision',
            'top_k_precision', 'top_k_recall'],  # mmit: sample-based. mAP
    metric_options=dict(top_k_precision=dict(topk=(1, 3)),
                        top_k_recall=dict(topk=(5, 10))),
    save_best='mmit_mean_average_precision')

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(  # 注册日志钩子的设置
    interval=10,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]