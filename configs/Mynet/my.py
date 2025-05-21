_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/default_runtime.py',
]
#edit here for custom import 
custom_imports = dict(imports=['opencd','mmpretrain.mynet','mmpretrain.transforms'], allow_failed_imports=False)
# dataset settings
epoch=1
data_root="your path"
txt_path="your path"
crop_size = (256, 256)
batch=1
num_workers=1
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    # dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    # dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    # dict(
    #     type='MultiImgPhotoMetricDistortion',
    #     brightness_delta=10,
    #     contrast_range=(0.8, 1.2),
    #     saturation_range=(0.8, 1.2),
    #     hue_delta=10),
    # dict(type='MultiImgPackSegInputs'),
        dict(type='CleanCaption', keys='gt_caption'),
    dict(
        type='MultiImgPackCaptionInputs',
        algorithm_keys=['gt_caption'],
        meta_keys=['image_id'],
    ),
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    # dict(type='MultiImgResize', scale=(1024, 1024), keep_ratio=True),
    # # add loading annotation after ``Resize`` because ground truth
    # # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=num_workers,
    dataset=dict(
        #mmpretrain/datasets/levir_cc.py
        type='LevirCCcaptions2',
        data_root=data_root,
        ann_file=data_root+'LevirCCcaptions.json',
        #ann_file=data_root+'all/LevirCCcaptions.json',
        data_prefix=dict(
            type='train',
            seg_map_path='label',
            img_path_from='A',
            img_path_to='B'),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=batch,
    num_workers=num_workers,
    dataset=dict(
        type='LevirCCcaptions2',
        data_root=data_root,
        ann_file=data_root+'LevirCCcaptions.json',
        data_prefix=dict(
            type='test',
            seg_map_path='label',
            img_path_from='A',
            img_path_to='B'),
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    #mmpretrain/mynet/metrics
    type='LevirCCcaption',
    ann_file=data_root+'LevirCCcaptions.json',
    txt_path=txt_path
)
# val_evaluator = dict(
# )
# # If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=batch,
    num_workers=num_workers,
    dataset=dict(
        type='LevirCCcaptions2',
        data_root=data_root,
        ann_file=data_root+'LevirCCcaptions.json',
        data_prefix=dict(
            type='test',
            seg_map_path='label',
            img_path_from='A',
            img_path_to='B'),
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_evaluator = val_evaluator
# dataset_type = 'LEVIR_CD_Dataset'
# train_dataloader = dict(
#     batch_size=batch,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             seg_map_path='imgs/label',
#             img_path_from='imgs/A',
#             img_path_to='imgs/B'),
#         pipeline=train_pipeline)
# )

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             seg_map_path='imgs/label',
#             img_path_from='imgs/A',
#             img_path_to='imgs/B'),
#         pipeline=test_pipeline)
# )

# test_dataloader = val_dataloader
# val_evaluator = dict(
#     type='CDMetric',
# )

# test_evaluator = val_evaluator

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32)
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
fpn_norm_cfg = dict(type='LN2d', requires_grad=True)

sam_pretrain_ckpt_path = 'your path'
qformer_path='your path'  # noqa
vicuna_path='your path'
# model settings
crop_size = (256, 256)
model = dict(

    type='MyMiniGPT4',
    data_preprocessor=data_preprocessor,
    freeze_vit=True,
    freeze_q_former = True,
    freeze_LLM=True,
    vision_encoder=dict(
        #mmpretrain.mynet/models.py
        type='MMPretrainSamVisionEncoderDual',
        encoder_cfg=dict(
            
            type='mmpretrain.ViTSAM',
            arch='large',
            img_size=crop_size[0],
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            layer_cfgs=dict(type='TimeFusionTransformerEncoderLayer2'),
            init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path, prefix='backbone.'),
        ),
        peft_cfg=dict(
            r=16,
            target_modules=["qkv"],
            lora_dropout=0.01,
            bias='lora_only',
        ),
    ),
    q_former_model=dict(
        type='Qformer',
        model_style='your path',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32,
        pretrained= qformer_path # noqa
    ),
    peft_cfg=dict(
            r=16,
            target_modules=[
    "q_proj",
    "v_proj",
    "k_proj"
],
            lora_dropout=0.01,
            bias='lora_only',
        ),
    lang_encoder=dict(
        type='AutoModelForCausalLM', name_or_path=vicuna_path),
    tokenizer=dict(type='LlamaTokenizer', name_or_path=vicuna_path),
    ##这里改任务
    #task='detection',
    #task='caption',
    task='dual',
    prompt_template=dict([('en', '###Ask: {} ###Answer: '),
                          ('zh', '###问：{} ###答：')]),
    raw_prompts=dict([
        ('en', [('<Img><ImageHere></Img> '
                 'Describe the difference between the new RS images and the old one in detail.'),
                ('<Img><ImageHere></Img> '
                 'What is the main change between the two RS scenes? Describe it in detail.'),
                ('<Img><ImageHere></Img> '
                 'Please provide a detailed description of the difference between this two RS pictures.'),
                ('<Img><ImageHere></Img> '
                 'Could you describe what has been changed between this two RS pictures?')]),
        # ('zh', [('<Img><ImageHere></Img> '
        #          '详细描述这两张图片。'), ('<Img><ImageHere></Img> '
        #                         '浏览这张图片并描述你注意到什么。'),
        #         ('<Img><ImageHere></Img> '
        #          '请对这张图片进行详细的描述。'),
        #         ('<Img><ImageHere></Img> '
        #          '你能为我描述这张图片的内容吗？')])
    ]),
    max_txt_len=120,
    #end_sym=' .',
    end_sym='###',
    neck=dict(
        type='SequentialNeck',
        necks=[
            dict(
                type='FeatureFusionNeck',
                policy='concat',
                out_indices=(0,)),
            dict(
                type='SimpleFPN',
                backbone_channel=512,
                in_channels=[128, 256, 512, 512],
                out_channels=256,
                num_outs=5,
                norm_cfg=fpn_norm_cfg),
        ],
    ),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(256, 256),
        in_channels=[256]*5,
        in_index=[0, 1, 2, 3, 4],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2))
    )

# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=5,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=epoch,val_interval=6)
val_cfg = dict()
test_cfg = dict()