import os.path as osp
import numpy as np

import mmcv

from mmseg.apis import inference_segmentor, init_segmentor

print("hello world")



def test_test_time_augmentation_on_cpu():
    config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config = mmcv.Config.fromfile(config_file)

    # Remove pretrain model download for testing
    config.model.pretrained = None
    # Replace SyncBN with BN to inference on CPU
    norm_cfg = dict(type='BN', requires_grad=True)
    config.model.backbone.norm_cfg = norm_cfg
    config.model.decode_head.norm_cfg = norm_cfg
    config.model.auxiliary_head.norm_cfg = norm_cfg
    config.model.backbone.in_channels = 6

    # Enable test time augmentation
    # config.data.test.pipeline[1].flip = True
    # config.img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375], to_rgb=False)
    config.img_norm_cfg = dict(to_rgb=True)
    config.data.test.pipeline[1].flip = True
    print(config.data.test.pipeline[1].transforms[2])
    config.data.test.pipeline[1].transforms[2].to_rgb = False
    # config.test_pipeline = [
    #     dict(
    #         type='MultiScaleFlipAug',
    #         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #         flip=True,
    #         transforms=[
    #             dict(type='Normalize', **config.img_norm_cfg),
    #             dict(type='ImageToTensor', keys=['img']),
    #             dict(type='Collect', keys=['img'])
    #         ])
    # ]

    config.data.test.pipeline = config.test_pipeline

    print(f'Config:\n{config.pretty_text}')

    checkpoint_file = None
    model = init_segmentor(config, checkpoint_file, device='cpu')
    inputs_1 = np.random.randint(0, 255, (3, 288, 512), dtype=np.uint8)
    inputs_1 = np.expand_dims(inputs_1, axis=0)
    inputs_2 = np.random.randint(0, 255, (288, 512, 6), dtype=np.uint8)

    print(inputs_2.shape)

    img = mmcv.imread(
        osp.join(osp.dirname(__file__), 'tests/data/color.jpg'), 'color')
    print(img.shape)
    print(type(img), type(inputs_2))
    result = inference_segmentor(model, inputs_2)
    # assert result[0].shape == (288, 512)

    # print(f'Config:\n{config.pretty_text}')
    print("after")

test_test_time_augmentation_on_cpu()
