# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import numpy as np
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class FusedRPN2Head(BaseModule):
    r"""Multi-level fused semantic segmentation head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=183,
                 conv_cfg=None,
                 norm_cfg=None,
                 ignore_label=None,
                 loss_weight=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=0.2),
                 init_cfg=dict(
                     type='Kaiming', override=dict(name='conv_logits'))):
        super(FusedRPN2Head, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        dilations=[3,5,8,3,5,8]
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    dilation = dilations[i]))
        self.conv_embedding=ConvModule(
                conv_out_channels,
                conv_out_channels,
                1,
                #padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_0=ConvModule(
                257,
                256,
                1,
              #  padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_1=ConvModule(
                257,
                256,
                1,
              #  padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_2=ConvModule(
                257,
                256,
                1,
              #  padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_3=ConvModule(
                257,
                256,
                1,
              #  padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.conv_4=ConvModule(
                257,
                256,
                1,
              #  padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.downsample_0=ConvModule(
                1,
                1,
                3,
                padding=1,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.downsample_1=ConvModule(
                1,
                1,
                3,
                padding=1,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.downsample_2=ConvModule(
                1,
                1,
                3,
                padding=1,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.downsample_3=ConvModule(
                1,
                1,
                3,
                padding=1,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
#            self.conv_embedding =ConvModule(
#                    conv_out_channels,
#                    conv_out_channels,
#                    1,
#                    conv_cfg=self.conv_cfg,
#                    norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        if ignore_label:
            loss_seg['ignore_index'] = ignore_label
        if loss_weight:
            loss_seg['loss_weight'] = loss_weight
        if ignore_label or loss_weight:
            warnings.warn('``ignore_label`` and ``loss_weight`` would be '
                          'deprecated soon. Please set ``ingore_index`` and '
                          '``loss_weight`` in ``loss_seg`` instead.')
        self.criterion = build_loss(loss_seg)

    @auto_fp16()
    def forward(self, x):
        return 0

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, proposal_list, density_pred):
        device = torch.device('cuda:0')
        proposal_map = np.zeros((4,512,512))
        proposal_maps = torch.from_numpy(proposal_map).to(device).unsqueeze(1)
        #print(proposal_maps.shape)
        #print(density_pred.shape)
        loss_semantic_tmp = self.criterion(proposal_maps, density_pred)
       # print(proposal_list[0].long().shape)
 
        for i in range(len(proposal_list)):
            proposal = proposal_list[i].long()
            #print(proposal)
            for j in range(proposal.shape[0]):
                y = round(((proposal[j][0] + proposal[j][2])/2).item())
                x = round(((proposal[j][1] + proposal[j][3])/2).item())
                if x < 512 and y < 512:
                    proposal_map[i][x][y] =  255
        proposal_map = torch.from_numpy(proposal_map).to(device)
        #labels = labels.squeeze(1).long()
        #print(labels.shape)
        #print("\n\n\n\n")
        #print(mask_pred.shape)
        proposal_map = proposal_map.unsqueeze(1)
        #print(proposal_map.shape)
        #print(density_pred.shape)

        loss_semantic_seg = self.criterion(proposal_map, density_pred)
        return loss_semantic_seg - loss_semantic_tmp
