# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class FusedDensityHead(BaseModule):


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
                 loss_seg=dict(type='CrossEntropyLoss',loss_weight=0.2),
                 loss_center=dict(type='CrossEntropyLoss',loss_weight=0.2),
                 loss_hw=dict(type='CrossEntropyLoss',loss_weight=0.2),
                 init_cfg=dict(
                     type='Kaiming', override=dict(name='conv_logits'))):
        super(FusedDensityHead, self).__init__(init_cfg)
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        dilations=[1,1,1]
        self.lateral_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.convs_dilation = nn.ModuleList()
        self.convs_dilation_offset = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.decode_convs = nn.ModuleList()
        self.fpn_cg_conv = nn.ModuleList()
        Pool = nn.MaxPool2d
        self.pool = Pool(kernel_size = 2,stride=2)
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))

        self.skip_convs.append(
            ConvModule(
                self.in_channels,
                self.in_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))
        #print(self.num_convs)
        for i in range(self.num_ins - 1):
            self.skip_convs.append(
                ConvModule(
                    self.in_channels * 2,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))


        for i in range(3):
            self.convs_dilation.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    padding=dilations[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    dilation=dilations[i]))

        self.mid_convs.append(
             ConvModule(
                self.in_channels * 2,
                self.in_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))
        for i in range(3):
            self.mid_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

    #    self.fpn_cg_conv.append(
    #        ConvModule(
    #            self.in_channels,
    #            self.in_channels,
    #            3,
    #            padding=1,
    #            conv_cfg=self.conv_cfg,
    #            norm_cfg=self.norm_cfg))
    #    for i in range(4):
    #        self.fpn_cg_conv.append(
    #            ConvModule(
    #                self.in_channels,
    #                self.in_channels,
    #                3,
    #                padding=1,
    #                conv_cfg=self.conv_cfg,
    #                norm_cfg=self.norm_cfg))



        for i in range(self.num_ins - 1):
            self.decode_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))


        self.convs_downsample = nn.ModuleList()
        self.convs_downsample.append(
            ConvModule(
                in_channels,
                conv_out_channels,
                3,
                stride=(2,2),
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg))
        for i in range(self.num_convs-1):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs_downsample.append(
                ConvModule(
                    in_channels * 2,
                    conv_out_channels,
                    3,
                    padding=1,
                    stride=2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.plus_para = nn.Parameter(torch.FloatTensor([0.0,0.0,0.0]))
        self.fusion_para = nn.Parameter(torch.FloatTensor([1.0,1.0,1.0,1.0]))
        self.conv_embedding=ConvModule(
                conv_out_channels,
                conv_out_channels,
                1,
                #padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
#            self.conv_embedding =ConvModule(
#                    conv_out_channels,
#                    conv_out_channels,
#                    1,
#                    conv_cfg=self.conv_cfg,
#                    norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        self.conv_HW = nn.Conv2d(conv_out_channels, 2, 1)
        if ignore_label:
            loss_seg['ignore_index'] = ignore_label
        if loss_weight:
            loss_seg['loss_weight'] = loss_weight
        if ignore_label or loss_weight:
            warnings.warn('``ignore_label`` and ``loss_weight`` would be '
                          'deprecated soon. Please set ``ingore_index`` and '
                          '``loss_weight`` in ``loss_seg`` instead.')
        self.criterion = build_loss(loss_seg)
        self.center_loss = build_loss(loss_center)
        self.hw_loss = build_loss(loss_hw)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    @auto_fp16()
    def forward(self, feats):
        from .feature_visualization import draw_feature_map
        
        x = []
        skip = []
        x.append(feats[0])
        #print(self.convs[0])
        skip.append(x[0])
        for i in range(4):
            x.append(torch.cat([self.convs_downsample[i](x[i]),feats[i+1]],1))
            skip.append(x[i+1])
            #print(x[i+1].shape)
        for i in range(4):
            skip[i] = self.skip_convs[i](skip[i])
            #print(x[i].shape)
        res = self.mid_convs[2](self.mid_convs[1](self.mid_convs[0](x[4])))
        for i in range(4):
            res = self.up(res)
            #print(res.shape)
            #print(skip[3-i].shape)
            res += skip[3-i] * self.fusion_para[i]
            if i < 3:
                res += self.pool(skip[2-i]) * self.plus_para[i]
                #print(self.plus_para[i])
            res = self.decode_convs[i](res)

        for i in range(3):
            res = self.convs_dilation[i](res)

        heat_pred = self.conv_logits(res)
        HW_pred = self.conv_HW(res)
        return heat_pred, HW_pred
#        return heat_pred



    def forward_00(self, feats):
        #print("asdasdasdasdasdasdasdasda")
        #from .feature_visualization import draw_feature_map
        #draw_feature_map(feats)
       # print(len(feats))
        #print(feats[0].shape)
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])

        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x = x + self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)

        mask_pred = self.conv_logits(x)
        HW_pred = self.conv_HW(x)
        x = self.conv_embedding(x)
        return mask_pred, HW_pred






    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):

        labels = labels.unsqueeze(1).long()

        #print(mask_pred.shape)
        #print(labels.shape)
        loss_semantic_seg = self.criterion(mask_pred, labels)
        return loss_semantic_seg

    @force_fp32(apply_to=('mask_pred', ))
    def loss_center(self, mask_pred, labels):

        labels = labels.unsqueeze(1).long()

        #print(mask_pred.shape)
        #print(labels.shape)
        loss_semantic_seg = self.center_loss(mask_pred, labels)
        return loss_semantic_seg

    @force_fp32(apply_to=('mask_pred', ))
    def loss_hw(self, mask_pred, labels):

        labels = labels.transpose(1,3).long()
        #print(torch.ne(labels, 0))
        mask_pred = mask_pred * torch.ne(labels, 0)
        #print(labels.shape)
        #print(mask_pred.shape)
        loss_semantic_seg = self.hw_loss(mask_pred, labels)
        return loss_semantic_seg
