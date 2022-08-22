# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'
        self.pool = nn.MaxPool2d(2,2)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_convs_b = nn.ModuleList()
        self.fpn_convs_o = nn.ModuleList()
        self.fpn_convs_o_2 = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            fpn_conv_b = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            fpn_conv_o = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_convs_b.append(fpn_conv_b)
            self.fpn_convs_o.append(fpn_conv_o)
        fpn_conv_o = ConvModule(
            out_channels + 2,
            out_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.fpn_convs_o_2.append(fpn_conv_o)
        for i in range(5):
            fpn_conv_o = ConvModule(
                out_channels*2 + 2,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs_o_2.append(fpn_conv_o)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward_back(self, inputs, density_pred=None, ct_pred=None):
        from .feature_visualization import draw_feature_map
        outs = inputs
        outs_new = []
        den = (density_pred / 255)

        if den.ndim == 3:
            den = den.unsqueeze(1)
#        outs_new.append(self.fpn_convs_o_2[0](torch.cat([outs[0],den],1)) + outs[0])
#        den = self.pool(den)
#        outs_new.append(self.fpn_convs_o_2[1](torch.cat([outs[1],den],1)) + outs[1])
#        den = self.pool(den)
#        outs_new.append(self.fpn_convs_o_2[2](torch.cat([outs[2],den],1)) + outs[2])
#        den = self.pool(den)
#        outs_new.append(self.fpn_convs_o_2[3](torch.cat([outs[3],den],1)) + outs[3])
#        den = self.pool(den)
#        outs_new.append(self.fpn_convs_o_2[4](torch.cat([outs[4],den],1)) + outs[4])

        #draw_feature_map(ct)
       # if ct_pred:
        ct = (ct_pred / 255)

        outs_new.append(self.fpn_convs_o_2[0](torch.cat([outs[0], den, ct],1)) + outs[0])
        den = self.pool(den)
        ct = self.pool(ct)
        outs_new.append(self.fpn_convs_o_2[1](torch.cat([outs[1],self.pool(outs_new[0]),den, ct],1)) + outs[1])
        den = self.pool(den)
        ct = self.pool(ct)
        outs_new.append(self.fpn_convs_o_2[2](torch.cat([outs[2],self.pool(outs_new[1]),den, ct],1)) + outs[2])
        den = self.pool(den)
        ct = self.pool(ct)
        outs_new.append(self.fpn_convs_o_2[3](torch.cat([outs[3],self.pool(outs_new[2]),den, ct],1)) + outs[3])
        den = self.pool(den)
        ct = self.pool(ct)
        outs_new.append(self.fpn_convs_o_2[4](torch.cat([outs[4],self.pool(outs_new[3]),den, ct],1)) + outs[4])
       #else:
        #outs_new.append(self.fpn_convs_o_2[0](torch.cat([outs[0], den],1)) + outs[0])
        #den = self.pool(den)
        #outs_new.append(self.fpn_convs_o_2[1](torch.cat([outs[1],self.pool(outs_new[0]),den],1)) + outs[1])
        #den = self.pool(den)
        #outs_new.append(self.fpn_convs_o_2[2](torch.cat([outs[2],self.pool(outs_new[1]),den],1)) + outs[2])
        #den = self.pool(den)
        #outs_new.append(self.fpn_convs_o_2[3](torch.cat([outs[3],self.pool(outs_new[2]),den],1)) + outs[3])
        #den = self.pool(den)
        #outs_new.append(self.fpn_convs_o_2[4](torch.cat([outs[4],self.pool(outs_new[3]),den],1)) + outs[4])
        #draw_feature_map(outs_new[0])
        return tuple(outs_new)




    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
       # outs_new=[]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        #outs_new.append(outs[0])
        #print(self.fpn_convs_b[0])
        #outs_new.append(self.fpn_convs_o[0](self.fpn_convs_b[0](outs_new[0]) + outs[1]))

        #print(outs[2].shape)
        #outs_new.append(self.fpn_convs_o[1](self.fpn_convs_b[1](outs_new[1]) + outs[2]))
        #outs_new.append(self.fpn_convs_o[2](self.fpn_convs_b[2](outs_new[2]) + outs[3]))
        #outs_new.append(self.fpn_convs_o[3](self.fpn_convs_b[3](outs_new[3]) + outs[4]))
        from .feature_visualization import draw_feature_map
       # draw_feature_map(outs[0])
        return tuple(outs)
