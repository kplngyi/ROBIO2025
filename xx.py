import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import init
from regex import F
from torch.nn import init

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    CombinedConv,
    Ensure4d,
    Expression,
    SafeLog,
    SqueezeFinalOutput,
)


# ---------------- Focal Loss ------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.gather(1, target.unsqueeze(1)).mean()

# ---------------- SE Layer -------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ------------ Modified ShallowFBCSPNet -------------
class ShallowFBCSPNetIM(EEGModuleMixin, nn.Sequential):
    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length="auto",
        conv_nonlin=nn.ELU,
        pool_mode="mean",
        activation_pool_nonlin=SafeLog,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.3,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if final_conv_length == "auto":
            assert self.n_times is not None

        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = activation_pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.add_module("ensuredims", Ensure4d())

        if self.split_first_layer:
            self.add_module("dimshuffle", Rearrange("batch C T 1 -> batch 1 T C"))
            self.add_module(
                "conv_time_spat",
                CombinedConv(
                    in_chans=self.n_chans,
                    n_filters_time=self.n_filters_time,
                    n_filters_spat=self.n_filters_spat,
                    filter_time_length=self.filter_time_length,
                    bias_time=True,
                    bias_spat=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.n_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )

        self.add_module("conv_nonlin_exp", self.conv_nonlin())
        self.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        self.add_module("pool_nonlin_exp", self.pool_nonlin())

        # ------------------ 插入注意力机制 ------------------
        self.add_module("se_block", SELayer(n_filters_conv))

        # ------------------ 增加一层卷积 -------------------
        self.add_module(
            "conv_deep",
            nn.Conv2d(
                n_filters_conv, n_filters_conv, kernel_size=(3, 1), padding=(1, 0), bias=False
            ),
        )
        self.add_module("conv_deep_bn", nn.BatchNorm2d(n_filters_conv))
        self.add_module("conv_deep_act", self.conv_nonlin())

        self.add_module("drop", nn.Dropout(p=self.drop_prob))

        self.eval()
        if self.final_conv_length == "auto":
            self.final_conv_length = self.get_output_shape()[2]

        # ------------ Final Classification Layer ---------------
        final = nn.Sequential()
        final.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv, self.n_outputs, kernel_size=(self.final_conv_length, 1), bias=True
            ),
        )
        final.add_module("squeeze", SqueezeFinalOutput())
        self.add_module("final_layer", final)

        # ------------------- 权重初始化 -------------------
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_time_spat.conv_time.weight, gain=1)
            init.constant_(self.conv_time_spat.conv_time.bias, 0)
            init.xavier_uniform_(self.conv_time_spat.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_time_spat.conv_spat.bias, 0)
        else:
            init.xavier_uniform_(self.conv_time.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_time.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.final_layer.conv_classifier.weight, gain=1)
        init.constant_(self.final_layer.conv_classifier.bias, 0)

        self.train()