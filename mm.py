import torch
import torch.nn as nn
from braindecode.models.base import EEGModuleMixin

class MyEnsure4d(nn.Module):
    def forward(self, x):
        # 如果是 3D: (B, C, T) → (B, 1, C, T)
        if x.ndim == 3:
            return x.unsqueeze(1)
        # 如果是 4D: 不变
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Expected input of 3 or 4 dimensions, got shape: {x.shape}")
        

class MySqueezeFinalOutput(nn.Module):
    def forward(self, x):
        # 自动去掉所有为1的维度
        return x.squeeze()
    
class Square(nn.Module):
    def forward(self, x):
        return x ** 2

class MyFirstEEGNet(EEGModuleMixin, nn.Sequential):
    def __init__(self, in_chans, input_window_samples, n_classes):
        super().__init__(
            MyEnsure4d(),  # 保证输入维度是 (B, 1, C, T)
            nn.Conv2d(1, 16, (1, 25), stride=1, padding=(0, 12)),  # 时域卷积
            nn.ReLU(),
            nn.Conv2d(16, 32, (in_chans, 1)),  # 空间卷积
            nn.BatchNorm2d(num_features=in_chans),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(self._calc_out_features(in_chans, input_window_samples), n_classes),
            MySqueezeFinalOutput()
        )

    def _calc_out_features(self, in_chans, input_window_samples):
        # 计算 Linear 层输入维度
        with torch.no_grad():
            x = torch.randn(1, 1, in_chans, input_window_samples)
            x = super().forward(x)
        return x.shape[1]
    