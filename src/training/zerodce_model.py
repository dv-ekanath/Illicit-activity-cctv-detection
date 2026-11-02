import torch
import torch.nn as nn


class ZeroDCE(nn.Module):
    """
    Zero-DCE++ — Fully convolutional curve estimation network.
    Trained at 256×256, runs at any resolution at inference.
    Input:  RGB image (B, 3, H, W) normalized to [0, 1]
    Output: (enhanced image, list of alpha curve maps)
    """

    def __init__(self, num_iterations=8):
        super(ZeroDCE, self).__init__()
        self.num_iterations = num_iterations

        self.conv1 = nn.Sequential(nn.Conv2d(3,  32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 3 * num_iterations, 3, padding=1, bias=True),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(torch.cat([f3, f4], dim=1))
        f6 = self.conv6(torch.cat([f2, f5], dim=1))
        curve_params = self.conv7(torch.cat([f1, f6], dim=1))

        alphas = torch.split(curve_params, 3, dim=1)
        enhanced = x
        for alpha in alphas:
            enhanced = enhanced + alpha * (enhanced - enhanced * enhanced)

        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced, alphas