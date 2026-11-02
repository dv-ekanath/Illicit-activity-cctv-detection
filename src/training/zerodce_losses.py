"""
zerodce_losses.py
=================
Unsupervised losses for Zero-DCE++ with exact weights per project spec:
    W_spa = 1.0
    W_exp = 10.0
    W_col = 0.5
    W_tv  = 200.0
    E     = 0.5  (surveillance target exposure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss Components ────────────────────────────────────────────────────────────

class SpatialConsistencyLoss(nn.Module):
    """Preserves local spatial structure between input and enhanced image."""

    def __init__(self):
        super().__init__()
        kernel_left  = torch.FloatTensor([[0, 0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0,-1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0, 0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('w_left',  kernel_left)
        self.register_buffer('w_right', kernel_right)
        self.register_buffer('w_up',    kernel_up)
        self.register_buffer('w_down',  kernel_down)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhanced):
        org_gray = torch.mean(org,      1, keepdim=True)
        enh_gray = torch.mean(enhanced, 1, keepdim=True)
        org_p    = self.pool(org_gray)
        enh_p    = self.pool(enh_gray)

        loss = sum(
            torch.mean(torch.pow(
                F.conv2d(org_p, w, padding=1) - F.conv2d(enh_p, w, padding=1), 2
            ))
            for w in [self.w_left, self.w_right, self.w_up, self.w_down]
        )
        return loss


class ExposureControlLoss(nn.Module):
    """
    Drives mean patch intensity toward target E.
    E = 0.5 for surveillance (not 0.6) — avoids over-brightening.
    """

    def __init__(self, patch_size=16, well_exposure=0.5):
        super().__init__()
        self.pool          = nn.AvgPool2d(patch_size)
        self.well_exposure = well_exposure

    def forward(self, enhanced):
        mean = self.pool(torch.mean(enhanced, 1, keepdim=True))
        return torch.mean(torch.pow(mean - self.well_exposure, 2))


class ColorConstancyLoss(nn.Module):
    """Gray-world: R, G, B channel means should be equal after enhancement."""

    def forward(self, enhanced):
        r = torch.mean(enhanced[:, 0, :, :])
        g = torch.mean(enhanced[:, 1, :, :])
        b = torch.mean(enhanced[:, 2, :, :])
        return torch.pow(r - g, 2) + torch.pow(r - b, 2) + torch.pow(g - b, 2)


class IlluminationSmoothnessLoss(nn.Module):
    """Total variation on curve maps — keeps enhancement spatially smooth."""

    def forward(self, alphas):
        loss = 0.0
        for alpha in alphas:
            b, c, h, w = alpha.size()
            h_tv = torch.pow(alpha[:, :, 1:, :] - alpha[:, :, :h-1, :], 2).sum()
            w_tv = torch.pow(alpha[:, :, :, 1:] - alpha[:, :, :, :w-1], 2).sum()
            loss += 2 * (h_tv / ((h-1)*w) + w_tv / (h*(w-1))) / b
        return loss / len(alphas)


# ── Combined Loss ──────────────────────────────────────────────────────────────

class ZeroDCELoss(nn.Module):
    """
    Standard 4-loss unsupervised objective.
    Used in Stage 1 (ExDARK) and as the enhancement component in Stage 2.

    Exact weights per project spec:
        W_spa = 1.0,  W_exp = 10.0,  W_col = 0.5,  W_tv = 200.0
        E = 0.5
    """

    def __init__(self):
        super().__init__()
        self.spatial      = SpatialConsistencyLoss()
        self.exposure     = ExposureControlLoss(patch_size=16, well_exposure=0.5)
        self.color        = ColorConstancyLoss()
        self.illumination = IlluminationSmoothnessLoss()

        # Exact weights from project spec
        self.W_spa = 1.0
        self.W_exp = 10.0
        self.W_col = 0.5
        self.W_tv  = 200.0

    def forward(self, original, enhanced, alphas):
        l_spa = self.spatial(original, enhanced)
        l_exp = self.exposure(enhanced)
        l_col = self.color(enhanced)
        l_tv  = self.illumination(alphas)

        total = (
            self.W_spa * l_spa +
            self.W_exp * l_exp +
            self.W_col * l_col +
            self.W_tv  * l_tv
        )

        breakdown = {
            'spatial':      l_spa.item(),
            'exposure':     l_exp.item(),
            'color':        l_col.item(),
            'illumination': l_tv.item(),
            'total':        total.item(),
        }
        return total, breakdown