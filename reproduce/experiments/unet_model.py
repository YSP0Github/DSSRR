"""
P0-B: 1-D U-Net 数据驱动基线（自监督掩码重建）

- 架构：轻量 1-D U-Net（~3 层编解码，GroupNorm），输入 [corrupted, mask] 两通道，
  输出修复波形单通道。
- 训练：在 E1 合成信号族上生成随机补丁，随机注入 6 类异常（与 E1 同分布），
  以掩码重建（inpainting）目标训练。真实数据零标注、零微调，直接跨域推理。
- 输出模型：experiments/models/unet_synth.pt（state_dict + 配置）
"""

from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from docs.dssrr_paper.experiments.E1_synthetic import FAMILIES  # noqa: E402

PATCH = 1536          # 训练补丁长度（8 的倍数）
BATCH = 16
EPOCHS = 30
N_TRAIN = 6000
N_VAL = 200
LR = 1e-3
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "unet_synth.pt")


class UNet1D(nn.Module):
    """1-D U-Net，输入长度须为 8 的倍数。"""

    def __init__(self, in_ch=2, base=16):
        super().__init__()
        def block(i, o, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=p),
                nn.GroupNorm(min(8, o), o),
                nn.ReLU(inplace=True),
            )
        self.enc0 = block(in_ch, base, k=7, p=3)
        self.down1 = nn.Sequential(block(base, base * 2, k=4, s=2, p=1),
                                   block(base * 2, base * 2))
        self.down2 = nn.Sequential(block(base * 2, base * 4, k=4, s=2, p=1),
                                   block(base * 4, base * 4))
        self.down3 = nn.Sequential(block(base * 4, base * 8, k=4, s=2, p=1),
                                   block(base * 8, base * 8))
        self.bottleneck = block(base * 8, base * 8)
        self.up3 = nn.ConvTranspose1d(base * 8, base * 4, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(block(base * 8, base * 4), block(base * 4, base * 4))
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(block(base * 4, base * 2), block(base * 2, base * 2))
        self.up1 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(block(base * 2, base), block(base, base))
        self.out = nn.Conv1d(base, 1, 7, padding=3)

    def forward(self, x):
        e0 = self.enc0(x)
        d1 = self.down1(e0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        b = self.bottleneck(d3)
        u3 = self.up3(b)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.dec3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.dec2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, e0], dim=1)
        u1 = self.dec1(u1)
        return self.out(u1)


# ----------------------------------------------------------------------
# 训练数据生成（与 E1 同分布但种子不同，避免与测试实例重叠）
# ----------------------------------------------------------------------

def corrupt_patch(clean: np.ndarray, anom: str, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """返回 (corrupted, mask)；mask=1 已知，0 为注入异常区。"""
    x = clean.copy()
    n = len(x)
    mask = np.ones(n, dtype=np.float32)
    std = float(np.std(x)) + 1e-12
    if anom == "spike":
        i = int(rng.integers(10, n - 10))
        x[i] += rng.uniform(10, 100) * std * (1.0 if rng.random() < 0.5 else -1.0)
        mask[i] = 0.0
    elif anom == "dropout_short":
        ln = int(n * rng.uniform(0.05, 0.15))
        s = int(rng.integers(10, n - ln - 10))
        x[s:s + ln] = 0.0
        mask[s:s + ln] = 0.0
    elif anom == "dropout_long":
        ln = int(n * rng.uniform(0.20, 0.40))
        s = int(rng.integers(10, n - ln - 10))
        x[s:s + ln] = 0.0
        mask[s:s + ln] = 0.0
    elif anom == "noise_burst":
        ln = int(n * rng.uniform(0.03, 0.10))
        s = int(rng.integers(10, n - ln - 10))
        x[s:s + ln] += rng.standard_normal(ln) * std * rng.uniform(5, 10)
        mask[s:s + ln] = 0.0
    elif anom == "step":
        ln = int(n * rng.uniform(0.10, 0.25))
        s = int(rng.integers(10, n - ln - 10))
        x[s:s + ln] += rng.uniform(2, 5) * std * (1.0 if rng.random() < 0.5 else -1.0)
        mask[s:s + ln] = 0.0
    elif anom == "saturation":
        ln = int(n * rng.uniform(0.10, 0.25))
        s = int(rng.integers(10, n - ln - 10))
        x[s:s + ln] = float(np.median(x[s:s + ln]))
        mask[s:s + ln] = 0.0
    return x, mask


ANOM_LIST = ["spike", "dropout_short", "dropout_long",
             "noise_burst", "step", "saturation"]


def make_dataset(n_patches: int, rng: np.random.Generator):
    xs, ys, ms = [], [], []
    fam_names = list(FAMILIES)
    for i in range(n_patches):
        fam = fam_names[i % len(fam_names)]
        seed = int(zlib.crc32(f"unet_trn|{fam}|{i // len(fam_names)}".encode()))
        sig = FAMILIES[fam](seed)
        start = int(rng.integers(0, len(sig) - PATCH))
        clean = sig[start:start + PATCH].astype(np.float32)
        anom = ANOM_LIST[int(rng.integers(0, len(ANOM_LIST)))]
        corr, mask = corrupt_patch(clean, anom, rng)
        # 按已知区统计归一化（推理时同口径）
        known = corr[mask > 0]
        mu = float(np.mean(known)); sd = float(np.std(known)) + 1e-8
        x = (corr - mu) / sd
        y = (clean - mu) / sd
        xs.append(x); ys.append(y); ms.append(mask)
    return (np.stack(xs)[:, None, :], np.stack(ys)[:, None, :],
            np.stack(ms)[:, None, :])


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(0)
    np.random.seed(0)

    rng = np.random.default_rng(0)
    x_tr, y_tr, m_tr = make_dataset(N_TRAIN, rng)
    x_va, y_va, m_va = make_dataset(N_VAL, rng)

    model = UNet1D()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    n_tr = len(x_tr)
    best_val = float("inf")
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n_tr)
        tot = 0.0; nb = 0
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]
            xb = torch.from_numpy(x_tr[idx]); yb = torch.from_numpy(y_tr[idx])
            mb = torch.from_numpy(m_tr[idx])
            inp = torch.cat([xb, mb], dim=1)
            out = model(inp)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx); nb += len(idx)
        sched.step()
        # 验证
        model.eval()
        with torch.no_grad():
            va_loss = 0.0; nv = 0
            for i in range(0, len(x_va), BATCH):
                xb = torch.from_numpy(x_va[i:i + BATCH])
                yb = torch.from_numpy(y_va[i:i + BATCH])
                mb = torch.from_numpy(m_va[i:i + BATCH])
                inp = torch.cat([xb, mb], dim=1)
                va_loss += float(loss_fn(model(inp), yb)) * len(xb); nv += len(xb)
            va_loss /= nv
        print(f"[unet] epoch {ep + 1}/{EPOCHS} train={tot / nb:.5f} val={va_loss:.5f}", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": model.state_dict(),
                        "config": {"patch": PATCH, "base": 16}},
                       MODEL_PATH)
    print(f"[unet] done. best val MSE={best_val:.5f} -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
