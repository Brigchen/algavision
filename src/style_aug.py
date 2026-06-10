# -*- coding: utf-8 -*-
"""
style_aug.py — 跨域风格迁移数据增强

解决场景：
    少量"标准照"（书籍/网站收集的高质量显微图）+ 大量目标域图像（FlowCAM /
    自动显微镜，成像风格各异）→ 需要把标准照"变形"成接近目标域风格，
    实现低成本域适应数据增强，提升识别模型在真实样品上的泛化能力。

提供三种风格化策略（均无需额外训练）：
    1. FDA（Fourier Domain Adaptation）
       交换源图与风格图的低频幅度分量。计算快，效果自然，推荐首选。
    2. HistMatch（直方图匹配）
       逐通道匹配源图像到风格图的像素强度分布。适合整体曝光/色调迁移。
    3. AdaIN（Adaptive Instance Normalization，统计对齐）
       将源图特征的均值/标准差对齐到风格图。最轻量，适合批量增强。

核心类：
    StylePool          — 收集并管理风格参考图像池
    StyleTransform     — torchvision-compatible PIL→PIL 风格化 transform
    MicroscopyAugment  — 专为显微成像设计的增强流水线（含风格迁移 + 旋转 + 模糊等）

@author: brigchen@xmu.edu.cn
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ---------------------------------------------------------------------------
# 1. 基础风格化算法（PIL Image → PIL Image，纯 numpy/scipy，无 torch 依赖）
# ---------------------------------------------------------------------------

def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0


def _from_float(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), 'RGB')


def fda_transfer(source: Image.Image, style: Image.Image,
                 beta: float = 0.01) -> Image.Image:
    """Fourier Domain Adaptation（Yang et al. 2020）。

    交换 source 与 style 在低频带（比例 beta）的幅度谱，保留 source 相位
    （即内容/形态），迁移 style 的整体照明/纹理风格。

    Args:
        source: 内容图像（标准照），将被风格化。
        style:  风格参考图（目标域图像，如 FlowCAM 样品）。
        beta:   低频截断比例，0.01~0.1 为常用范围；越大迁移越剧烈。
    """
    src_size = source.size  # (W, H)
    style_r = style.resize(src_size, Image.BILINEAR)

    src_arr  = _to_float(source)    # (H, W, 3)
    sty_arr  = _to_float(style_r)

    out = np.zeros_like(src_arr)
    H, W = src_arr.shape[:2]
    h_cut = max(1, int(H * beta))
    w_cut = max(1, int(W * beta))
    cy, cx = H // 2, W // 2

    for c in range(3):
        src_f  = np.fft.fftshift(np.fft.fft2(src_arr[:, :, c]))
        sty_f  = np.fft.fftshift(np.fft.fft2(sty_arr[:, :, c]))

        src_amp = np.abs(src_f)
        src_pha = np.angle(src_f)
        sty_amp = np.abs(sty_f)

        # 用风格图低频幅度替换源图低频幅度
        mixed_amp = src_amp.copy()
        mixed_amp[cy - h_cut: cy + h_cut,
                  cx - w_cut: cx + w_cut] = \
            sty_amp[cy - h_cut: cy + h_cut,
                    cx - w_cut: cx + w_cut]

        recon = np.fft.ifft2(np.fft.ifftshift(mixed_amp * np.exp(1j * src_pha)))
        out[:, :, c] = np.real(recon)

    return _from_float(out)


def histogram_match(source: Image.Image, style: Image.Image) -> Image.Image:
    """逐通道直方图匹配：将 source 的像素分布对齐到 style。

    适合校正整体曝光差异（如明视野 vs 荧光）。
    """
    src  = np.asarray(source.convert('RGB'), dtype=np.float32)
    sty  = np.asarray(style.convert('RGB'),  dtype=np.float32)
    out  = np.zeros_like(src)

    for c in range(3):
        s_vals, s_idx, s_cnt = np.unique(
            src[:, :, c].ravel(), return_inverse=True, return_counts=True)
        t_vals, t_cnt = np.unique(sty[:, :, c].ravel(), return_counts=True)

        s_cdf = np.cumsum(s_cnt).astype(np.float64)
        s_cdf /= s_cdf[-1]
        t_cdf = np.cumsum(t_cnt).astype(np.float64)
        t_cdf /= t_cdf[-1]

        # 最近邻匹配 CDF
        interp = np.interp(s_cdf, t_cdf, t_vals)
        out[:, :, c] = interp[s_idx].reshape(src.shape[:2])

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGB')


def adain_transfer(source: Image.Image, style: Image.Image) -> Image.Image:
    """AdaIN 统计对齐：将 source 的均值/标准差对齐到 style，逐通道操作。

    最轻量，适合对大批量标准照做快速全局风格化预处理。
    """
    src = _to_float(source)
    sty = _to_float(style)
    out = np.zeros_like(src)
    eps = 1e-5
    for c in range(3):
        s_mu, s_std = src[:, :, c].mean(), src[:, :, c].std() + eps
        t_mu, t_std = sty[:, :, c].mean(), sty[:, :, c].std() + eps
        out[:, :, c] = (src[:, :, c] - s_mu) / s_std * t_std + t_mu
    return _from_float(out)


# ---------------------------------------------------------------------------
# 2. StylePool — 风格参考图像池管理
# ---------------------------------------------------------------------------

class StylePool:
    """管理目标域风格参考图像池（FlowCAM / 自动显微镜采集图）。

    典型用法：
        pool = StylePool.from_folder('./field_images', max_images=300)
        style_img = pool.sample()
    """

    def __init__(self, images: List[Image.Image]):
        self._images = images

    def __len__(self):
        return len(self._images)

    @classmethod
    def from_folder(cls, folder: str | Path,
                    max_images: int = 300,
                    recursive: bool = True) -> 'StylePool':
        """从目录加载风格参考图像，懒加载为 PIL Image 列表。

        只加载路径，不一次性全部读入内存（read_lazy=True 默认）。
        """
        folder = Path(folder)
        if recursive:
            paths = [p for p in folder.rglob('*')
                     if p.suffix.lower() in _IMG_EXTS]
        else:
            paths = [p for p in folder.iterdir()
                     if p.suffix.lower() in _IMG_EXTS]

        if not paths:
            raise ValueError(f"风格来源目录中未找到图像: {folder}")

        random.shuffle(paths)
        paths = paths[:max_images]
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert('RGB'))
            except Exception:
                pass
        if not images:
            raise ValueError(f"无法读取任何风格参考图: {folder}")
        return cls(images)

    def sample(self, k: int = 1) -> Image.Image | List[Image.Image]:
        """随机采样一张（k=1）或多张风格图。"""
        sampled = random.choices(self._images, k=k)
        return sampled[0] if k == 1 else sampled


# ---------------------------------------------------------------------------
# 3. StyleTransform — torchvision-compatible PIL→PIL transform
# ---------------------------------------------------------------------------

_METHODS = ('fda', 'histmatch', 'adain')

class StyleTransform:
    """随机从风格池中取一张参考图，对输入 PIL Image 做风格迁移。

    设计为 torchvision.transforms.Compose 中的一个环节（PIL→PIL）。
    应放在 Resize 之后、ToTensor 之前。

    Args:
        style_pool: StylePool 实例。
        method:     'fda' | 'histmatch' | 'adain' | 'random'（随机选一种）。
        p:          应用概率（p=1.0 每张都迁移，p=0.5 随机一半）。
        beta:       FDA 低频截断比例。
    """

    def __init__(self, style_pool: StylePool,
                 method: str = 'fda',
                 p: float = 0.8,
                 beta: float = 0.01):
        if method not in (*_METHODS, 'random'):
            raise ValueError(f"method 须为 {(*_METHODS, 'random')}")
        self.pool   = style_pool
        self.method = method
        self.p      = p
        self.beta   = beta

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        style = self.pool.sample()
        m = random.choice(_METHODS) if self.method == 'random' else self.method
        try:
            if m == 'fda':
                return fda_transfer(img, style, beta=self.beta)
            elif m == 'histmatch':
                return histogram_match(img, style)
            else:
                return adain_transfer(img, style)
        except Exception:
            return img  # 任何失败静默回退到原图

    def __repr__(self):
        return (f"StyleTransform(method={self.method}, p={self.p}, "
                f"pool_size={len(self.pool)})")


# ---------------------------------------------------------------------------
# 4. MicroscopyAugment — 专为显微/FlowCAM 成像设计的增强流水线
# ---------------------------------------------------------------------------

class MicroscopyAugment:
    """针对浮游生物显微成像场景的专用增强策略（PIL→PIL）。

    相比通用 ImageNet 增强的区别：
    - 全角度旋转（浮游生物朝向任意）
    - 关闭水平翻转偏置（显微图无语义翻转含义）
    - CLAHE 对比度增强（模拟不同曝光/聚焦条件）
    - 轻微高斯模糊（模拟焦深不足）
    - 可选随机噪声（模拟 CCD/CMOS 噪声）
    - 风格迁移（可选，需提供 style_pool）
    """

    def __init__(self,
                 img_size: int = 224,
                 style_pool: Optional[StylePool] = None,
                 style_method: str = 'fda',
                 style_p: float = 0.7,
                 style_beta: float = 0.01,
                 rotation_degrees: float = 360.0,
                 blur_p: float = 0.2,
                 noise_p: float = 0.1,
                 clahe_p: float = 0.3,
                 color_jitter_strength: float = 0.2):
        self.img_size = img_size
        self.style_transform = (
            StyleTransform(style_pool, method=style_method,
                           p=style_p, beta=style_beta)
            if style_pool is not None else None
        )
        self.rotation_degrees = rotation_degrees
        self.blur_p           = blur_p
        self.noise_p          = noise_p
        self.clahe_p          = clahe_p
        self.cj_strength      = color_jitter_strength

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. 先做风格迁移（在 resize 之前，保留更多风格信息）
        if self.style_transform is not None:
            img = self.style_transform(img)

        # 2. Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 3. 全角度随机旋转（浮游生物朝向任意）
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees / 2,
                                    self.rotation_degrees / 2)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 4. 随机垂直/水平翻转（显微图无语义方向，两者都合理）
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 5. CLAHE-like 对比度增强（用 PIL autocontrast 近似）
        if random.random() < self.clahe_p:
            from PIL import ImageOps
            img = ImageOps.autocontrast(img, cutoff=random.uniform(0, 2))

        # 6. 颜色抖动（亮度/对比度）
        if self.cj_strength > 0:
            from PIL import ImageEnhance
            s = self.cj_strength
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(1 - s, 1 + s))
            img = ImageEnhance.Contrast(img).enhance(
                random.uniform(1 - s, 1 + s))

        # 7. 轻微高斯模糊（模拟焦深不足/运动模糊）
        if random.random() < self.blur_p:
            radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 8. 随机加性高斯噪声（模拟传感器噪声）
        if random.random() < self.noise_p:
            arr = np.asarray(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def __repr__(self):
        style_info = repr(self.style_transform) if self.style_transform else "None"
        return (f"MicroscopyAugment(img_size={self.img_size}, "
                f"rotation={self.rotation_degrees}°, "
                f"style={style_info})")


# ---------------------------------------------------------------------------
# 5. 便捷工厂函数
# ---------------------------------------------------------------------------

def build_microscopy_transform(img_size: int,
                                style_folder: Optional[str] = None,
                                style_method: str = 'fda',
                                style_p: float = 0.7,
                                style_beta: float = 0.01,
                                max_style_images: int = 300,
                                mode: str = 'train') -> 'callable':
    """构建适用于浮游生物显微成像的完整 transform 流水线。

    返回 torchvision.transforms.Compose，可直接替换 _create_transform 结果。

    Args:
        img_size:     目标分辨率。
        style_folder: 风格参考图（目标域）目录。为 None 则不做风格迁移。
        style_method: 'fda' / 'histmatch' / 'adain' / 'random'。
        style_p:      风格迁移概率。
        style_beta:   FDA beta 参数（0.01~0.1）。
        max_style_images: 风格池最大图像数。
        mode:         'train'（含增强）或 'val'（仅 resize+normalize）。
    """
    from torchvision import transforms

    _NORM_MEAN = [0.485, 0.456, 0.406]
    _NORM_STD  = [0.229, 0.224, 0.225]

    if mode != 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ])

    style_pool = None
    if style_folder:
        try:
            style_pool = StylePool.from_folder(
                style_folder, max_images=max_style_images)
            print(f"[StyleAug] 风格池加载完成: {len(style_pool)} 张参考图 "
                  f"← {style_folder}")
        except Exception as e:
            print(f"[StyleAug] 风格池加载失败（跳过风格迁移）: {e}")

    micro_aug = MicroscopyAugment(
        img_size=img_size,
        style_pool=style_pool,
        style_method=style_method,
        style_p=style_p,
        style_beta=style_beta,
    )

    return transforms.Compose([
        micro_aug,           # PIL → PIL（含 resize、旋转、风格迁移）
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, tempfile, os
    src_path = sys.argv[1] if len(sys.argv) > 1 else None
    sty_path = sys.argv[2] if len(sys.argv) > 2 else None

    # 若无命令行参数，生成合成测试图
    if src_path is None:
        tmp = Path(tempfile.mkdtemp())
        src_img = Image.fromarray(
            np.random.randint(180, 230, (128, 128, 3), dtype=np.uint8))
        sty_img = Image.fromarray(
            np.random.randint(30, 80,  (96,  96,  3), dtype=np.uint8))
        src_img.save(tmp / 'src.png')
        sty_img.save(tmp / 'sty.png')
        src_path = str(tmp / 'src.png')
        sty_path = str(tmp / 'sty.png')
        print(f"生成合成测试图: {tmp}")

    src = Image.open(src_path).convert('RGB')
    sty = Image.open(sty_path).convert('RGB')

    out_fda   = fda_transfer(src, sty, beta=0.02)
    out_hist  = histogram_match(src, sty)
    out_adain = adain_transfer(src, sty)

    print(f"FDA     输出尺寸: {out_fda.size},   像素范围: {np.array(out_fda).min()}-{np.array(out_fda).max()}")
    print(f"HistMatch 输出: {out_hist.size},   像素范围: {np.array(out_hist).min()}-{np.array(out_hist).max()}")
    print(f"AdaIN    输出: {out_adain.size},   像素范围: {np.array(out_adain).min()}-{np.array(out_adain).max()}")

    pool = StylePool([sty] * 5)
    tf   = StyleTransform(pool, method='random', p=1.0)
    out  = tf(src)
    print(f"StyleTransform 输出: {out.size}")

    micro = MicroscopyAugment(img_size=224, style_pool=pool)
    out_m = micro(src)
    print(f"MicroscopyAugment 输出: {out_m.size}")
    print("STYLE_AUG_SMOKE_OK")
