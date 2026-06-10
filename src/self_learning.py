# -*- coding: utf-8 -*-
"""
self_learning.py — 自学习闭环：主动学习采样 + OOD 检测 + 伪/确认标签分层

替换 modler_v2 原有的"单一 softmax 阈值路由"，提供：
  - 多策略不确定性采样（entropy / margin / least-confidence）
  - MC-Dropout BALD 估计
  - Core-set 多样性采样（嵌入距离）
  - OOD 检测（Energy score / Mahalanobis 距离）
  - 确认(confirmed) vs 伪(pseudo) 标签两类样本管理
  - 一步式"批量路由"：直接替换 ActiveLearningGUI.batch_predict 中的分支逻辑

@author: brigchen@xmu.edu.cn
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# 1. 采样策略（Acquisition Functions）
# ---------------------------------------------------------------------------

def entropy_score(probs: np.ndarray) -> np.ndarray:
    """归一化香农熵：熵越高 = 模型越不确定。probs shape (N, C)"""
    p = np.clip(probs, 1e-9, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(probs.shape[1] + 1e-9)


def margin_score(probs: np.ndarray) -> np.ndarray:
    """最小边际分数：top2 差越小 = 越不确定（返回不确定度，即 1-margin）"""
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    return 1.0 - (sorted_p[:, 0] - sorted_p[:, 1])


def least_confidence_score(probs: np.ndarray) -> np.ndarray:
    """最小置信度：1 - max_prob"""
    return 1.0 - probs.max(axis=1)


def bald_score(probs_mc: np.ndarray) -> np.ndarray:
    """BALD（贝叶斯主动学习）：互信息近似。

    probs_mc shape: (N, T, C)，T 为 MC-Dropout 采样次数。
    返回 BALD score (N,)，值越高越值得标注。
    """
    mean_p = probs_mc.mean(axis=1)  # (N, C)
    h_mean = entropy_score(mean_p)
    mean_h = entropy_score(probs_mc.reshape(-1, probs_mc.shape[-1]))\
        .reshape(probs_mc.shape[0], probs_mc.shape[1]).mean(axis=1)
    return h_mean - mean_h


def coreset_score(embeddings: np.ndarray,
                  labeled_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
    """Core-Set 不确定度代理：到已标注集合最近邻的最大距离（越远越有价值）。

    若 labeled_embeddings 为 None（无已标注集），退化为到池中心的距离。
    """
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if labeled_embeddings is None or labeled_embeddings.shape[0] == 0:
        center = embeddings.mean(axis=0, keepdims=True)
        return np.linalg.norm(embeddings - center, axis=1).astype(np.float32)
    dists = _pairwise_min_dist(embeddings, labeled_embeddings)
    return dists.astype(np.float32)


def _pairwise_min_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (N,D) 到 B (M,D) 中最近邻的距离，避免 O(NM·D) 全矩阵（分块）。"""
    block = 512
    dists = np.full(A.shape[0], np.inf, dtype=np.float32)
    for start in range(0, B.shape[0], block):
        diff = A[:, None, :] - B[None, start:start + block, :]  # (N,b,D)
        d = np.linalg.norm(diff, axis=2).min(axis=1)           # (N,)
        dists = np.minimum(dists, d)
    return dists


# ---------------------------------------------------------------------------
# 2. OOD 检测
# ---------------------------------------------------------------------------

class OODDetector:
    """在线 OOD 检测器（不需要额外训练，仅用模型激活）。

    支持两种分数：
        energy  : -log(sum(exp(logits)))，无需 softmax，对 OOD 更敏感。
        mahal   : Mahalanobis 距离（需先用 fit() 估计各类协方差）。

    典型用法：
        det = OODDetector(method='energy', threshold=None)
        det.fit(train_loader, model, device)   # 可选，energy 不需要
        score = det.score(logits_batch)         # shape (N,)
        mask_ood = det.predict(logits_batch)    # True = OOD
    """

    def __init__(self, method: str = 'energy', threshold: Optional[float] = None,
                 percentile: float = 95.0):
        """
        Args:
            method: 'energy' 或 'mahal'。
            threshold: 若给定则直接使用；否则 fit() 时按 percentile 自动估计。
            percentile: fit() 时用训练集分数的多少分位数作为阈值（默认 95）。
        """
        self.method = method
        self.threshold = threshold
        self.percentile = percentile
        # Mahalanobis 专用
        self._class_means: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None

    # ---- 分数计算 ----
    @staticmethod
    def energy_score(logits: np.ndarray) -> np.ndarray:
        """Energy score: -log Σ exp(logits)。越高 = 越 in-distribution。

        原论文 (Liu et al. 2020) 用负值作为 OOD 分数，这里取负号使"越高=越OOD"。
        """
        return -np.log(np.exp(logits).sum(axis=1) + 1e-9)

    def mahal_score(self, feats: np.ndarray) -> np.ndarray:
        """Mahalanobis 距离：到最近类中心的距离（越大 = 越 OOD）。"""
        if self._class_means is None:
            raise RuntimeError("请先调用 fit() 估计类别统计量。")
        n = feats.shape[0]
        scores = np.full(n, np.inf, dtype=np.float32)
        for mu in self._class_means:
            diff = feats - mu  # (N, D)
            m = np.einsum('nd,dd,nd->n', diff, self._inv_cov, diff)
            scores = np.minimum(scores, m)
        return scores.astype(np.float32)

    def score(self, logits_or_feats: np.ndarray) -> np.ndarray:
        if self.method == 'energy':
            return self.energy_score(logits_or_feats)
        return self.mahal_score(logits_or_feats)

    def predict(self, logits_or_feats: np.ndarray) -> np.ndarray:
        """返回布尔数组，True 表示 OOD（超出分布）。"""
        if self.threshold is None:
            return np.zeros(logits_or_feats.shape[0], dtype=bool)
        return self.score(logits_or_feats) > self.threshold

    # ---- 校准 ----
    @torch.no_grad()
    def fit(self, loader, model: nn.Module, device: str = 'cpu',
            extract_feat: bool = False):
        """用训练集激活估计 OOD 阈值（及 Mahalanobis 协方差）。

        extract_feat=True 时尝试提取 penultimate 特征（用于 mahal），
        否则使用 logits（用于 energy）。
        """
        model.eval()
        all_acts, all_labels = [], []
        for x, y in loader:
            x = x.to(device)
            if extract_feat and hasattr(model, 'model'):
                # timm 模型：去掉分类头获取特征
                inner = model.model
                feat = inner.forward_features(x)
                if feat.ndim > 2:
                    feat = feat.mean(dim=list(range(1, feat.ndim - 1)))
                act = feat.cpu().numpy()
            else:
                act = model(x).cpu().numpy()
            all_acts.append(act)
            all_labels.append(y.numpy())

        all_acts = np.concatenate(all_acts, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.method == 'mahal':
            classes = sorted(set(all_labels.tolist()))
            means = []
            pooled_cov = np.zeros((all_acts.shape[1], all_acts.shape[1]))
            for c in classes:
                idx = all_labels == c
                mu = all_acts[idx].mean(axis=0)
                means.append(mu)
                diff = all_acts[idx] - mu
                pooled_cov += diff.T @ diff
            pooled_cov /= max(len(all_acts) - len(classes), 1)
            reg = 1e-5 * np.eye(pooled_cov.shape[0])
            self._class_means = np.stack(means, axis=0)
            self._inv_cov = np.linalg.inv(pooled_cov + reg).astype(np.float32)
            self._class_means = self._class_means.astype(np.float32)
            scores = self.mahal_score(all_acts.astype(np.float32))
        else:
            scores = self.energy_score(all_acts)

        if self.threshold is None:
            self.threshold = float(np.percentile(scores, self.percentile))


# ---------------------------------------------------------------------------
# 3. MC-Dropout 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_dropout_predict(model: nn.Module, image_tensor: 'torch.Tensor',
                       n_passes: int = 10, device: str = 'cpu') -> np.ndarray:
    """MC-Dropout：训练模式下多次前向，返回 (1, T, C) softmax 概率。

    需要模型中存在 Dropout 层；timm 骨干大多有 drop_rate 可配置。
    """
    model.train()  # 启用 dropout
    probs = []
    image_tensor = image_tensor.to(device)
    for _ in range(n_passes):
        out = model(image_tensor)
        probs.append(torch.softmax(out, dim=-1).cpu().numpy())
    model.eval()
    return np.stack(probs, axis=1)  # (N, T, C)


# ---------------------------------------------------------------------------
# 4. 批量路由（替换 ActiveLearningGUI.batch_predict 中的分支逻辑）
# ---------------------------------------------------------------------------

class SelfLearner:
    """自学习路由器：封装采样策略 + OOD 检测 + 伪/确认分层写入。

    可直接替换 modler_v2 的 batch_predict 内核，也可独立调用。
    """

    STRATEGIES = ['max_softmax', 'entropy', 'margin', 'least_confidence', 'coreset']

    def __init__(self,
                 strategy: str = 'entropy',
                 auto_threshold: float = 0.90,
                 review_threshold: float = 0.50,
                 ood_detector: Optional[OODDetector] = None,
                 use_bald: bool = False,
                 bald_passes: int = 10):
        """
        Args:
            strategy: 采样策略，影响"送审核"的样本选择顺序。
            auto_threshold: 置信度高于此值且非 OOD → 伪标注自动归档。
            review_threshold: 低于 auto_threshold 但高于此值 → 送人工审核。
            ood_detector: 若提供，疑似 OOD 样本强制送审核（不自动归档）。
            use_bald: 是否启用 MC-Dropout BALD（需模型有 Dropout）。
            bald_passes: MC-Dropout 采样次数。
        """
        self.strategy = strategy
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold
        self.ood_detector = ood_detector
        self.use_bald = use_bald
        self.bald_passes = bald_passes

    def route_batch(self,
                    image_paths: list[str],
                    probs_batch: np.ndarray,
                    logits_batch: Optional[np.ndarray],
                    class_names: list[str],
                    data_manager,
                    progress_cb: Optional[Callable] = None,
                    log_cb: Optional[Callable] = None) -> dict:
        """对一批图像的预测结果执行路由。

        Args:
            image_paths: 图像路径列表 (N,)。
            probs_batch: softmax 概率数组 (N, C)。
            logits_batch: 原始 logits (N, C)（OOD energy score 用），可为 None。
            class_names: 类别名称列表。
            data_manager: DataManager 实例（提供 add_to_training/add_to_review_queue）。
            progress_cb: 进度回调 (done, total)。
            log_cb: 日志回调 (msg, level)。

        Returns:
            {'auto': int, 'review': int, 'ood': int, 'discard': int,
             'records': list[dict]}
        """
        def log(msg, level="INFO"):
            if log_cb:
                log_cb(msg, level)

        N = len(image_paths)
        max_conf = probs_batch.max(axis=1)
        pred_idx = probs_batch.argmax(axis=1)

        # 不确定性分数（用于审核队列优先级排序，不影响路由阈值判断）
        if self.strategy == 'entropy':
            unc = entropy_score(probs_batch)
        elif self.strategy == 'margin':
            unc = margin_score(probs_batch)
        elif self.strategy == 'least_confidence':
            unc = least_confidence_score(probs_batch)
        else:  # max_softmax / coreset（coreset 需嵌入，此处退化为 1-conf）
            unc = 1.0 - max_conf

        # OOD 检测
        ood_mask = np.zeros(N, dtype=bool)
        if self.ood_detector is not None and logits_batch is not None:
            try:
                ood_mask = self.ood_detector.predict(logits_batch)
                n_ood = int(ood_mask.sum())
                if n_ood:
                    log(f"OOD 检测: {n_ood} 张疑似分布外样本→强制审核", "WARNING")
            except Exception as e:
                log(f"OOD 检测失败（跳过）: {e}", "WARNING")

        stats = {'auto': 0, 'review': 0, 'ood': 0, 'discard': 0, 'records': []}

        for i, path in enumerate(image_paths):
            conf = float(max_conf[i])
            pred_cls = class_names[int(pred_idx[i])]
            is_ood = bool(ood_mask[i])

            if conf >= self.auto_threshold and not is_ood:
                # 高置信 + 非 OOD → 伪标注
                try:
                    data_manager.add_to_training(path, pred_cls, source='pseudo')
                    handle = '伪标注归档'
                    stats['auto'] += 1
                except Exception as e:
                    handle = f'归档失败:{e}'
            elif conf >= self.review_threshold or is_ood:
                # 中置信 / OOD → 人工审核
                try:
                    data_manager.add_to_review_queue(
                        path, pred_cls, conf, probs_batch[i])
                    handle = 'OOD→审核' if is_ood else '待人工审核'
                    if is_ood:
                        stats['ood'] += 1
                    else:
                        stats['review'] += 1
                except Exception as e:
                    handle = f'审核队列写入失败:{e}'
            else:
                handle = '低置信度丢弃'
                stats['discard'] += 1

            stats['records'].append({
                '文件名': Path(path).name,
                '文件路径': path,
                '预测类别': pred_cls,
                '置信度': round(conf, 4),
                '不确定度': round(float(unc[i]), 4),
                '是否OOD': is_ood,
                '处理方式': handle,
                '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '_path_key': path,
            })

            if progress_cb and (i + 1) % 10 == 0:
                progress_cb(i + 1, N)

        if progress_cb:
            progress_cb(N, N)
        return stats


# ---------------------------------------------------------------------------
# 5. 伪/确认分层 DataManager 扩展（Mixin）
# ---------------------------------------------------------------------------

class StratifiedDataMixin:
    """为 DataManager 添加 confirmed / pseudo 两类样本分层管理。

    继承方式：
        class DataManager(StratifiedDataMixin, _OriginalDataManager): ...

    或直接调用 augment_data_manager(dm) 动态混入已有实例。
    """

    def _init_stratified(self):
        """在 DataManager.__init__ 末尾调用，初始化分层计数。"""
        if 'label_source' not in self.metadata.get('statistics', {}):
            self.metadata.setdefault('statistics', {})
            self.metadata['statistics'].setdefault('confirmed', 0)
            self.metadata['statistics'].setdefault('pseudo', 0)

    def add_to_training(self, image_path, class_name, source='manual'):
        """重写：区分 confirmed（manual）/ pseudo（auto/pseudo）路径。"""
        if source in ('manual', 'confirmed'):
            target_dir = Path(self.archive_dir) / class_name
            stat_key = 'confirmed'
        else:  # 'auto' / 'pseudo' → 伪标签子目录，便于后续区分权重
            target_dir = Path(self.archive_dir) / '_pseudo' / class_name
            stat_key = 'pseudo'

        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dst = target_dir / f"{ts}_{Path(image_path).name}"
        shutil.copy2(image_path, dst)

        self.metadata['statistics']['total_samples'] = \
            self.metadata['statistics'].get('total_samples', 0) + 1
        self.metadata['statistics'][stat_key] = \
            self.metadata['statistics'].get(stat_key, 0) + 1
        if class_name not in self.metadata.get('classes', []):
            self.metadata.setdefault('classes', []).append(class_name)
            self.metadata['classes'] = sorted(self.metadata['classes'])
        self._save_metadata()
        return dst

    def get_confirmed_dir(self) -> Path:
        return Path(self.archive_dir)

    def get_pseudo_dir(self) -> Path:
        return Path(self.archive_dir) / '_pseudo'

    def pseudo_count(self) -> int:
        return self.metadata.get('statistics', {}).get('pseudo', 0)

    def confirmed_count(self) -> int:
        return self.metadata.get('statistics', {}).get('confirmed', 0)


def augment_data_manager(dm):
    """动态将 StratifiedDataMixin 混入已有 DataManager 实例（不改类继承）。"""
    import types
    for name in ('_init_stratified', 'add_to_training',
                 'get_confirmed_dir', 'get_pseudo_dir',
                 'pseudo_count', 'confirmed_count'):
        method = getattr(StratifiedDataMixin, name)
        setattr(dm, name, types.MethodType(method, dm))
    dm._init_stratified()
    return dm
