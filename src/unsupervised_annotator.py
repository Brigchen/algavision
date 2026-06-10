# -*- coding: utf-8 -*-
"""
unsupervised_annotator.py — 无监督/弱监督标注核心

实现"冷启动"无监督标注流水线（无需种子模型、无需预定义类别）：
    扫描无标签图像 -> 自监督骨干(DINOv2)提取嵌入 -> 降维 -> HDBSCAN 聚类
    -> 每簇代表图（供人工一键命名）-> 导出为 类别/ 子目录结构

聚类使用 sklearn 内置 HDBSCAN（噪声点 label=-1，正好对应显微碎屑/杂质）。
UMAP 可选（安装则用，否则退化为 PCA）。

@author: brigchen@xmu.edu.cn
"""
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models_registry import build_embedding_extractor

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# DINOv2 / ImageNet 归一化
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def list_images(folder, recursive=True):
    """收集文件夹内所有图像路径。"""
    folder = Path(folder)
    if recursive:
        files = [p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])


@torch.no_grad()
def extract_embeddings(image_paths, model, transform, device='cpu',
                       batch_size=32, progress_cb=None):
    """批量提取嵌入，返回 (embeddings[N,D], valid_paths)。

    无法读取的图像会被跳过；progress_cb(done, total) 用于 GUI 进度。
    """
    embeddings = []
    valid_paths = []
    total = len(image_paths)
    batch_imgs, batch_paths = [], []

    def _flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        feats = model(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = feats.detach().cpu().numpy()
        embeddings.append(feats)
        valid_paths.extend(batch_paths)
        batch_imgs.clear()
        batch_paths.clear()

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            batch_imgs.append(transform(img))
            batch_paths.append(str(path))
        except Exception:
            pass  # 跳过损坏图像
        if len(batch_imgs) >= batch_size:
            _flush()
        if progress_cb and (i + 1) % 10 == 0:
            progress_cb(i + 1, total)
    _flush()
    if progress_cb:
        progress_cb(total, total)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []
    return np.concatenate(embeddings, axis=0).astype(np.float32), valid_paths


def l2_normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def reduce_dims(emb, n_components=50, method='pca'):
    """降维。method='umap' 时若未安装则自动回退 PCA。"""
    if emb.shape[0] <= 2 or emb.shape[1] <= n_components:
        return emb
    if method == 'umap':
        try:
            import umap  # noqa
            reducer = umap.UMAP(n_components=min(n_components, 50),
                                metric='cosine', random_state=42)
            return reducer.fit_transform(emb).astype(np.float32)
        except Exception:
            method = 'pca'  # 回退
    from sklearn.decomposition import PCA
    n = min(n_components, emb.shape[0] - 1, emb.shape[1])
    return PCA(n_components=n, random_state=42).fit_transform(emb).astype(np.float32)


def cluster_embeddings(emb, method='hdbscan', min_cluster_size=10,
                       n_clusters=None):
    """对嵌入聚类，返回整型标签数组（HDBSCAN 中 -1 表示噪声/杂质）。"""
    if emb.shape[0] == 0:
        return np.array([], dtype=int)
    if emb.shape[0] < max(2, min_cluster_size):
        # 样本太少，全部归为一个簇
        return np.zeros(emb.shape[0], dtype=int)

    if method == 'kmeans':
        from sklearn.cluster import KMeans
        k = n_clusters or max(2, int(np.sqrt(emb.shape[0] / 2)))
        k = min(k, emb.shape[0])
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)

    # 默认 HDBSCAN（sklearn>=1.3 内置）
    from sklearn.cluster import HDBSCAN
    min_samples = max(1, min_cluster_size // 2)
    labels = HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
        metric='euclidean',
    ).fit_predict(emb)
    return labels


def representatives_by_cluster(emb, labels, top_k=25):
    """每个簇按到簇中心的距离升序返回代表样本下标。

    返回 dict: {cluster_id: [idx, ...]}（含噪声簇 -1）。
    """
    reps = {}
    for cid in sorted(set(labels.tolist())):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        center = emb[idxs].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(emb[idxs] - center, axis=1)
        order = idxs[np.argsort(dists)]
        reps[cid] = order[:top_k].tolist()
    return reps


class UnsupervisedAnnotator:
    """无监督标注编排器：从无标签文件夹到聚类结果。"""

    def __init__(self, model_key='DINOv2-S', device=None):
        self.model_key = model_key
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.cfg = None
        self.transform = None

        # 运行结果
        self.image_paths = []
        self.embeddings = None
        self.reduced = None
        self.labels = None
        self.reps = {}

    def _ensure_model(self):
        if self.model is None:
            self.model, self.cfg = build_embedding_extractor(
                self.model_key, pretrained=True, device=self.device
            )
            self.transform = build_transform(self.cfg['default_size'])

    def run(self, folder, recursive=True, batch_size=32,
            reduce_method='pca', n_components=50,
            cluster_method='hdbscan', min_cluster_size=10, n_clusters=None,
            top_k=25, progress_cb=None, log_cb=None):
        """执行完整流水线，返回 (labels, reps, valid_paths)。"""
        def log(msg):
            if log_cb:
                log_cb(msg)

        self._ensure_model()
        log(f"使用嵌入骨干: {self.model_key} ({self.cfg['model_name']}), 设备: {self.device}")

        all_paths = list_images(folder, recursive=recursive)
        log(f"发现 {len(all_paths)} 张图像")
        if not all_paths:
            self.image_paths, self.labels, self.reps = [], np.array([], dtype=int), {}
            return self.labels, self.reps, self.image_paths

        log("提取嵌入中...")
        emb, valid_paths = extract_embeddings(
            all_paths, self.model, self.transform, self.device,
            batch_size=batch_size, progress_cb=progress_cb
        )
        self.image_paths = valid_paths
        self.embeddings = l2_normalize(emb)
        log(f"嵌入完成: {self.embeddings.shape}")

        log(f"降维({reduce_method}) -> 聚类({cluster_method})...")
        self.reduced = reduce_dims(self.embeddings, n_components=n_components,
                                   method=reduce_method)
        self.labels = cluster_embeddings(
            self.reduced, method=cluster_method,
            min_cluster_size=min_cluster_size, n_clusters=n_clusters
        )
        self.reps = representatives_by_cluster(self.reduced, self.labels, top_k=top_k)

        n_clusters_found = len([c for c in set(self.labels.tolist()) if c != -1])
        n_noise = int(np.sum(self.labels == -1))
        log(f"聚类完成: {n_clusters_found} 个簇, {n_noise} 个噪声/杂质样本")
        return self.labels, self.reps, self.image_paths

    def export(self, output_dir, cluster_names, copy=True, include_noise=False):
        """按 {cluster_id: name} 导出到 output_dir/<name>/ 结构。

        cluster_names 中未命名（或映射为空）的簇会被跳过；
        噪声簇(-1)默认跳过，include_noise=True 时导出到 'unclustered'。
        返回导出统计 dict。
        """
        output_dir = Path(output_dir)
        stats = {}
        for idx, path in enumerate(self.image_paths):
            cid = int(self.labels[idx])
            if cid == -1:
                if not include_noise:
                    continue
                name = 'unclustered'
            else:
                name = cluster_names.get(cid) or cluster_names.get(str(cid))
                if not name:
                    continue
            dst_dir = output_dir / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / Path(path).name
            if dst.exists():
                dst = dst_dir / f"{Path(path).stem}_{idx}{Path(path).suffix}"
            if copy:
                shutil.copy2(path, dst)
            else:
                shutil.move(path, dst)
            stats[name] = stats.get(name, 0) + 1
        return stats


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='无监督聚类标注（命令行）')
    ap.add_argument('folder', help='无标签图像文件夹')
    ap.add_argument('--model', default='DINOv2-S')
    ap.add_argument('--min-cluster-size', type=int, default=10)
    ap.add_argument('--method', default='hdbscan', choices=['hdbscan', 'kmeans'])
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    ann = UnsupervisedAnnotator(model_key=args.model)
    labels, reps, paths = ann.run(
        args.folder, batch_size=args.batch_size,
        cluster_method=args.method, min_cluster_size=args.min_cluster_size,
        log_cb=print,
    )
    for cid, idxs in reps.items():
        tag = '噪声/杂质' if cid == -1 else f'簇 {cid}'
        print(f"{tag}: {len(np.where(labels == cid)[0])} 张，代表图示例: "
              f"{[Path(paths[i]).name for i in idxs[:3]]}")
