# -*- coding: utf-8 -*-
"""
version 2.3.11 
Created on Tue Dec 23 15:05:44 2025
update on Thu Jun 20 11:00:00 2024 : correct bugs on train dataset dir 
update on Wed Jun 20 17:00:00 2024 : (1)treading update for GUI responsiveness
                                    (2)function of best hyperparameters search
update on Fri Jun 21 14:00:00 2024 : correct bugs on no update of prediction table for review
update on Tue Jan 27 16:00:00 2026 : fix review table not updating by saving full predictions
update on Tue Mar 10 19:00:00 2026 : (1) fix copy errors for existing images
                                    (2) add threshold of review
update on Wed Mar 11 9:00:00 2026 : (1) excel export identification results
                                    
@author: brigchen@xmu.edu.cn
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import traceback
import threading

from models_registry import (
    MODEL_REGISTRY,
    build_model,
    freeze_backbone as registry_freeze_backbone,
    unfreeze_backbone as registry_unfreeze_backbone,
)
from self_learning import SelfLearner, OODDetector, augment_data_manager

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_optimal_batch_size(img_size=224):
    """根据GPU显存自动计算合适的批次大小"""
    if not torch.cuda.is_available():
        return 16  # CPU默认使用较小批次
    
    try:
        # 获取GPU显存信息（单位：GB）
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # 根据图像大小和显存估算批次大小
        if gpu_memory >= 24:  # 24GB+
            if img_size <= 224:
                return 128
            elif img_size <= 384:
                return 64
            else:
                return 32
        elif gpu_memory >= 16:  # 16GB
            if img_size <= 224:
                return 64
            elif img_size <= 384:
                return 32
            else:
                return 16
        elif gpu_memory >= 12:  # 12GB
            if img_size <= 224:
                return 48
            elif img_size <= 384:
                return 24
            else:
                return 12
        elif gpu_memory >= 8:  # 8GB
            if img_size <= 224:
                return 32
            elif img_size <= 384:
                return 16
            else:
                return 8
        elif gpu_memory >= 6:  # 6GB
            if img_size <= 224:
                return 24
            elif img_size <= 384:
                return 12
            else:
                return 8
        else:  # 4GB及以下
            if img_size <= 224:
                return 16
            elif img_size <= 384:
                return 8
            else:
                return 4
    except Exception as e:
        print(f"自动计算批次大小失败: {e}")
        return 32  # 失败时使用默认值

class PhytoplanktonDataset(Dataset):
    """浮游植物数据集类"""
    
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 获取所有类别（子目录）
        if class_to_idx is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = [cls_name for cls_name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"数据集加载完成: {len(self.samples)} 张图像，{len(self.classes)} 个类别")
        print(f'数据集目录: {self.root_dir}')
        print("类别列表:", self.classes)
        print("类别到索引映射:", self.class_to_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        return self.classes[idx]


class _TransformSubset(Dataset):
    """为 random_split 得到的子集单独应用 transform。

    修复原实现的缺陷：原代码对三个子集共享的底层 dataset 反复赋值
    transform，最后一次赋值（val_transform）会覆盖全部子集，导致训练集
    实际未使用数据增强。此包装类使每个子集拥有独立的 transform。
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  # 底层 dataset.transform=None，返回 PIL 图像
        if self.transform:
            image = self.transform(image)
        return image, label


class EfficientNetV2Classifier(nn.Module):
    """通用分类骨干封装（经由 models_registry 构建，支持任意 SOTA 架构）。

    类名保持不变以兼容现有 checkpoint 与调用方；内部已模型无关。
    """

    def __init__(self, num_classes, model_name='tf_efficientnetv2_b0',
                 pretrained=True, img_size=None, dynamic_img_size=False):
        super(EfficientNetV2Classifier, self).__init__()

        self.model = build_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            img_size=img_size,
            dynamic_img_size=dynamic_img_size,
        )

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """冻结骨干网络 - 仅保留分类头可训练（通用，适配任意 timm 架构）"""
        registry_freeze_backbone(self.model)

    def unfreeze_backbone(self):
        """解冻骨干网络 - 所有参数可训练"""
        registry_unfreeze_backbone(self.model)


class PhytoplanktonClassifier:
    """分类器训练和预测管理类"""
    
    # 模型配置映射（来自统一注册表，支持 EfficientNetV2/ConvNeXtV2/ViT/Swin/MaxViT/DINOv2）
    MODEL_CONFIGS = MODEL_REGISTRY
    
    def __init__(self, num_classes, model_type='B0', img_size=None,
                 learning_rate=1e-3, optimizer_type='Adam', weight_decay=1e-4,
                 scheduler_type='ReduceLROnPlateau', augmentation_level='中等',
                 style_folder: str = None):
        self.num_classes = num_classes
        self.style_folder = style_folder  
        # 兼容旧版本保存的带后缀键名（如 'B0 (快速)'）及未知键
        if model_type not in self.MODEL_CONFIGS:
            normalized = model_type.split('(')[0].split(' ')[0].strip()
            model_type = normalized if normalized in self.MODEL_CONFIGS else 'B2'

        self.model_type = model_type  
        self.learning_rate = learning_rate  
        self.optimizer_type = optimizer_type  
        self.weight_decay = weight_decay  
        self.scheduler_type = scheduler_type  
        self.augmentation_level = augmentation_level  
        
        # 获取模型配置
        config = self.MODEL_CONFIGS[model_type]
        model_name = config['model_name']
        self.img_size = img_size or config['default_size']
        
        print(f"\n初始化模型: {model_type}")
        print(f"模型名称: {model_name}")
        print(f"输入尺寸: {self.img_size}×{self.img_size}")
        print(f"参数量: {config['params']}")
        
        # 构建模型（动态分辨率骨干如 DINOv2 需要在构建时传入 img_size）
        dynamic_img_size = config.get('dynamic_img_size', False)
        self.model = EfficientNetV2Classifier(
            num_classes, model_name=model_name,
            img_size=self.img_size, dynamic_img_size=dynamic_img_size
        ).to(device)

        # 混合精度训练（AMP）：仅在 CUDA 下启用，CPU 自动退化为普通训练
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 根据优化器类型创建优化器  
        if optimizer_type == 'Adam':  
            self.optimizer = optim.Adam(  
                self.model.parameters(),   
                lr=learning_rate,   
                weight_decay=weight_decay  
            )  
        elif optimizer_type == 'AdamW':  
            self.optimizer = optim.AdamW(  
                self.model.parameters(),   
                lr=learning_rate,   
                weight_decay=weight_decay  
            )  
        elif optimizer_type == 'SGD':  
            self.optimizer = optim.SGD(  
                self.model.parameters(),   
                lr=learning_rate,   
                momentum=0.9,   
                weight_decay=weight_decay  
            )  
        elif optimizer_type == 'RMSprop':  
            self.optimizer = optim.RMSprop(  
                self.model.parameters(),   
                lr=learning_rate,   
                weight_decay=weight_decay  
            )  
        
        # 根据调度器类型创建调度器  
        if scheduler_type == 'ReduceLROnPlateau':  
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(  
                self.optimizer, mode='min', factor=0.5, patience=2  
            )  
        elif scheduler_type == 'CosineAnnealing':  
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(  
                self.optimizer, T_max=10, eta_min=1e-6  
            )  
        elif scheduler_type == 'StepLR':  
            self.scheduler = optim.lr_scheduler.StepLR(  
                self.optimizer, step_size=5, gamma=0.5  
            )  
        else:  
            self.scheduler = None  
        
        # 根据增强级别设置数据变换
        self.train_transform = self._create_transform(augmentation_level)
        self.val_transform = self._create_transform('无')

        
    def _create_transform(self, augmentation_level):
        """根据增强级别创建数据变换。

        新增两级：
          '显微镜增强' — 全角度旋转 + CLAHE + 模糊 + 噪声，不含风格迁移
          '风格迁移'   — 上述显微镜增强 + FDA/直方图/AdaIN 跨域风格化
                         （需在初始化时传入 style_folder）
        """
        _NORM_MEAN = [0.485, 0.456, 0.406]
        _NORM_STD  = [0.229, 0.224, 0.225]

        # 专用显微成像增强级别走独立流水线
        if augmentation_level in ('显微镜增强', '风格迁移'):
            from style_aug import build_microscopy_transform
            use_style = (augmentation_level == '风格迁移')
            return build_microscopy_transform(
                img_size=self.img_size,
                style_folder=self.style_folder if use_style else None,
                style_method='fda',
                style_p=0.7,
                mode='train',
            )

        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
        ]

        if augmentation_level == '轻度':
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        elif augmentation_level == '中等':
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        elif augmentation_level == '强烈':
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]
        else:  # 无
            augmentations = []

        return transforms.Compose(
            base_transforms + augmentations + [
                transforms.ToTensor(),
                transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
            ]
        )
    
    def prepare_dataloaders(self, data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15):
        """准备数据加载器"""
        full_dataset = PhytoplanktonDataset(data_dir, transform=None)
        
        self.class_names = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx
        
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        train_subset, val_subset, test_subset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 底层 dataset.transform 保持为 None，由各子集独立包装 transform，
        # 避免共享底层 dataset 时训练增强被验证 transform 覆盖。
        train_dataset = _TransformSubset(train_subset, self.train_transform)
        val_dataset = _TransformSubset(val_subset, self.val_transform)
        test_dataset = _TransformSubset(test_subset, self.val_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        print(f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            # 梯度裁剪前需先 unscale，避免在缩放后的梯度上裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=10, early_stop_patience=3, callback=None, run_manager=None):
        """完整训练流程"""
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n开始训练，共 {epochs} 个epochs")
        print(f"设备: {device}")
        print(f"模型: {self.model_type}")
        
        if run_manager:
            run_manager.log(f"开始训练 - 模型: {self.model_type}, Epochs: {epochs}")
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                log_msg = f"学习率已调整: {old_lr:.2e} -> {new_lr:.2e}"
                print(log_msg)
                if run_manager:
                    run_manager.log(log_msg)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(new_lr)
            
            epoch_msg = (f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                        f"LR: {new_lr:.2e}")
            print(epoch_msg)
            
            if run_manager:
                run_manager.log(epoch_msg)
            
            # 回调函数（用于更新GUI）
            if callback:
                callback(epoch, epochs, train_loss, train_acc, val_loss, val_acc)
            
            # 保存检查点
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"✓ 新的最佳验证准确率: {best_val_acc:.2f}%")
                if run_manager:
                    run_manager.log(f"新的最佳验证准确率: {best_val_acc:.2f}%", "SUCCESS")
            else:
                patience_counter += 1
                print(f"验证准确率未提升 ({patience_counter}/{early_stop_patience})")
            
            # # 自动保存检查点（如果提供了run_manager）
            if run_manager:
                try:
                    checkpoint_path = run_manager.save_checkpoint(
                        self.model, 
                        self.optimizer, 
                        epoch, 
                        val_acc,
                        self.history, 
                        self.class_names, 
                        self.class_to_idx,
                        self.model_type, 
                        self.img_size, 
                        is_best=is_best
                    )
                except Exception as e:
                    error_msg = f"保存检查点失败: {str(e)}"
                    print(error_msg)
                    if run_manager:
                        run_manager.log(error_msg, "ERROR")
            
            # 早停
            if patience_counter >= early_stop_patience:
                stop_msg = "早停触发，停止训练"
                print(f"\n{stop_msg}")
                if run_manager:
                    run_manager.log(stop_msg, "WARNING")
                break
        
        # 训练结束，保存最终统计
        if run_manager:
            run_manager.log(f"训练完成 - 最佳验证准确率: {best_val_acc:.2f}%", "SUCCESS")
            try:
                run_manager.save_statistics({
                    'epochs_trained': len(self.history['train_loss']),
                    'best_val_acc': best_val_acc,
                    'final_train_loss': self.history['train_loss'][-1],
                    'final_val_loss': self.history['val_loss'][-1],
                    'model_type': self.model_type
                })
                # 只在训练结束后保存一次最终训练曲线
                run_manager.save_training_curves(self.history, "training_curves.png")
            except Exception as e:
                run_manager.log(f"保存统计信息失败: {str(e)}", "ERROR")
        
        return self.history
   
    def fine_tune(self, train_loader, val_loader, epochs=5, callback=None):
        """微调模型"""
        print("\n开始微调模型...")
        self.model.unfreeze_backbone()
        return self.train(train_loader, val_loader, epochs=epochs, 
                         early_stop_patience=2, callback=callback)
    
    def predict(self, image_path):
        """预测单张图像"""
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()[0]
    
    def save_model(self, path):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'history': self.history,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'img_size': self.img_size
        }
        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint['class_to_idx']
        self.history = checkpoint['history']
        self.num_classes = checkpoint['num_classes']
        loaded_type = checkpoint.get('model_type', 'B2')
        if loaded_type not in self.MODEL_CONFIGS:
            normalized = loaded_type.split('(')[0].split(' ')[0].strip()
            loaded_type = normalized if normalized in self.MODEL_CONFIGS else self.model_type
        self.model_type = loaded_type
        self.img_size = checkpoint.get('img_size', 224)
        print(f"模型已加载: {path}")


class DataManager:
    """数据管理器"""
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.train_dir = None
        self.archive_dir = self.base_dir / "archived_data"
        self.review_dir = None
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def set_train_directory(self, train_dir):
        """设置训练数据目录"""
        self.train_dir = Path(train_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        if "directories" not in self.metadata:
            self.metadata["directories"] = {}
        self.metadata["directories"]["train_dir"] = str(self.train_dir)
        self._save_metadata()
        return True

    def set_review_directory(self, review_dir):
        """设置待审核目录"""
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        if "directories" not in self.metadata:
            self.metadata["directories"] = {}
        self.metadata["directories"]["review_dir"] = str(self.review_dir)
        self._save_metadata()
        return True

    def set_archive_directory(self, archive_dir=None):
        """设置归档目录"""
        if archive_dir:
            self.archive_dir = Path(archive_dir)
        else:
            self.archive_dir = self.base_dir / "archived_data"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if "directories" not in self.metadata:
            self.metadata["directories"] = {}
        self.metadata["directories"]["archive_dir"] = str(self.archive_dir)
        self._save_metadata()
        return True

    def set_directories(self, train_dir, review_dir, archive_dir=None):  
        self.set_train_directory(train_dir)  
        self.set_review_directory(review_dir)  
        self.set_archive_directory(archive_dir)  
        return True
    
    def _load_metadata(self):
        return {
            "classes": [],
            "model_versions": [],
            "directories": {},
            "statistics": {
                "total_samples": 0,
                "auto_archived": 0,
                "human_reviewed": 0
            }
        }
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def get_class_names(self):
        """获取类别名称"""
        if not self.train_dir or not self.train_dir.exists():
            return []
        classes = [d.name for d in self.train_dir.iterdir() if d.is_dir()]
        self.metadata["classes"] = sorted(classes)
        self._save_metadata()
        return self.metadata["classes"]
    
    def clear_review_queue(self):
        """清除所有待审核图像"""
        if not self.review_dir or not self.review_dir.exists():
            return 0
        deleted_count = 0
        try:
            for file_path in self.review_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            return deleted_count
        except Exception as e:
            raise Exception(f"清除待审核队列失败: {str(e)}")
  
    def add_to_training(self, image_path, class_name, source="manual"):
        """将图像归档到对应目录"""
        if source == "auto":
            target_dir = self.archive_dir / class_name
        else:
            target_dir = self.archive_dir / class_name
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dest_path = target_dir / f"{timestamp}_{Path(image_path).name}"
        shutil.copy2(image_path, dest_path)
        
        # 更新统计
        self.metadata["statistics"]["total_samples"] += 1
        if source == "auto":
            self.metadata["statistics"]["auto_archived"] += 1
        else:
            self.metadata["statistics"]["human_reviewed"] += 1
        
        if class_name not in self.metadata["classes"]:
            self.metadata["classes"].append(class_name)
            self.metadata["classes"] = sorted(self.metadata["classes"])
        
        self._save_metadata()
        return dest_path

    def create_new_class(self, class_name):
        """创建新类别目录"""
        archive_class_dir = self.archive_dir / class_name
        archive_class_dir.mkdir(parents=True, exist_ok=True)
        
        if self.train_dir and self.train_dir.exists():
            train_class_dir = self.train_dir / class_name
            train_class_dir.mkdir(parents=True, exist_ok=True)
        
        if class_name not in self.metadata["classes"]:
            self.metadata["classes"].append(class_name)
            self.metadata["classes"] = sorted(self.metadata["classes"])
            self._save_metadata()
        
        return archive_class_dir


    def add_to_review_queue(self, image_path, predicted_class, confidence, predictions=None):
        """添加到待审核队列"""
        if self.review_dir is None:
            raise ValueError("请先设置待审核目录！")
        src_path = Path(image_path).resolve() 
        review_file = self.review_dir / Path(image_path).name
        dst_path = review_file.resolve() 

        # 源文件与目标文件相同时跳过复制
        if src_path != dst_path:  
            # 目标文件名冲突时追加时间戳，避免覆盖其他文件  
            if review_file.exists():  
                stem = review_file.stem  
                suffix = review_file.suffix  
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
                review_file = self.review_dir / f"{stem}_{timestamp}{suffix}"  
            shutil.copy2(image_path, review_file)  
        # else: 文件已在待审核目录，无需复制，直接写JSON元数据 

        # shutil.copy2(image_path, review_file)
        
        # 准备元数据
        data = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "original_path": str(image_path)
        }
        
        # 保存完整预测概率以便在GUI中显示
        if predictions is not None:
            if hasattr(predictions, 'tolist'):
                data["predictions"] = predictions.tolist()
            else:
                data["predictions"] = list(predictions)

        info_file = review_file.with_suffix('.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return review_file
    
    def get_review_queue(self):
        if self.review_dir is None:
            return []
        images = list(self.review_dir.glob("*.jpg")) + \
                 list(self.review_dir.glob("*.png")) + \
                 list(self.review_dir.glob("*.jpeg"))
        return sorted(images)
    
    def log_model_version(self, version_info):
        self.metadata["model_versions"].append(version_info)
        self._save_metadata()

class RunManager:
    """运行管理器 - 自动保存和日志管理"""
    
    def __init__(self, base_dir="./runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 创建新的运行目录
        self.run_dir = self._create_run_dir()
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        
        # 创建子目录
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # 日志文件
        self.log_file = self.logs_dir / "training.log"
        self.stats_file = self.logs_dir / "statistics.json"
        self.config_file = self.run_dir / "config.json"
        
        # 初始化日志
        self._init_log()
        
        print(f"运行目录已创建: {self.run_dir}")
    
    def _create_run_dir(self):
        """创建新的运行目录，自动编号"""
        existing_runs = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('run')])
        
        if not existing_runs:
            run_number = 1
        else:
            # 获取最后一个运行的编号
            last_run = existing_runs[-1].name
            try:
                last_number = int(last_run.replace('run', ''))
                run_number = last_number + 1
            except:
                run_number = len(existing_runs) + 1
        
        run_dir = self.base_dir / f"run{run_number:02d}"
        run_dir.mkdir(exist_ok=True)
        
        return run_dir
    
    def _init_log(self):
        """初始化日志文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"{'='*60}\n\n")
    
    def log(self, message, level="INFO"):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 同时打印到控制台
        print(log_entry.strip())
    
    def save_config(self, config_dict):
        """保存训练配置（包括模型设置和超参数设置）"""
        config_dict['timestamp'] = datetime.now().isoformat()
        config_dict['run_dir'] = str(self.run_dir)
        config_dict['device'] = str(device)
        
        # 添加GPU信息
        if torch.cuda.is_available():
            config_dict['gpu_name'] = torch.cuda.get_device_name(0)
            config_dict['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        self.log(f"配置已保存: {self.config_file}")

    def save_statistics_to_file(self, stats_dict):
        """保存数据统计到单独的文件"""
        stats_file = self.logs_dir / "data_statistics.json"
        stats_dict['timestamp'] = datetime.now().isoformat()
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        self.log(f"数据统计已保存: {stats_file}")


    def save_checkpoint(self, model, optimizer, epoch, val_acc, history, class_names, class_to_idx, 
                       model_type, img_size, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': history,
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'num_classes': len(class_names),
            'model_type': model_type,
            'img_size': img_size,
            'timestamp': datetime.now().isoformat()
        }
        
        # # 保存每个epoch的检查点
        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:03d}.pth"
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.log(f"最佳模型已保存: {best_path}, 准确率: {val_acc:.2f}%", "SUCCESS")
        
        # 保存最新模型
        latest_path = self.checkpoints_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def save_statistics(self, stats_dict):
        """保存统计信息"""
        stats_dict['timestamp'] = datetime.now().isoformat()
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    
    def save_training_curves(self, history, save_name="training_curves.png"):
        """保存训练曲线图"""
        try:
            fig = plt.figure(figsize=(15, 10))
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            # 2x2 子图
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            # 1. 损失对比
            ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
            ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('损失曲线对比', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 2. 准确率对比
            ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
            ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('准确率曲线对比', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 3. 训练损失详细
            ax3.plot(epochs, history['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Training Loss', fontsize=12)
            ax3.set_title('训练损失详细', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 最小值标记
            min_idx = np.argmin(history['train_loss'])
            ax3.plot(epochs[min_idx], history['train_loss'][min_idx], 'r*', markersize=15)
            ax3.text(epochs[min_idx], history['train_loss'][min_idx],
                    f'  Min: {history["train_loss"][min_idx]:.4f}',
                    fontsize=10, va='bottom')
            
            # 4. 验证准确率详细
            ax4.plot(epochs, history['val_acc'], 'r-', linewidth=2, marker='s', markersize=4)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
            ax4.set_title('验证准确率详细', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 最大值标记
            max_idx = np.argmax(history['val_acc'])
            ax4.plot(epochs[max_idx], history['val_acc'][max_idx], 'g*', markersize=15)
            ax4.text(epochs[max_idx], history['val_acc'][max_idx],
                    f'  Max: {history["val_acc"][max_idx]:.2f}%',
                    fontsize=10, va='bottom')
            
            fig.tight_layout()
            
            # 保存图表
            save_path = self.logs_dir / save_name
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.log(f"训练曲线已保存: {save_path}")
            
        except Exception as e:
            self.log(f"保存训练曲线失败: {str(e)}", "ERROR")
    
    def get_latest_checkpoint(self):
        """获取最新的检查点路径"""
        latest_path = self.checkpoints_dir / "latest_model.pth"
        if latest_path.exists():
            return latest_path
        return None
    
    def get_best_checkpoint(self):
        """获取最佳模型路径"""
        best_path = self.checkpoints_dir / "best_model.pth"
        if best_path.exists():
            return best_path
        return None


class TrainingProgressWindow:
    """训练进度窗口"""
    
    def __init__(self, parent, total_epochs, logger_callback=None):
        self.window = tk.Toplevel(parent)
        self.window.title("训练进度")
        self.window.geometry("1000x600")
        
        self.total_epochs = total_epochs
        self.logger_callback = logger_callback  # 添加日志回调
        
        # 进度信息
        info_frame = ttk.Frame(self.window, padding="10")
        info_frame.pack(fill=tk.X)
        
        self.epoch_label = ttk.Label(info_frame, text="Epoch: 0/0", font=("Arial", 12))
        self.epoch_label.pack()
        
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            info_frame, 
            variable=self.progress_var,
            maximum=total_epochs,
            length=750
        )
        progress_bar.pack(pady=10)
        
        self.info_text = tk.Text(info_frame, height=4, width=90)
        self.info_text.pack(pady=5)
        
        # 创建训练曲线图
        self.fig = Figure(figsize=(12, 5), dpi=100)  
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)  
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  
        
        # 初始化子图
        self.ax1 = self.fig.add_subplot(1, 2, 1)  # 损失曲线  
        self.ax2 = self.fig.add_subplot(1, 2, 2)  # 准确率曲线  
        
        self.train_losses = []  
        self.val_losses = []  
        self.train_accs = []  
        self.val_accs = []  

        
    def update(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
        """更新进度"""
        self.progress_var.set(epoch)
        self.epoch_label.config(text=f"Epoch: {epoch}/{total_epochs}")
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"训练损失: {train_loss:.4f}\n")
        self.info_text.insert(tk.END, f"训练准确率: {train_acc:.2f}%\n")
        self.info_text.insert(tk.END, f"验证损失: {val_loss:.4f}\n")
        self.info_text.insert(tk.END, f"验证准确率: {val_acc:.2f}%\n")
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        self._update_plots()  
        
        try:  
            self.window.after(100, self.window.update)  
        except:  
            pass  
        
        if self.logger_callback:  
            log_msg = (f"Epoch {epoch}/{total_epochs} - "  
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "  
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")  
            self.logger_callback(log_msg, "INFO") 

    def _update_plots(self):  
        """更新两个子图"""  
        epochs = list(range(1, len(self.train_losses) + 1))  
        
        self.ax1.clear()  
        self.ax2.clear()  
        
        self.ax1.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)  
        self.ax1.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)  
        self.ax1.set_xlabel('Epoch', fontsize=12)  
        self.ax1.set_ylabel('Loss', fontsize=12)  
        self.ax1.set_title('损失曲线', fontsize=14, fontweight='bold')  
        self.ax1.legend(fontsize=11)  
        self.ax1.grid(True, alpha=0.3)  
        self.ax1.set_xlim(1, self.total_epochs)  
        
        self.ax2.plot(epochs, self.train_accs, 'b-', label='训练准确率', linewidth=2)  
        self.ax2.plot(epochs, self.val_accs, 'r-', label='验证准确率', linewidth=2)  
        self.ax2.set_xlabel('Epoch', fontsize=12)  
        self.ax2.set_ylabel('Accuracy (%)', fontsize=12)  
        self.ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')  
        self.ax2.legend(fontsize=11)  
        self.ax2.grid(True, alpha=0.3)  
        self.ax2.set_xlim(1, self.total_epochs)  
        self.ax2.set_ylim(0, 100)  
        
        self.fig.tight_layout()  
        self.canvas.draw()  

class CollapsibleFrame(ttk.Frame):  
    """可折叠的面板"""  
    def __init__(self, parent, text="", **kwargs):  
        super().__init__(parent, **kwargs)  
        
        self.show = tk.BooleanVar(value=True)  
        self.text = text
        
        self.title_frame = ttk.Frame(self, relief=tk.RIDGE, borderwidth=1)  
        self.title_frame.pack(fill=tk.X, pady=(0, 2))  
        
        self.toggle_button = tk.Button(  
            self.title_frame,  
            text=f"▼ {text}",  
            command=self.toggle,
            relief=tk.FLAT,
            anchor=tk.W,
            bg='#E1E1E1',
            activebackground='#D1D1D1',
            font=('Arial', 9, 'bold'),
            cursor='hand2'
        )  
        self.toggle_button.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=2, padx=2)  
        
        self.content_frame = ttk.Frame(self, relief=tk.FLAT, borderwidth=0, padding="5")  
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))  
        
    def toggle(self):  
        """切换折叠/展开状态"""  
        if self.show.get():  
            self.content_frame.pack_forget()  
            self.toggle_button.config(text=f"▶ {self.text}")  
            self.show.set(False)  
        else:  
            self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))  
            self.toggle_button.config(text=f"▼ {self.text}")  
            self.show.set(True)


class ActiveLearningGUI:
    """人机协同主动学习系统GUI"""
    
    def __init__(self, root, data_dir):
        self.root = root
        self.root.title("浮游植物主动学习系统 - Phytoplankton Active Learning System (PyTorch)")
        self.root.geometry("1920x1080")
                        
        # self.train_dir_path = None
        # self.review_dir_path = None
        # self.archive_dir_path = None

        self.train_dir_path   = str(Path("./class/training_data").resolve())  
        self.review_dir_path  = str(Path("./class/review_queue").resolve())  
        self.archive_dir_path = str(Path("./class/archived_data").resolve()) 

        self.data_manager = DataManager(data_dir)
        augment_data_manager(self.data_manager)   # 注入 confirmed/pseudo 分层支持
        self.class_names = self.data_manager.get_class_names()
        
        self.classifier = None
        self.run_manager = None

        # 自学习路由器（P3）—— 初始化为默认 entropy 策略，无 OOD
        self.self_learner = SelfLearner(
            strategy='entropy',
            auto_threshold=0.95,
            review_threshold=0.50,
            ood_detector=None,
        )
        
        self.current_image_path = None
        self.current_predictions = None
        self.review_queue = []
        self.current_review_index = 0
        
        self.confidence_threshold = 0.95
        self.review_threshold = 0.50

        self.prediction_records = []   # 批量预测记录（以文件路径为 key 的字典，便于后续人工审核覆写）  
        self.review_records = {}       # 人工审核记录：{原始文件路径: {类别, 时间}}  

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 绑定键盘快捷键  
        self.root.bind('<Left>', lambda e: self.prev_review())  
        self.root.bind('<Right>', lambda e: self.next_review())  
        self.root.bind('<Return>', lambda e: self.confirm_and_archive() if self.confirm_button['state'] == 'normal' else None)  
        self.root.bind('<Delete>', lambda e: self.delete_image() if self.delete_button['state'] == 'normal' else None)  

        style = ttk.Style()  
        style.configure('Danger.TButton', foreground='red', font=('Arial', 9, 'bold'))  

        self._create_widgets()
        self._load_review_queue()
    
    # ─────────────────────────────────────────────────────────────
    # THEME / STYLE
    # ─────────────────────────────────────────────────────────────
    def _apply_styles(self):
        s = ttk.Style()
        try:
            s.theme_use('clam')
        except Exception:
            pass

        FB  = ("Microsoft YaHei", 10)
        FBB = ("Microsoft YaHei", 10, "bold")
        FT  = ("Microsoft YaHei", 12, "bold")
        FS  = ("Microsoft YaHei", 9)
        FM  = ("Consolas", 9)

        s.configure('TFrame',         background='#F8FAFC')
        s.configure('Card.TFrame',    background='#FFFFFF',  relief='flat')
        s.configure('Header.TFrame',  background='#1E3A5F')

        s.configure('TLabel',         font=FB,  background='#FFFFFF')
        s.configure('Card.TLabel',    font=FB,  background='#FFFFFF', foreground='#374151')
        s.configure('Muted.TLabel',   font=FS,  background='#FFFFFF', foreground='#9CA3AF')
        s.configure('Title.TLabel',   font=FT,  background='#FFFFFF', foreground='#1E3A5F')
        s.configure('Value.TLabel',   font=FBB, background='#FFFFFF', foreground='#2563EB')
        s.configure('OK.TLabel',      font=FBB, background='#FFFFFF', foreground='#059669')
        s.configure('Warn.TLabel',    font=FBB, background='#FFFFFF', foreground='#D97706')
        s.configure('Err.TLabel',     font=FBB, background='#FFFFFF', foreground='#DC2626')
        s.configure('Step.TLabel',    font=FBB, background='#EFF6FF', foreground='#1E3A5F')

        s.configure('TLabelframe',        background='#FFFFFF', relief='groove', borderwidth=1)
        s.configure('TLabelframe.Label',  font=FBB, foreground='#1E3A5F', background='#FFFFFF')

        s.configure('TNotebook',          background='#EFF6FF', tabmargins=[2, 4, 0, 0])
        s.configure('TNotebook.Tab',      font=FBB, padding=[16, 7],
                    background='#DBEAFE', foreground='#1E3A5F')
        s.map('TNotebook.Tab',
              background=[('selected', '#2563EB'), ('active', '#93C5FD')],
              foreground=[('selected', '#FFFFFF'),  ('active', '#1E3A5F')])

        s.configure('TButton',          font=FB,  padding=[8, 4])
        s.configure('Primary.TButton',  font=FBB, padding=[10, 5])
        s.configure('Danger.TButton',   font=FBB, foreground='#DC2626')

        s.configure('Treeview',         font=FB,  rowheight=26, background='#FFFFFF',
                    fieldbackground='#FFFFFF', foreground='#374151')
        s.configure('Treeview.Heading', font=FBB, background='#EFF6FF', foreground='#1E3A5F')
        s.map('Treeview', background=[('selected', '#DBEAFE')])

        s.configure('TCombobox',  font=FB)
        s.configure('TSpinbox',   font=FB)
        s.configure('TCheckbutton', font=FB, background='#FFFFFF')
        s.configure('TProgressbar', thickness=6, background='#2563EB')
        s.configure('TScale',       background='#FFFFFF')
        s.configure('TSeparator',   background='#E2E8F0')

        self._FB = FB; self._FBB = FBB; self._FT = FT; self._FS = FS; self._FM = FM

    # ─────────────────────────────────────────────────────────────
    # MAIN WIDGET FACTORY
    # ─────────────────────────────────────────────────────────────
    def _create_widgets(self):
        self._apply_styles()
        self.root.configure(bg='#EFF6FF')

        # ── top header bar ────────────────────────────────────────
        hdr = tk.Frame(self.root, bg='#1E3A5F', height=54)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)
        self._build_header(hdr)

        # ── bottom status bar ─────────────────────────────────────
        bot = tk.Frame(self.root, bg='#E2E8F0', height=26)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        bot.pack_propagate(False)
        self._create_status_bar(bot)

        # ── body: notebook (left) + right panel ──────────────────
        body = tk.Frame(self.root, bg='#EFF6FF')
        body.pack(fill=tk.BOTH, expand=True)

        # right panel: fixed 390px
        right = tk.Frame(body, bg='#FFFFFF', bd=0)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)
        right.configure(width=390)
        right.pack_propagate(False)
        self._build_right_panel(right)

        # left: notebook fills remaining space
        nb_wrap = tk.Frame(body, bg='#EFF6FF')
        nb_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                     padx=(6, 3), pady=6)

        self.notebook = ttk.Notebook(nb_wrap)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        for title in ['①  数据准备', '②  自动标注', '③  模型训练',
                      '④  批量预测', '⑤  人工审核']:
            f = ttk.Frame(self.notebook, padding=12)
            f.configure(style='TFrame')
            self.notebook.add(f, text=title)

        tabs = self.notebook.tabs()
        self._build_tab_data(     self.notebook.nametowidget(tabs[0]))
        self._build_tab_annotate( self.notebook.nametowidget(tabs[1]))
        self._build_tab_train(    self.notebook.nametowidget(tabs[2]))
        self._build_tab_predict(  self.notebook.nametowidget(tabs[3]))
        self._build_tab_review(   self.notebook.nametowidget(tabs[4]))

    # ─────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────
    def _build_header(self, parent):
        tk.Label(parent, text='ALGAVISION  模型训练与自学习',
                 font=('Microsoft YaHei', 15, 'bold'),
                 bg='#1E3A5F', fg='#FFFFFF').pack(side=tk.LEFT, padx=18, pady=10)

        dev_txt = f'设备: {device}'
        if torch.cuda.is_available():
            try:
                gname = torch.cuda.get_device_name(0)
                gmem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                dev_txt = f'GPU: {gname} ({gmem:.1f} GB)'
            except Exception:
                pass
        self._header_dev_lbl = tk.Label(
            parent, text=dev_txt,
            font=('Microsoft YaHei', 9), bg='#1E3A5F', fg='#93C5FD')
        self._header_dev_lbl.pack(side=tk.RIGHT, padx=18)

        self._header_model_lbl = tk.Label(
            parent, text='未加载模型',
            font=('Microsoft YaHei', 9), bg='#1E3A5F', fg='#FCD34D')
        self._header_model_lbl.pack(side=tk.RIGHT, padx=6)

    def _header_update_model(self):
        if self.classifier and hasattr(self.classifier, 'model_type'):
            nc = getattr(self.classifier, 'num_classes', '?')
            self._header_model_lbl.config(
                text=f'模型: {self.classifier.model_type}  |  {nc} 类',
                fg='#6EE7B7')
        else:
            self._header_model_lbl.config(text='未加载模型', fg='#FCD34D')

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────
    def _card(self, parent, title, pady=(0, 8)):
        """Create a white-bg LabelFrame card."""
        f = ttk.LabelFrame(parent, text=title, padding=10)
        f.pack(fill=tk.X, pady=pady)
        return f

    def _row(self, parent, label, widget_factory, pady=3):
        """Create a label+widget row inside a card, returns widget."""
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=pady)
        ttk.Label(r, text=label, style='Card.TLabel',
                  width=10, anchor=tk.E).pack(side=tk.LEFT, padx=(0, 6))
        w = widget_factory(r)
        w.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return w

    def _dir_row(self, parent, label, dir_label_attr, cmd, default_text):
        """Directory picker row: label + path display + button."""
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=4)
        ttk.Label(f, text=label, style='Card.TLabel').pack(anchor=tk.W)
        inner = ttk.Frame(f); inner.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(inner, text=default_text,
                        style='Muted.TLabel', wraplength=340, anchor=tk.W)
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        setattr(self, dir_label_attr, lbl)
        ttk.Button(inner, text='浏览…', command=cmd, width=6
                   ).pack(side=tk.RIGHT, padx=(4, 0))

    def _sep(self, parent):
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=6)

    # ─────────────────────────────────────────────────────────────
    # TAB ① 数据准备
    # ─────────────────────────────────────────────────────────────
    def _build_tab_data(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        # ── step hint ─────────────────────────────────────────────
        hint = ttk.Frame(parent, style='TFrame')
        hint.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(hint,
            text='STEP 1 · 配置数据目录 — 所有路径设置完成后可进入训练或自动标注流程',
            style='Step.TLabel').pack(anchor=tk.W)

        # ── left col: train / archive / review ────────────────────
        left = ttk.LabelFrame(parent, text='数据目录配置', padding=12)
        left.grid(row=1, column=0, sticky='nsew', padx=(0, 5))

        self._dir_row(left, '训练数据集目录',
                      'train_dir_label', self.select_train_dir,
                      str(Path('./class/training_data').resolve()))
        self._sep(left)
        self._dir_row(left, '归档目录（伪标签 & 确认样本）',
                      'archive_dir_label', self.select_archive_dir,
                      str(Path('./class/archived_data').resolve()))
        self._sep(left)
        self._dir_row(left, '待审核队列目录',
                      'review_dir_label', self.select_review_dir,
                      str(Path('./class/review_queue').resolve()))
        self._sep(left)
        self._dir_row(left, '风格来源目录（可选）',
                      'style_dir_label', self.select_style_dir,
                      '未设置 — 选「风格迁移」增强时生效')

        # ── right col: dataset statistics ─────────────────────────
        right = ttk.LabelFrame(parent, text='数据集概览', padding=12)
        right.grid(row=1, column=1, sticky='nsew', padx=(5, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # stat cards row
        cards = ttk.Frame(right); cards.pack(fill=tk.X, pady=(0, 8))
        for i, (attr, title, color) in enumerate([
                ('_stat_total',     '总样本',   '#2563EB'),
                ('_stat_confirmed', '已确认',   '#059669'),
                ('_stat_pseudo',    '伪标签',   '#D97706'),
                ('_stat_review',    '待审核',   '#7C3AED'),
                ('_stat_classes',   '类别数',   '#0891B2'),
        ]):
            c = tk.Frame(cards, bg='#F8FAFC', bd=1, relief='solid')
            c.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0 if i==0 else 4, 0))
            tk.Label(c, text=title, font=('Microsoft YaHei', 8),
                     bg='#F8FAFC', fg='#6B7280').pack(pady=(6, 0))
            lbl = tk.Label(c, text='–', font=('Microsoft YaHei', 16, 'bold'),
                           bg='#F8FAFC', fg=color)
            lbl.pack(pady=(0, 6))
            setattr(self, attr, lbl)

        # class list
        cls_frame = ttk.LabelFrame(right, text='类别列表', padding=6)
        cls_frame.pack(fill=tk.BOTH, expand=True)
        cls_frame.rowconfigure(0, weight=1)
        cls_frame.columnconfigure(0, weight=1)

        self.stats_text = tk.Text(cls_frame, font=('Consolas', 9),
                                  bg='#F8FAFC', fg='#374151',
                                  relief='flat', wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(cls_frame, command=self.stats_text.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.stats_text.configure(yscrollcommand=sb.set)
        self.stats_text.tag_configure('title', font=('Microsoft YaHei', 10, 'bold'),
                                      foreground='#1E3A5F')
        self.stats_text.tag_configure('value', font=('Microsoft YaHei', 10, 'bold'),
                                      foreground='#059669')

    # ─────────────────────────────────────────────────────────────
    # TAB ② 自动标注
    # ─────────────────────────────────────────────────────────────
    def _build_tab_annotate(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        hint = ttk.Frame(parent); hint.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(hint,
            text='STEP 2 · 自动标注 — 对无标签图像进行无监督聚类命名，或配置主动学习采样策略',
            style='Step.TLabel').pack(anchor=tk.W)

        # ── left: unsupervised annotation ─────────────────────────
        lf = ttk.LabelFrame(parent, text='无监督聚类标注', padding=14)
        lf.grid(row=1, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(lf,
            text=('从无标签图像 (FlowCAM / 显微镜) 中自动提取特征 → 聚类 → '
                  '人工对每个簇一次性命名，生成可直接用于训练的标注数据集。'),
            style='Muted.TLabel', wraplength=340, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Separator(lf, orient='horizontal').pack(fill=tk.X, pady=10)

        # embedding backbone info
        info_f = ttk.Frame(lf); info_f.pack(fill=tk.X, pady=4)
        ttk.Label(info_f, text='嵌入骨干:', style='Card.TLabel', width=10,
                  anchor=tk.E).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(info_f,
                  text='DINOv2-S / DINOv2-B（自监督，无需标签）\n'
                       '或任意注册表模型 (num_classes=0)',
                  style='Muted.TLabel').pack(side=tk.LEFT)

        # feature list
        for txt in ('■  支持 HDBSCAN (自动簇数) / KMeans',
                    '■  噪声点 (-1) 对应碎屑 / 杂质，自动隔离',
                    '■  每簇展示代表图，一键命名即导出类别目录',
                    '■  完成后可直接设为训练数据目录'):
            ttk.Label(lf, text=txt, style='Muted.TLabel').pack(anchor=tk.W, pady=1)

        ttk.Separator(lf, orient='horizontal').pack(fill=tk.X, pady=10)

        self.unsup_btn = ttk.Button(lf, text='🔬  启动无监督标注工具',
                                    command=self.open_unsupervised_annotator,
                                    style='Primary.TButton')
        self.unsup_btn.pack(fill=tk.X, ipady=4)

        # ── right: self-learning strategy ────────────────────────
        rf = ttk.LabelFrame(parent, text='主动学习策略 (批量预测路由)', padding=14)
        rf.grid(row=1, column=1, sticky='nsew', padx=(5, 0))

        ttk.Label(rf,
            text='批量预测时，自学习路由器根据不确定度对样本分流：\n'
                 '高置信 → 伪标注归档  |  中置信 / OOD → 人工审核  |  低置信 → 丢弃',
            style='Muted.TLabel', wraplength=340, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Separator(rf, orient='horizontal').pack(fill=tk.X, pady=10)

        # strategy selector
        sf = ttk.Frame(rf); sf.pack(fill=tk.X, pady=4)
        ttk.Label(sf, text='采样策略:', style='Card.TLabel',
                  width=12, anchor=tk.E).pack(side=tk.LEFT, padx=(0, 6))
        self.sl_strategy_var = tk.StringVar(value='entropy')
        ttk.Combobox(sf, textvariable=self.sl_strategy_var,
                     values=SelfLearner.STRATEGIES,
                     state='readonly', width=20).pack(side=tk.LEFT)

        # strategy descriptions
        desc = {
            'entropy':           '信息熵 — 综合不确定度，推荐',
            'margin':            '边际差 — top2 概率差越小越不确定',
            'least_confidence':  '最小置信度 — 1 - max_prob',
            'coreset':           'Core-Set — 嵌入空间多样性',
            'max_softmax':       'Max Softmax — 基础基线',
        }
        desc_lbl = ttk.Label(rf, text='↑ 信息熵: 综合不确定度，推荐',
                             style='Muted.TLabel')
        desc_lbl.pack(anchor=tk.W, pady=(2, 8))

        def _on_strategy(*_):
            s = self.sl_strategy_var.get()
            desc_lbl.config(text=f'↑ {s}: {desc.get(s, "")}')
        self.sl_strategy_var.trace_add('write', _on_strategy)

        ttk.Separator(rf, orient='horizontal').pack(fill=tk.X, pady=6)

        # OOD
        ood_f = ttk.Frame(rf); ood_f.pack(fill=tk.X, pady=4)
        self.sl_ood_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ood_f, text='启用 OOD 检测 (Energy Score)',
                        variable=self.sl_ood_var).pack(anchor=tk.W)
        ttk.Label(rf,
            text='OOD 样本 (碎屑/气泡/域外生物) 强制送审核，不计入伪标签归档。',
            style='Muted.TLabel', wraplength=340, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    # ─────────────────────────────────────────────────────────────
    # TAB ③ 模型训练
    # ─────────────────────────────────────────────────────────────
    def _build_tab_train(self, parent):
        parent.columnconfigure(0, weight=0, minsize=340)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        hint = ttk.Frame(parent); hint.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(hint,
            text='STEP 3 · 模型训练 — 配置模型结构与超参数，执行训练或微调',
            style='Step.TLabel').pack(anchor=tk.W)

        # ── left config panel ─────────────────────────────────────
        cfg = ttk.Frame(parent); cfg.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        cfg.columnconfigure(0, weight=1)

        # Model architecture card
        mc = self._card(cfg, '骨干网络')
        ttk.Label(mc, text='架构:', style='Card.TLabel').grid(
            row=0, column=0, sticky='e', padx=(0, 6), pady=3)
        self.model_var = tk.StringVar(value='B2')
        mc_cb = ttk.Combobox(mc, textvariable=self.model_var,
                             values=list(PhytoplanktonClassifier.MODEL_CONFIGS.keys()),
                             state='readonly', width=20)
        mc_cb.grid(row=0, column=1, sticky='ew', pady=3)
        mc_cb.bind('<<ComboboxSelected>>', self._on_model_selected)
        mc.columnconfigure(1, weight=1)

        self.model_info_label = ttk.Label(mc, text='推荐生产环境 | ~10M 参数 | 260×260',
                                          style='Muted.TLabel', wraplength=280)
        self.model_info_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 0))

        ttk.Label(mc, text='Epochs:', style='Card.TLabel').grid(
            row=2, column=0, sticky='e', padx=(0, 6), pady=(6, 3))
        self.epochs_var = tk.IntVar(value=20)
        ttk.Spinbox(mc, from_=1, to=200, textvariable=self.epochs_var, width=8
                    ).grid(row=2, column=1, sticky='w', pady=(6, 3))

        # Hyperparams card
        hc = self._card(cfg, '超参数')

        def _hp_row(r, label, widget_factory):
            ttk.Label(hc, text=label, style='Card.TLabel', width=9, anchor=tk.E
                      ).grid(row=r, column=0, sticky='e', padx=(0, 6), pady=3)
            w = widget_factory()
            w.grid(row=r, column=1, sticky='ew', pady=3)
            return w
        hc.columnconfigure(1, weight=1)

        self.lr_var = tk.StringVar(value='1e-3')
        _hp_row(0, '学习率:', lambda: ttk.Combobox(
            hc, textvariable=self.lr_var,
            values=['1e-2','5e-3','1e-3','5e-4','1e-4','5e-5','1e-5'],
            state='readonly', width=12))

        self.batch_var = tk.IntVar(value=32)
        self.auto_batch_var = tk.BooleanVar(value=False)
        batch_f = ttk.Frame(hc)
        self.batch_combo = ttk.Combobox(
            batch_f, textvariable=self.batch_var,
            values=[8,16,24,32,48,64,96,128], state='readonly', width=8)
        self.batch_combo.pack(side=tk.LEFT)
        ttk.Checkbutton(batch_f, text='自动', variable=self.auto_batch_var,
                        command=self.toggle_auto_batch).pack(side=tk.LEFT, padx=(6,0))
        self.auto_batch_label = ttk.Label(batch_f, text='', style='Muted.TLabel')
        self.auto_batch_label.pack(side=tk.LEFT, padx=(4,0))
        ttk.Label(hc, text='批次:', style='Card.TLabel', width=9, anchor=tk.E
                  ).grid(row=1, column=0, sticky='e', padx=(0, 6), pady=3)
        batch_f.grid(row=1, column=1, sticky='ew', pady=3)

        self.optimizer_var = tk.StringVar(value='AdamW')
        _hp_row(2, '优化器:', lambda: ttk.Combobox(
            hc, textvariable=self.optimizer_var,
            values=['Adam','AdamW','SGD','RMSprop'], state='readonly', width=12))

        self.scheduler_var = tk.StringVar(value='CosineAnnealing')
        _hp_row(3, 'LR调度:', lambda: ttk.Combobox(
            hc, textvariable=self.scheduler_var,
            values=['ReduceLROnPlateau','CosineAnnealing','StepLR','None'],
            state='readonly', width=16))

        self.weight_decay_var = tk.StringVar(value='1e-4')
        _hp_row(4, '权重衰减:', lambda: ttk.Combobox(
            hc, textvariable=self.weight_decay_var,
            values=['0','1e-5','1e-4','1e-3'], state='readonly', width=12))

        self.patience_var = tk.IntVar(value=5)
        _hp_row(5, '早停:', lambda: ttk.Spinbox(
            hc, from_=1, to=30, textvariable=self.patience_var, width=8))

        self.augmentation_var = tk.StringVar(value='显微镜增强')
        _hp_row(6, '增强:', lambda: ttk.Combobox(
            hc, textvariable=self.augmentation_var,
            values=['无','轻度','中等','强烈','显微镜增强','风格迁移'],
            state='readonly', width=14))

        # Action buttons card
        ac = self._card(cfg, '训练操作')

        btn_specs = [
            ('initial_training_btn',  '▶  开始训练',  'Primary.TButton', self.initial_training),
            ('fine_tune_btn',         '⟳  微调模型',  'TButton',         self.fine_tune_model),
            ('load_model_btn',        '📂  加载模型',  'TButton',         self.load_model),
            ('save_model_btn',        '💾  保存模型',  'TButton',         self.save_model),
        ]
        for r, (attr, text, style, cmd) in enumerate(btn_specs):
            b = ttk.Button(ac, text=text, style=style, command=cmd)
            b.pack(fill=tk.X, pady=3, ipady=3)
            setattr(self, attr, b)

        ttk.Separator(ac, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(ac, text='🔧  超参数自动调优',
                   command=self.hyperparameter_tuning).pack(fill=tk.X, pady=3)

        # ── right: training curve area ────────────────────────────
        rc = ttk.LabelFrame(parent, text='训练进度', padding=10)
        rc.grid(row=1, column=1, sticky='nsew')
        rc.columnconfigure(0, weight=1)
        rc.rowconfigure(1, weight=1)

        # placeholder for TrainingProgressWindow embed
        self._train_curve_frame = rc
        ttk.Label(rc,
            text='训练开始后此处将显示 Loss / Accuracy 实时曲线',
            style='Muted.TLabel').pack(anchor=tk.W)

        # training history log (text widget)
        self._train_log = tk.Text(rc, font=('Consolas', 9), height=8,
                                  bg='#F8FAFC', fg='#374151', relief='flat',
                                  wrap=tk.WORD)
        self._train_log.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        sb = ttk.Scrollbar(rc, command=self._train_log.yview)
        self._train_log.configure(yscrollcommand=sb.set)

    # ─────────────────────────────────────────────────────────────
    # TAB ④ 批量预测
    # ─────────────────────────────────────────────────────────────
    def _build_tab_predict(self, parent):
        parent.columnconfigure(0, weight=0, minsize=300)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        hint = ttk.Frame(parent); hint.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(hint,
            text='STEP 4 · 批量预测 — 对无标签图像推理，自学习路由自动分流至归档 / 审核 / 丢弃',
            style='Step.TLabel').pack(anchor=tk.W)

        # ── left: thresholds + strategy ──────────────────────────
        lf = ttk.Frame(parent); lf.grid(row=1, column=0, sticky='nsew', padx=(0, 6))

        # threshold card
        tc = self._card(lf, '置信度阈值')

        def _thr_row(label, var, lbl_attr, from_, to, default, cmd):
            f = ttk.Frame(tc); f.pack(fill=tk.X, pady=5)
            ttk.Label(f, text=label, style='Card.TLabel').pack(anchor=tk.W)
            inner = ttk.Frame(f); inner.pack(fill=tk.X)
            sc = ttk.Scale(inner, from_=from_, to=to, variable=var,
                           orient=tk.HORIZONTAL, command=cmd)
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = ttk.Label(inner, text=f'{default:.2f}',
                            style='Value.TLabel', width=5)
            lbl.pack(side=tk.LEFT, padx=(6, 0))
            setattr(self, lbl_attr, lbl)
            return sc

        self.threshold_var = tk.DoubleVar(value=0.90)
        _thr_row('自动归档阈值 — 置信度 ≥ 此值 → 伪标注',
                 self.threshold_var, 'threshold_label',
                 0.50, 0.99, 0.90, self.update_threshold_label)

        ttk.Label(tc, text='', style='Muted.TLabel').pack()  # spacer

        self.review_threshold_var = tk.DoubleVar(value=0.50)
        _thr_row('审核阈值 — 置信度 ≥ 此值 → 人工审核',
                 self.review_threshold_var, 'review_threshold_label',
                 0.0, 0.89, 0.50, self.update_review_threshold_label)

        ttk.Label(tc,
            text='置信度 < 审核阈值 → 丢弃（不占用审核队列）',
            style='Muted.TLabel', wraplength=260).pack(anchor=tk.W, pady=(6, 0))

        # action card
        bc = self._card(lf, '预测操作')
        self.select_image_btn = ttk.Button(
            bc, text='🔍  选择单张图像预测', command=self.select_image)
        self.select_image_btn.pack(fill=tk.X, pady=3, ipady=3)

        self.batch_predict_btn = ttk.Button(
            bc, text='⚡  批量预测文件夹',
            command=self.batch_predict, style='Primary.TButton')
        self.batch_predict_btn.pack(fill=tk.X, pady=3, ipady=3)

        ttk.Separator(bc, orient='horizontal').pack(fill=tk.X, pady=6)

        ttk.Button(bc, text='📊  导出 Excel 报告',
                   command=self.export_to_excel).pack(fill=tk.X, pady=3)

        # ── right: batch stats ────────────────────────────────────
        rf = ttk.LabelFrame(parent, text='本轮预测统计', padding=14)
        rf.grid(row=1, column=1, sticky='nsew')
        rf.columnconfigure((0, 1, 2, 3), weight=1)

        for i, (attr, label, color) in enumerate([
                ('_pred_stat_total',   '已处理',  '#374151'),
                ('_pred_stat_auto',    '伪标注归档', '#059669'),
                ('_pred_stat_review',  '送审核',  '#D97706'),
                ('_pred_stat_discard', '已丢弃',  '#6B7280'),
        ]):
            c = tk.Frame(rf, bg='#F8FAFC', bd=1, relief='solid')
            c.grid(row=0, column=i, sticky='nsew', padx=(0 if i==0 else 4, 0), pady=(0, 10))
            tk.Label(c, text=label, font=('Microsoft YaHei', 9),
                     bg='#F8FAFC', fg='#6B7280').pack(pady=(8, 0))
            l = tk.Label(c, text='–', font=('Microsoft YaHei', 20, 'bold'),
                         bg='#F8FAFC', fg=color)
            l.pack(pady=(0, 8))
            setattr(self, attr, l)

        # result treeview (mirrors right panel, for batch overview)
        ttk.Label(rf, text='最近预测结果（亦显示于右侧面板）',
                  style='Muted.TLabel').grid(row=1, column=0, columnspan=4, sticky='w', pady=(0, 4))

    # ─────────────────────────────────────────────────────────────
    # TAB ⑤ 人工审核
    # ─────────────────────────────────────────────────────────────
    def _build_tab_review(self, parent):
        parent.columnconfigure(0, weight=0, minsize=280)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        hint = ttk.Frame(parent); hint.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(hint,
            text='STEP 5 · 人工审核 — 对低置信 / OOD 样本确认类别后归档，闭合自学习循环',
            style='Step.TLabel').pack(anchor=tk.W)

        # ── left: queue + navigation ─────────────────────────────
        lf = ttk.LabelFrame(parent, text='审核队列', padding=12)
        lf.grid(row=1, column=0, sticky='nsew', padx=(0, 6))

        # count
        cnt_f = ttk.Frame(lf); cnt_f.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(cnt_f, text='待审核:', style='Card.TLabel').pack(side=tk.LEFT)
        self.review_count_label = ttk.Label(
            cnt_f, text='0', style='Err.TLabel')
        self.review_count_label.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(cnt_f, text='张', style='Card.TLabel').pack(side=tk.LEFT)

        self._sep(lf)

        # nav
        nav = ttk.Frame(lf); nav.pack(fill=tk.X, pady=4)
        ttk.Button(nav, text='◀  上一张', command=self.prev_review,
                   width=12).pack(side=tk.LEFT)
        ttk.Button(nav, text='下一张  ▶', command=self.next_review,
                   width=12).pack(side=tk.RIGHT)

        self.review_index_label = ttk.Label(lf, text='', style='Muted.TLabel')
        self.review_index_label.pack(pady=4)

        self.review_prediction_label = ttk.Label(
            lf, text='', style='Value.TLabel', wraplength=240, justify=tk.LEFT)
        self.review_prediction_label.pack(anchor=tk.W, pady=4)

        self._sep(lf)

        ttk.Label(lf, text='快捷键', style='Card.TLabel').pack(anchor=tk.W)
        for hint_txt in ('← / → 切换图像', 'Enter 确认归档', 'Del 删除图像'):
            ttk.Label(lf, text=f'  {hint_txt}', style='Muted.TLabel').pack(anchor=tk.W)

        self._sep(lf)

        ttk.Button(lf, text='🗑  清空审核队列',
                   command=self.clear_all_review_images,
                   style='Danger.TButton').pack(fill=tk.X)

        # ── right: correct + archive ──────────────────────────────
        rf = ttk.LabelFrame(parent, text='分类操作', padding=14)
        rf.grid(row=1, column=1, sticky='nsew')
        rf.columnconfigure(0, weight=1)

        # class correction
        ttk.Label(rf, text='修正为:', style='Card.TLabel').pack(anchor=tk.W)
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(
            rf, textvariable=self.class_var, state='readonly', width=28)
        self.class_combo.pack(fill=tk.X, pady=(4, 10))
        self.class_combo.bind('<<ComboboxSelected>>', self._on_class_selected)

        # action buttons
        self.confirm_button = ttk.Button(
            rf, text='✓  确认归档到所选类别',
            command=self.confirm_and_archive,
            style='Primary.TButton', state='disabled')
        self.confirm_button.pack(fill=tk.X, pady=4, ipady=4)

        self.new_class_button = ttk.Button(
            rf, text='＋  新建类别并归档',
            command=self.create_new_class)
        self.new_class_button.pack(fill=tk.X, pady=4, ipady=4)

        self.delete_button = ttk.Button(
            rf, text='✗  删除此图像（不归档）',
            command=self.delete_image,
            style='Danger.TButton', state='disabled')
        self.delete_button.pack(fill=tk.X, pady=4, ipady=4)

        ttk.Separator(rf, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(rf, text='审核完成后可返回「③ 模型训练」→ 微调模型，完成自学习迭代。',
                  style='Muted.TLabel', wraplength=340,
                  justify=tk.LEFT).pack(anchor=tk.W)

    # ─────────────────────────────────────────────────────────────
    # RIGHT PANEL (image preview + result tree + log)
    # ─────────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        # Image preview
        img_lf = ttk.LabelFrame(parent, text='图像预览', padding=4)
        img_lf.pack(fill=tk.X, pady=(0, 4))

        self.image_label = ttk.Label(img_lf, anchor=tk.CENTER)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        # reserve space with a placeholder
        img_lf.configure(height=320)
        img_lf.pack_propagate(False)

        # Prediction result tree
        pred_lf = ttk.LabelFrame(parent, text='预测结果', padding=4)
        pred_lf.pack(fill=tk.X, pady=4)
        pred_lf.configure(height=160)
        pred_lf.pack_propagate(False)

        cols = ('类别', '置信度', '状态')
        self.result_tree = ttk.Treeview(
            pred_lf, columns=cols, show='headings', height=5)
        for col, w in zip(cols, (140, 90, 90)):
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=w, anchor=tk.CENTER)
        self.result_tree.tag_configure('high',   background='#D1FAE5')
        self.result_tree.tag_configure('medium', background='#FEF3C7')
        self.result_tree.tag_configure('low',    background='#FEE2E2')
        vsb = ttk.Scrollbar(pred_lf, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=vsb.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Log panel
        log_lf = ttk.LabelFrame(parent, text='操作日志', padding=4)
        log_lf.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.logger_text = tk.Text(
            log_lf, font=('Consolas', 9), bg='#0F172A', fg='#E2E8F0',
            relief='flat', wrap=tk.WORD, insertbackground='white')
        self.logger_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(log_lf, command=self.logger_text.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.logger_text.configure(yscrollcommand=lsb.set)

        self.logger_text.tag_configure('INFO',    foreground='#93C5FD')
        self.logger_text.tag_configure('SUCCESS', foreground='#6EE7B7')
        self.logger_text.tag_configure('WARNING', foreground='#FCD34D')
        self.logger_text.tag_configure('ERROR',   foreground='#FCA5A5')
        self.logger_text.tag_configure('HEADER',  foreground='#C4B5FD',
                                        font=('Consolas', 9, 'bold'))

    # ─────────────────────────────────────────────────────────────
    # STATUS BAR
    # ─────────────────────────────────────────────────────────────
    def _create_status_bar(self, parent):
        self.status_var = tk.StringVar(value=f'就绪  |  设备: {device}')
        tk.Label(parent, textvariable=self.status_var,
                 font=('Microsoft YaHei', 9),
                 bg='#E2E8F0', fg='#374151',
                 anchor=tk.W, padx=10).pack(fill=tk.X, expand=True)
       
    def _create_control_panel_LEGACY(self, parent):
        """已废弃，保留仅供参考"""
        control_container = ttk.Frame(parent)
        control_container.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_container.rowconfigure(0, weight=1)
        control_container.columnconfigure(0, weight=1)
        
        canvas = tk.Canvas(control_container, width=350, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand = True) 
        
        v_scrollbar = ttk.Scrollbar(control_container, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y) 
        canvas.configure(yscrollcommand=v_scrollbar.set)
        
        control_frame = ttk.LabelFrame(canvas, text="控制面板", padding="5")
        canvas_frame = canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)
        
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = control_container.winfo_width() - v_scrollbar.winfo_width() - 5
            canvas.itemconfig(canvas_frame, width=canvas_width)
        
        control_frame.bind('<Configure>', on_configure)

        control_container.update_idletasks()  
        
        def on_mousewheel(event):  
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")  
        
        def bind_mousewheel(event):  
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):  
            canvas.unbind_all("<MouseWheel>")

        canvas.bind('<Enter>', bind_mousewheel)  
        canvas.bind('<Leave>', unbind_mousewheel) 

        # ========== 数据集设置区 ==========
        dataset_section = CollapsibleFrame(control_frame, text="数据集设置")
        dataset_section.pack(fill=tk.X, pady=5, padx=5)

        train_dir_frame = ttk.Frame(dataset_section.content_frame)
        train_dir_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(train_dir_frame, text="训练数据集:").pack(anchor=tk.W)
        ttk.Button(train_dir_frame, text="选择训练数据目录", command=self.select_train_dir, width=26).pack(pady=2)
        self.train_dir_label = ttk.Label(train_dir_frame, text="未选择", wraplength=280, foreground="gray")
        self.train_dir_label.pack(pady=2)

        archive_dir_frame = ttk.Frame(dataset_section.content_frame)
        archive_dir_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(archive_dir_frame, text="归档目录(可选):").pack(anchor=tk.W)
        ttk.Button(archive_dir_frame, text="选择归档目录", command=self.select_archive_dir, width=26).pack(pady=2)
        self.archive_dir_label = ttk.Label(archive_dir_frame, text="未选择（默认: ./class/archived_data）", wraplength=280, foreground="gray")
        self.archive_dir_label.pack(pady=2)

        review_dir_frame = ttk.Frame(dataset_section.content_frame)
        review_dir_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(review_dir_frame, text="待审核目录:").pack(anchor=tk.W)
        ttk.Button(review_dir_frame, text="选择待审核目录", command=self.select_review_dir, width=26).pack(pady=2)
        self.review_dir_label = ttk.Label(review_dir_frame, text="未选择", wraplength=280, foreground="gray")
        self.review_dir_label.pack(pady=2)

        style_dir_frame = ttk.Frame(dataset_section.content_frame)
        style_dir_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(style_dir_frame, text="风格来源目录(可选):").pack(anchor=tk.W)
        ttk.Label(
            style_dir_frame,
            text="选择 FlowCAM/显微镜目标域图像文件夹\n（增强级别选「风格迁移」时生效）",
            font=("Arial", 8), foreground="gray",
        ).pack(anchor=tk.W)
        ttk.Button(style_dir_frame, text="选择风格来源目录",
                   command=self.select_style_dir, width=26).pack(pady=2)
        self.style_dir_label = ttk.Label(
            style_dir_frame, text="未选择", wraplength=280, foreground="gray")
        self.style_dir_label.pack(pady=2)

        ttk.Separator(dataset_section.content_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=5)

        # ========== 无监督标注区 ==========
        unsup_section = CollapsibleFrame(control_frame, text="无监督自动标注 (P2)")
        unsup_section.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(
            unsup_section.content_frame,
            text="从无标签图像自动聚类，人工一键命名各簇，\n生成可直接用于训练的标注数据集。",
            font=("Arial", 8), foreground="gray", justify=tk.LEFT,
        ).pack(pady=(4, 2), padx=5)

        self.unsup_btn = ttk.Button(
            unsup_section.content_frame,
            text="🔬 启动无监督标注工具",
            command=self.open_unsupervised_annotator,
            width=30,
        )
        self.unsup_btn.pack(pady=5, padx=5)
        unsup_section.toggle()  # 默认折叠

        # ========== 自学习策略区 (P3) ==========
        sl_section = CollapsibleFrame(control_frame, text="自学习策略 (P3)")
        sl_section.pack(fill=tk.X, pady=5, padx=5)

        strategy_frame = ttk.Frame(sl_section.content_frame)
        strategy_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(strategy_frame, text="采样策略:").pack(side=tk.LEFT)
        self.sl_strategy_var = tk.StringVar(value="entropy")
        ttk.Combobox(
            strategy_frame, textvariable=self.sl_strategy_var,
            values=SelfLearner.STRATEGIES, state="readonly", width=18,
        ).pack(side=tk.LEFT, padx=4)

        ood_frame = ttk.Frame(sl_section.content_frame)
        ood_frame.pack(fill=tk.X, pady=2, padx=5)
        self.sl_ood_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ood_frame, text="启用 OOD 检测（Energy Score）",
                        variable=self.sl_ood_var).pack(anchor=tk.W)

        ttk.Label(
            sl_section.content_frame,
            text="策略影响审核队列优先排序；OOD 样本强制送审核，不自动归档。",
            font=("Arial", 8), foreground="gray", wraplength=280, justify=tk.LEFT,
        ).pack(pady=4, padx=5)
        sl_section.toggle()  # 默认折叠

        # ========== 模型配置区 ==========
        model_config_section = CollapsibleFrame(control_frame, text="模型配置")
        model_config_section.pack(fill=tk.X, pady=5, padx=2)
        
        ttk.Label(model_config_section.content_frame, text="选择模型精度:").pack(pady=(5, 2))
        self.model_var = tk.StringVar(value='B2')
        model_combo = ttk.Combobox(model_config_section.content_frame, textvariable=self.model_var, values=list(PhytoplanktonClassifier.MODEL_CONFIGS.keys()), state="readonly", width=25)
        model_combo.pack(pady=5, padx=5) 
        model_combo.bind('<<ComboboxSelected>>', self._on_model_selected)
        
        self.model_info_label = ttk.Label(model_config_section.content_frame, text="推荐用于生产环境/参数量: ~10M", wraplength=300, justify=tk.LEFT, foreground="blue")
        self.model_info_label.pack(pady=5, padx=5)

        epochs_frame = ttk.Frame(model_config_section.content_frame)
        epochs_frame.pack(pady=10, fill=tk.X)
        ttk.Label(epochs_frame, text="训练轮数:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=10)
        epochs_spinbox = ttk.Spinbox(epochs_frame, from_=1, to=100, textvariable=self.epochs_var, width=12)
        epochs_spinbox.pack(side=tk.LEFT, padx=5)
        
        # ========== 超参数配置区 ==========
        hyperparam_section = CollapsibleFrame(control_frame, text="超参数配置")
        hyperparam_section.pack(fill=tk.X, pady=5, padx=5)
        hyperparam_section.toggle()  # 默认折叠
        
        lr_frame = ttk.Frame(hyperparam_section.content_frame)
        lr_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(lr_frame, text="学习率:").pack(side=tk.LEFT, padx=(0, 5)) 
        self.lr_var = tk.StringVar(value="1e-3")
        lr_combo = ttk.Combobox(lr_frame, textvariable=self.lr_var, values=["1e-2", "5e-3", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5"], width=12, state="readonly")
        lr_combo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)) 
        
        batch_frame = ttk.Frame(hyperparam_section.content_frame)
        batch_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(batch_frame, text="批次大小:").pack(side=tk.LEFT, padx=(0, 5))

        self.auto_batch_var = tk.BooleanVar(value=False)
        auto_batch_check = ttk.Checkbutton(batch_frame, text="自动", variable=self.auto_batch_var, command=self.toggle_auto_batch)
        auto_batch_check.pack(side=tk.LEFT, padx=5)
        
        self.batch_var = tk.IntVar(value=32)
        self.batch_combo = ttk.Combobox(batch_frame, textvariable=self.batch_var, values=[8, 16, 24, 32, 48, 64, 96, 128], width=10, state="readonly")
        self.batch_combo.pack(side=tk.LEFT, padx=5)
        
        self.auto_batch_label = ttk.Label(batch_frame, text="", foreground="blue")
        self.auto_batch_label.pack(side=tk.LEFT, padx=5)
        
        optimizer_frame = ttk.Frame(hyperparam_section.content_frame)
        optimizer_frame.pack(fill=tk.X, pady=2)
        ttk.Label(optimizer_frame, text="优化器:").pack(side=tk.LEFT)
        self.optimizer_var = tk.StringVar(value="Adam")
        optimizer_combo = ttk.Combobox(optimizer_frame, textvariable=self.optimizer_var, values=["Adam", "AdamW", "SGD", "RMSprop"], width=10, state="readonly")
        optimizer_combo.pack(side=tk.LEFT, padx=5)
        
        scheduler_frame = ttk.Frame(hyperparam_section.content_frame)
        scheduler_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scheduler_frame, text="LR调度:").pack(side=tk.LEFT)
        self.scheduler_var = tk.StringVar(value="ReduceLROnPlateau")
        scheduler_combo = ttk.Combobox(scheduler_frame, textvariable=self.scheduler_var, values=["ReduceLROnPlateau", "CosineAnnealing", "StepLR", "None"], width=10, state="readonly")
        scheduler_combo.pack(side=tk.LEFT, padx=5)
        
        weight_decay_frame = ttk.Frame(hyperparam_section.content_frame)
        weight_decay_frame.pack(fill=tk.X, pady=2)
        ttk.Label(weight_decay_frame, text="权重衰减:").pack(side=tk.LEFT)
        self.weight_decay_var = tk.StringVar(value="1e-4")
        weight_decay_combo = ttk.Combobox(weight_decay_frame, textvariable=self.weight_decay_var, values=["0", "1e-5", "1e-4", "1e-3", "1e-2"], width=10, state="readonly")
        weight_decay_combo.pack(side=tk.LEFT, padx=5)
        
        patience_frame = ttk.Frame(hyperparam_section.content_frame)
        patience_frame.pack(fill=tk.X, pady=2)
        ttk.Label(patience_frame, text="早停耐心:").pack(side=tk.LEFT)
        self.patience_var = tk.IntVar(value=3)
        patience_spinbox = ttk.Spinbox(patience_frame, from_=1, to=20, textvariable=self.patience_var, width=10)
        patience_spinbox.pack(side=tk.LEFT, padx=5)
        
        aug_frame = ttk.Frame(hyperparam_section.content_frame)
        aug_frame.pack(fill=tk.X, pady=2)
        ttk.Label(aug_frame, text="数据增强:").pack(side=tk.LEFT)
        self.augmentation_var = tk.StringVar(value="中等")
        aug_combo = ttk.Combobox(
            aug_frame, textvariable=self.augmentation_var,
            values=["无", "轻度", "中等", "强烈", "显微镜增强", "风格迁移"],
            width=14, state="readonly")
        aug_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(hyperparam_section.content_frame, text="自动调优", command=self.hyperparameter_tuning, width=20).pack(pady=5)
        
        # ========== 模型训练区 ==========
        train_section = CollapsibleFrame(control_frame, text="模型训练")
        train_section.pack(fill=tk.X, pady=5, padx=2)

        self.initial_training_btn = ttk.Button(train_section.content_frame, text="初始训练", command=self.initial_training, width=30)  
        self.initial_training_btn.pack(pady=3, padx=5)  
        
        self.fine_tune_btn = ttk.Button(train_section.content_frame, text="微调模型", command=self.fine_tune_model, width=30)  
        self.fine_tune_btn.pack(pady=3, padx=5)  
        
        self.load_model_btn = ttk.Button(train_section.content_frame, text="加载模型", command=self.load_model, width=30)  
        self.load_model_btn.pack(pady=3, padx=5)  
        
        self.save_model_btn = ttk.Button(train_section.content_frame, text="保存模型", command=self.save_model, width=30)  
        self.save_model_btn.pack(pady=3, padx=5)  

        # ========== 预测区 ==========
        predict_section = CollapsibleFrame(control_frame, text="图像预测")  
        predict_section.pack(fill=tk.X, pady=5, padx=2)  

        self.select_image_btn = ttk.Button(predict_section.content_frame, text="选择图像", command=self.select_image, width=30)  
        self.select_image_btn.pack(pady=3, padx=5)  
        
        self.batch_predict_btn = ttk.Button(predict_section.content_frame, text="批量预测", command=self.batch_predict, width=30)  
        self.batch_predict_btn.pack(pady=3, padx=5) 

        threshold_frame = ttk.Frame(predict_section.content_frame)
        threshold_frame.pack(pady=10, fill=tk.X)
        ttk.Label(threshold_frame, text="自动归档阈值:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.90)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=0.99, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.95")
        self.threshold_label.pack(side=tk.LEFT)
        threshold_scale.configure(command=self.update_threshold_label)
                # 新增：人工审核阈值控件  
        review_threshold_frame = ttk.Frame(predict_section.content_frame)  
        review_threshold_frame.pack(pady=5, fill=tk.X)  
        ttk.Label(review_threshold_frame, text="人工审核阈值:").pack(side=tk.LEFT)  
        self.review_threshold_var = tk.DoubleVar(value=0.50)  
        review_threshold_scale = ttk.Scale(  
            review_threshold_frame, from_=0.0, to=0.94,  
            variable=self.review_threshold_var, orient=tk.HORIZONTAL  
        )  
        review_threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)  
        self.review_threshold_label = ttk.Label(review_threshold_frame, text="0.50")  
        self.review_threshold_label.pack(side=tk.LEFT)  
        review_threshold_scale.configure(command=self.update_review_threshold_label)  

        # 阈值说明  
        ttk.Label(  
            predict_section.content_frame,  
            text="📌 置信度 ≥ 自动归档阈值 → 自动归档\n"  
                 "   自动归档阈值 > 置信度 ≥ 人工审核阈值 → 加入审核\n"  
                 "   置信度 < 人工审核阈值 → 直接丢弃",  
            font=("Arial", 8), foreground="gray", justify=tk.LEFT  
        ).pack(pady=5, padx=5) 
        # ========== 人工审核区 ==========
        review_section = CollapsibleFrame(control_frame, text="人工审核队列")
        review_section.pack(fill=tk.X, pady=5, padx=5)

        count_frame = ttk.Frame(review_section.content_frame)
        count_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(count_frame, text="待审核图像:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.review_count_label = ttk.Label(count_frame, text="0", font=("Arial", 14, "bold"), foreground="red")
        self.review_count_label.pack(side=tk.LEFT)

        nav_frame = ttk.Frame(review_section.content_frame)
        nav_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Button(nav_frame, text="◀ 上一张", command=self.prev_review, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="下一张 ▶", command=self.next_review, width=12).pack(side=tk.LEFT, padx=2)

        self.review_index_label = ttk.Label(review_section.content_frame, text="", font=("Arial", 9), foreground="gray")
        self.review_index_label.pack(pady=5, padx=5)

        self.review_prediction_label = ttk.Label(review_section.content_frame, text="", font=("Arial", 9), foreground="blue", wraplength=280)
        self.review_prediction_label.pack(pady=5, padx=5)

        ttk.Separator(review_section.content_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=5)

        correct_frame = ttk.Frame(review_section.content_frame)
        correct_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(correct_frame, text="修正为:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(correct_frame, textvariable=self.class_var, state="readonly", width=12)
        self.class_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.class_combo.bind('<<ComboboxSelected>>', self._on_class_selected)

        action_frame = ttk.Frame(review_section.content_frame)
        action_frame.pack(fill=tk.X, pady=10, padx=5)

        self.confirm_button = ttk.Button(action_frame, text="✓ 确认归档", command=self.confirm_and_archive, width=26, state='disabled')
        self.confirm_button.pack(pady=3)

        self.new_class_button = ttk.Button(action_frame, text="➕ 新建类型", command=self.create_new_class, width=26)
        self.new_class_button.pack(pady=3)

        self.delete_button = ttk.Button(action_frame, text="✗ 删除图像", command=self.delete_image, width=26, state='disabled')
        self.delete_button.pack(pady=3)

        ttk.Separator(review_section.content_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=5)
        # ✅ 新增：导出Excel按钮（位于"清除所有待审核图像"按钮上方）
        ttk.Button(
            review_section.content_frame,
            text="📊 批量输出 Excel",
            command=self.export_to_excel,
            width=26
        ).pack(pady=5, padx=5)

        self.clear_review_button = ttk.Button(review_section.content_frame, text="🗑 清除所有待审核图像", command=self.clear_all_review_images, width=26, style='Danger.TButton')
        self.clear_review_button.pack(pady=5, padx=5)

        ttk.Label(review_section.content_frame, text="快捷键: ← 上一张 | → 下一张\nEnter 确认 | Delete 删除", font=("Arial", 8), foreground="gray").pack(pady=5, padx=5)

        self.update_statistics()

    def _on_class_selected(self, event):
        """类别选择变化时的回调"""
        if self.class_var.get():
            self.confirm_button.config(state='normal')
            self.log_message(f"修正类别选择: {self.class_var.get()}", "INFO")
        else:
            self.confirm_button.config(state='disabled')

    def toggle_auto_batch(self):
        """切换自动批次大小模式"""
        if self.auto_batch_var.get():
            self.batch_combo.config(state='disabled')
            model_type = self.model_var.get()
            img_size = PhytoplanktonClassifier.MODEL_CONFIGS[model_type]['default_size']
            optimal_batch = get_optimal_batch_size(img_size)
            self.batch_var.set(optimal_batch)
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.auto_batch_label.config(text=f"(自动: {optimal_batch}, GPU: {gpu_memory:.1f}GB)")
            else:
                self.auto_batch_label.config(text=f"(自动: {optimal_batch}, CPU模式)")
            
            self.update_status(f"自动批次大小: {optimal_batch}")
        else:
            self.batch_combo.config(state='readonly')
            self.auto_batch_label.config(text="")
  
    def _on_model_selected(self, event):
        """模型选择变化时更新信息"""
        model_type = self.model_var.get()
        config = PhytoplanktonClassifier.MODEL_CONFIGS[model_type]
        info_text = f"{config['description']}\n参数量: {config['params']}/输入尺寸: {config['default_size']}×{config['default_size']}"
        self.model_info_label.config(text=info_text)
        if hasattr(self, 'auto_batch_var') and self.auto_batch_var.get():
            self.toggle_auto_batch()

    def _create_display_panel_LEGACY(self, parent):
        """已废弃，保留仅供参考"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)
        display_frame.rowconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        image_section = ttk.LabelFrame(display_frame, text="图像预览", padding="10")
        image_section.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        image_section.columnconfigure(0, weight=1)
        image_section.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(image_section)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        logger_section = ttk.LabelFrame(display_frame, text="操作日志", padding="10")
        logger_section.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        logger_section.columnconfigure(0, weight=1)
        logger_section.rowconfigure(0, weight=1)
        
        self.logger_text = tk.Text(logger_section, height=20, width=40, wrap=tk.WORD, font=("Consolas", 9))
        self.logger_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        logger_scrollbar = ttk.Scrollbar(logger_section, orient=tk.VERTICAL, command=self.logger_text.yview)
        logger_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.logger_text.configure(yscrollcommand=logger_scrollbar.set)
        
        self.logger_text.tag_configure("INFO", foreground="blue")
        self.logger_text.tag_configure("SUCCESS", foreground="green")
        self.logger_text.tag_configure("WARNING", foreground="orange")
        self.logger_text.tag_configure("ERROR", foreground="red")
        self.logger_text.tag_configure("HEADER", foreground="purple", font=("Consolas", 9, "bold"))
        
        stats_section = ttk.LabelFrame(display_frame, text="数据统计", padding="10")
        stats_section.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        stats_section.columnconfigure(0, weight=1)
        stats_section.rowconfigure(0, weight=1)
        
        self.stats_text = tk.Text(stats_section, height=20, width=35, font=("Arial", 10), wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text.tag_configure("title", font=("Arial", 11, "bold"), foreground="navy")
        self.stats_text.tag_configure("value", font=("Arial", 10, "bold"), foreground="darkgreen")
        
        result_section = ttk.LabelFrame(display_frame, text="预测结果", padding="10")
        result_section.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        result_section.columnconfigure(0, weight=1)
        result_section.rowconfigure(0, weight=1)
        
        self.result_tree = ttk.Treeview(result_section, columns=("Class", "Confidence", "Status"), show="headings", height=8)
        self.result_tree.heading("Class", text="类别")
        self.result_tree.heading("Confidence", text="置信度")
        self.result_tree.heading("Status", text="状态")
        self.result_tree.column("Class", width=200)
        self.result_tree.column("Confidence", width=150)
        self.result_tree.column("Status", width=150)
        
        scrollbar = ttk.Scrollbar(result_section, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscroll=scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def log_message(self, message, level="INFO"):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        if hasattr(self, 'logger_text'):
            self.logger_text.insert(tk.END, log_entry, level)
            self.logger_text.see(tk.END)
            self.root.update_idletasks()
        if hasattr(self, 'run_manager') and self.run_manager:
            self.run_manager.log(message, level)

    def update_threshold_label(self, value):  
        """更新自动归档阈值，并联动限制人工审核阈值上限"""  
        val = round(float(value), 2)  
        self.confidence_threshold = val  
        self.threshold_label.config(text=f"{val:.2f}")  
        # 同步更新审核阈值滑块的最大值上限  
        if hasattr(self, 'review_threshold_var'):  
            new_upper = round(val - 0.01, 2)  
            # 动态更新滑块上限  
            self.review_threshold_scale.configure(to=new_upper)  
            # ✅ 若当前审核阈值已超过新上限，则自动压低  
            if self.review_threshold_var.get() >= val:  
                clamped = round(val - 0.05, 2)  
                clamped = max(0.0, clamped)   # 不能低于0  
                self.review_threshold_var.set(clamped)  
                self.review_threshold_label.config(text=f"{clamped:.2f}")  
                self.review_threshold = clamped  
                self.log_message(  
                    f"⚠ 人工审核阈值已自动调整为 {clamped:.2f}（不能高于归档阈值 {val:.2f}）",  
                    "WARNING"  
                )  

    def update_review_threshold_label(self, value):  
        """更新人工审核阈值，强制不超过自动归档阈值"""  
        val = round(float(value), 2)  

        # 硬性拦截，确保审核阈值 < 归档阈值  
        if val >= self.confidence_threshold:  
            val = round(self.confidence_threshold - 0.01, 2)  
            val = max(0.0, val)  
            self.review_threshold_var.set(val)  

        self.review_threshold = val  
        self.review_threshold_label.config(text=f"{val:.2f}") 

    def update_status(self, message):
        self.status_var.set(f"{datetime.now().strftime('%H:%M:%S')} - {message} | 设备: {device}")
        self.root.update_idletasks()
        self.log_message(message, "INFO")
    
    def update_statistics(self):
        """更新统计信息（Tab① 统计卡片 + 类别文本区）"""
        if not hasattr(self, 'stats_text'):
            return
        stats = self.data_manager.metadata.get("statistics", {})

        # ── Tab① 数值卡片 ─────────────────────────────────────────
        total     = stats.get('total_samples', 0)
        confirmed = stats.get('confirmed', stats.get('human_reviewed', 0))
        pseudo    = stats.get('pseudo', stats.get('auto_archived', 0))
        review    = len(self.review_queue)
        n_classes = len(self.class_names)

        for attr, val in [
                ('_stat_total',     str(total)),
                ('_stat_confirmed', str(confirmed)),
                ('_stat_pseudo',    str(pseudo)),
                ('_stat_review',    str(review)),
                ('_stat_classes',   str(n_classes)),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).config(text=val)

        # ── Tab① 类别列表文本区 ───────────────────────────────────
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)

        self.stats_text.insert(tk.END, '样本统计\n', 'title')
        for lbl, v in [('总样本', total), ('已确认', confirmed),
                       ('伪标签', pseudo), ('待审核', review)]:
            self.stats_text.insert(tk.END, f'  {lbl}: ', 'title')
            self.stats_text.insert(tk.END, f'{v}\n', 'value')

        self.stats_text.insert(tk.END, '\n类别列表\n', 'title')
        self.stats_text.insert(tk.END, f'  共 {n_classes} 个类别\n', 'value')
        for cls in self.class_names:
            self.stats_text.insert(tk.END, f'  • {cls}\n')

        self.stats_text.insert(tk.END, '\n模型信息\n', 'title')
        if self.classifier:
            self.stats_text.insert(tk.END,
                f'  {self.classifier.model_type}  |  {self.classifier.num_classes} 类\n', 'value')
        else:
            self.stats_text.insert(tk.END, '  未加载\n')

        self.stats_text.insert(tk.END, '\n系统信息\n', 'title')
        self.stats_text.insert(tk.END, f'  设备: {device}\n', 'value')
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gmem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.stats_text.insert(tk.END, f'  GPU: {gpu_name}  {gmem:.1f} GB\n', 'value')

        self.stats_text.config(state='disabled')

        # ── Tab④ 预测统计卡片（批量预测结束后由 route_stats 更新）─
        # (更新由 _update_pred_stats() 单独调用)

        # ── header 模型标签 ───────────────────────────────────────
        if hasattr(self, '_header_model_lbl'):
            self._header_update_model()

    def _update_pred_stats(self, total, auto, review, discard):
        """Update Tab④ stat cards after a batch predict run."""
        for attr, val in [
                ('_pred_stat_total',   str(total)),
                ('_pred_stat_auto',    str(auto)),
                ('_pred_stat_review',  str(review)),
                ('_pred_stat_discard', str(discard)),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).config(text=str(val))

    def initial_training(self):
        """初始训练"""  
        try:
            if not hasattr(self, 'train_dir_path'):
                messagebox.showwarning("警告", "请先选择数据集目录!")
                return
            
            self.log_message("="*50, "HEADER")  
            self.log_message("开始初始训练", "HEADER")  
            self.log_message("="*50, "HEADER")
            
            self.run_manager = RunManager() 
            self._disable_training_buttons()  
            
            model_type = self.model_var.get()  
            epochs = self.epochs_var.get()  
            learning_rate = float(self.lr_var.get())  
            batch_size = self.batch_var.get()  
            optimizer_type = self.optimizer_var.get()  
            weight_decay = float(self.weight_decay_var.get())  
            scheduler_type = self.scheduler_var.get()  
            patience = self.patience_var.get()  
            augmentation = self.augmentation_var.get()  
            
            self.update_status("准备数据集...")
            self.run_manager.log("开始准备数据集...") 
            
            config_dict = {  
                "训练类型": "初始训练",  
                "模型设置": {  
                    "模型类型": model_type,  
                    "模型名称": PhytoplanktonClassifier.MODEL_CONFIGS[model_type]['model_name'],  
                    "输入尺寸": PhytoplanktonClassifier.MODEL_CONFIGS[model_type]['default_size'],  
                    "参数量": PhytoplanktonClassifier.MODEL_CONFIGS[model_type]['params'],  
                    "类别数": len(self.class_names),  
                    "类别列表": self.class_names  
                },  
                "超参数设置": {  
                    "训练轮数": epochs,
                    "学习率": learning_rate,  
                    "批次大小": batch_size,  
                    "优化器": optimizer_type,  
                    "权重衰减": weight_decay,  
                    "学习率调度器": scheduler_type,  
                    "早停耐心": patience,  
                    "数据增强级别": augmentation  
                },  
                "数据集信息": {  
                    "数据集路径": str(self.train_dir_path),  
                    "训练比例": 0.7,  
                    "验证比例": 0.15,  
                    "测试比例": 0.15  
                }  
            }  
            self.run_manager.save_config(config_dict)
            
            config_msg = (  
                f"训练配置:\n"  
                f"模型: {model_type}\n"  
                f"Epochs: {epochs}\n"  
                f"学习率: {learning_rate}\n"  
                f"批次大小: {batch_size}\n"  
                f"优化器: {optimizer_type}\n"  
                f"权重衰减: {weight_decay}\n"  
                f"调度器: {scheduler_type}\n"  
                f"早停耐心: {patience}\n"  
                f"数据增强: {augmentation}"  
            )  
            
            if not messagebox.askokcancel("确认训练配置", config_msg):  
                self._enable_training_buttons()
                return  
            
            self.log_message(f"模型类型: {model_type}", "INFO")  
            self.log_message(f"训练轮数: {epochs}", "INFO")  
            self.log_message(f"学习率: {learning_rate}", "INFO")  
            self.log_message(f"批次大小: {batch_size}", "INFO")  
            self.update_status(f"开始训练... Run: {self.run_manager.run_dir.name}")  
            
            progress_window = TrainingProgressWindow(self.root, epochs, self.log_message) 
            
            style_folder = getattr(self, 'style_dir_path', None)
            training_thread = threading.Thread(
                target=self._training_worker,
                args=(epochs, model_type, learning_rate, batch_size,
                      optimizer_type, weight_decay, scheduler_type,
                      patience, augmentation, progress_window, style_folder),
                daemon=False
            )
            training_thread.start()
            
            print(f"\n{'='*60}")  
            print(f"运行目录: {self.run_manager.run_dir}")  
            print(f"日志文件: {self.run_manager.log_file}")  
            print(f"配置文件: {self.run_manager.config_file}")  
            print(f"{'='*60}\n")
            
        except Exception as e:  
            import traceback  
            traceback.print_exc()  
            messagebox.showerror("错误", f"训练启动失败:\n{str(e)}")  
            self.update_status(f"训练失败: {str(e)}")
            self._enable_training_buttons()

    def _training_worker(self, epochs, model_type, learning_rate, batch_size,
                         optimizer_type, weight_decay, scheduler_type,
                         patience, augmentation, progress_window,
                         style_folder=None):
        """在后台线程中执行训练"""
        try:
            print(f"[DEBUG] 训练线程已启动")

            self.classifier = PhytoplanktonClassifier(
                num_classes=len(self.class_names),
                model_type=model_type,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                weight_decay=weight_decay,
                scheduler_type=scheduler_type,
                augmentation_level=augmentation,
                style_folder=style_folder,
            )
            
            self.train_loader, self.val_loader, self.test_loader = \
                self.classifier.prepare_dataloaders(  
                    self.train_dir_path,  
                    batch_size=batch_size,  
                    train_ratio=0.7,  
                    val_ratio=0.15  
                )
            
            self.classifier.model.unfreeze_backbone()
            self.run_manager.log(f"数据集加载完成 - 训练: {len(self.train_loader.dataset)}, "  
                    f"验证: {len(self.val_loader.dataset)}, "  
                    f"测试: {len(self.test_loader.dataset)}")
            
            self._verify_model_parameters()  
            history = self.classifier.train(  
                self.train_loader,  
                self.val_loader,  
                epochs=epochs,  
                early_stop_patience=patience,  
                callback=progress_window.update,  
                run_manager=self.run_manager  
            )  
            
            self.root.after(0, self._training_completed, history, model_type)  
            
        except Exception as e:
            print(f"[ERROR] 训练线程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=str(e): messagebox.showerror("错误", f"训练失败:\n{e}"))

    def _verify_model_parameters(self):
        """验证模型参数梯度设置"""
        print("\n[DEBUG] 模型参数梯度检查:")
        trainable_count = 0
        frozen_count = 0
        for name, param in self.classifier.model.named_parameters():
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1
        print(f"总计: {trainable_count} 个可训练参数, {frozen_count} 个冻结参数")
        return trainable_count > 0

    def _training_completed(self, history, model_type):  
        """训练完成后的回调"""  
        try:  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
            model_filename = f"{model_type.replace(' ', '_')}_{timestamp}_acc{max(history['val_acc']):.2f}.pth"  
            model_path = self.run_manager.checkpoints_dir / model_filename  
            
            self.classifier.save_model(str(model_path))
            self._enable_training_buttons()
            self.update_statistics()   # refresh header model label
            messagebox.showinfo("成功",  
                f"初始训练完成!\n"  
                f"模型: {model_type}\n"  
                f"最佳验证准确率: {max(history['val_acc']):.2f}%\n\n"  
                f"模型已保存到: {model_path}")
        except Exception as e:  
            self._training_failed(str(e))  

    def _training_failed(self, error_msg):  
        messagebox.showerror("错误", f"训练失败:\n{error_msg}")  
        self.update_status(f"训练失败: {error_msg}")  
        self._enable_training_buttons() 

    def _disable_training_buttons(self):
        buttons_to_disable = ['initial_training_btn', 'fine_tune_btn', 'load_model_btn', 'save_model_btn', 'select_image_btn', 'batch_predict_btn']
        if not hasattr(self, '_training_buttons'):
            self._training_buttons = {}
        for btn_name in buttons_to_disable:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                self._training_buttons[btn_name] = btn
                btn.config(state='disabled')
        self._disable_hyperparam_controls()
        self.update_status("训练进行中，相关功能已禁用...")

    def _enable_training_buttons(self):
        if hasattr(self, '_training_buttons'):
            for btn_name, btn in self._training_buttons.items():
                btn.config(state='normal')
        self._enable_hyperparam_controls()
        self.update_status("训练完成，相关功能已恢复")

    def _disable_hyperparam_controls(self):
        pass

    def _enable_hyperparam_controls(self):
        pass

    def fine_tune_model(self):
        """微调模型"""
        if self.classifier is None or self.classifier.model is None:
            messagebox.showwarning("警告", "请先进行初始训练或加载模型!")
            return
        if not hasattr(self, 'train_dir_path'):
            messagebox.showwarning("警告", "请先选择数据集目录!")
            return
        
        try:
            epochs = max(5, self.epochs_var.get() // 2)
            batch_size = self.batch_var.get()
            learning_rate = float(self.lr_var.get()) / 10
            optimizer_type = self.optimizer_var.get()
            weight_decay = float(self.weight_decay_var.get())
            scheduler_type = self.scheduler_var.get()
            
            self.update_status("准备微调数据...")
            
            fine_tune_thread = threading.Thread(  
                target=self._fine_tune_worker,  
                args=(epochs, batch_size, learning_rate,   
                    optimizer_type, weight_decay, scheduler_type),  
                daemon=False  
            )  
            fine_tune_thread.start()  
            
        except Exception as e:  
            self._enable_training_buttons()  
            messagebox.showerror("错误", f"微调失败:\n{str(e)}")  

    def _fine_tune_worker(self, epochs, batch_size, learning_rate,  
                        optimizer_type, weight_decay, scheduler_type):  
        try:  
            self.train_loader, self.val_loader, self.test_loader = \
                self.classifier.prepare_dataloaders(  
                    self.train_dir_path,  
                    batch_size=batch_size,  
                    train_ratio=0.7,  
                    val_ratio=0.15  
                )  
            
            self.classifier.model.unfreeze_backbone()  
            
            if optimizer_type == 'Adam':
                self.classifier.optimizer = optim.Adam(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'AdamW':
                self.classifier.optimizer = optim.AdamW(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'SGD':
                self.classifier.optimizer = optim.SGD(self.classifier.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            elif optimizer_type == 'RMSprop':
                self.classifier.optimizer = optim.RMSprop(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            if scheduler_type == 'ReduceLROnPlateau':
                self.classifier.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.classifier.optimizer, mode='min', factor=0.5, patience=2)
            elif scheduler_type == 'CosineAnnealing':
                self.classifier.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.classifier.optimizer, T_max=epochs, eta_min=1e-7)
            elif scheduler_type == 'StepLR':
                self.classifier.scheduler = optim.lr_scheduler.StepLR(self.classifier.optimizer, step_size=3, gamma=0.5)
            
            progress_window = TrainingProgressWindow(self.root, epochs, self.log_message)  
            history = self.classifier.train(  
                self.train_loader,  
                self.val_loader,  
                epochs=epochs,  
                early_stop_patience=3,  
                callback=progress_window.update  
            )  
            self.root.after(0, self._fine_tune_completed, history)  
            
        except Exception as e:  
            self.root.after(0, lambda: self._training_failed(str(e)))  

    def _fine_tune_completed(self, history):  
        self.update_status("微调完成!")  
        self.update_statistics()  
        self._enable_training_buttons()  
        messagebox.showinfo("成功", f"模型微调完成!\n最佳验证准确率: {max(history['val_acc']):.2f}%")  

    def save_model(self):
        """保存模型"""
        if self.classifier is None or self.classifier.model is None:
            messagebox.showwarning("警告", "没有可保存的模型!")
            return
        
        if self.run_manager:
            initial_dir = str(self.run_manager.checkpoints_dir)
            default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        else:
            initial_dir = "./"
            default_name = "model.pth"
        
        file_path = filedialog.asksaveasfilename(
            initialdir=initial_dir, initialfile=default_name,
            defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.classifier.save_model(file_path)
                if self.run_manager: self.run_manager.log(f"模型手动保存: {file_path}")
                self.update_status(f"模型已保存: {file_path}")
                messagebox.showinfo("成功", f"模型保存成功!\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=device)
                num_classes = checkpoint['num_classes']
                model_type = checkpoint.get('model_type', 'B2')
                self.classifier = PhytoplanktonClassifier(num_classes=num_classes, model_type=model_type)
                self.classifier.load_model(file_path)
                self.class_names = self.classifier.class_names
                self.class_combo['values'] = self.class_names
                self.model_var.set(self.classifier.model_type)
                self.update_status(f"模型已加载: {file_path}")
                self.update_statistics()
                messagebox.showinfo("成功", f"模型加载成功!\n模型: {model_type}\n类别数: {len(self.class_names)}")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("错误", f"加载失败: {str(e)}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")])
        if file_path:
            self.current_image_path = file_path
            self.predict_single_image(file_path)
    
    def predict_single_image(self, image_path):
        """预测单张图像"""
        if self.classifier is None or self.classifier.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型!")
            return
        
        try:
            self.display_image(image_path)
            self.update_status("正在预测...")
            predictions = self.classifier.predict(image_path)
            self.current_predictions = predictions
            self.display_predictions(predictions)
            
            max_confidence = np.max(predictions)
            predicted_class = self.class_names[np.argmax(predictions)]
            
             # 读取当前人工审核阈值  
            review_thr = self.review_threshold if hasattr(self, 'review_threshold') else 0.0 

            if max_confidence >= self.confidence_threshold:
                self.data_manager.add_to_training(image_path, predicted_class, source="auto")
                self.update_status(f"高置信度 ({max_confidence:.3f})，已自动归档到 {predicted_class}")
                self.update_statistics()
            elif max_confidence >= review_thr:  
                # 中置信度 → 加入人工审核  
                self.data_manager.add_to_review_queue(image_path, predicted_class, max_confidence, predictions)  
                self.update_status(  
                    f"置信度 ({max_confidence:.3f}) 在审核区间 [{review_thr:.2f}, {self.confidence_threshold:.2f})，已加入审核队列"  
                )  
                self._load_review_queue()  
            else:  
                # 低置信度 → 直接丢弃  
                self.update_status(  
                    f"低置信度 ({max_confidence:.3f}) < 审核阈值 ({review_thr:.2f})，已丢弃"  
                )  
                self.log_message(  
                    f"🗑 丢弃: {Path(image_path).name}（置信度 {max_confidence:.3f} < {review_thr:.2f}）",  
                    "WARNING"  
                )  
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"预测失败: {str(e)}")
            self.update_status(f"预测失败: {str(e)}")

    def batch_predict(self):
        """批量预测"""
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if not folder_path: return
        if self.classifier is None or self.classifier.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型!")
            return
        # 提前警告用户（不阻止，因为add_to_review_queue已能处理）  
        if self.data_manager.review_dir:  
            pred_resolved = Path(folder_path).resolve()  
            review_resolved = self.data_manager.review_dir.resolve()  
            if pred_resolved == review_resolved:  
                self.log_message(  
                    "⚠ 预测目录与待审核目录相同，文件将原地记录（不复制）",  
                    "WARNING"  
                )  
        try:
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png','*.bmp','*.tif','*.tiff']:
                image_files.extend(Path(folder_path).glob(ext))
            if not image_files:
                messagebox.showwarning("警告", "文件夹中没有图像文件!")
                return
            
            # 自动清空：待审核队列（文件 + JSON元数据）  
            cleared_review = 0  
            if self.data_manager.review_dir and self.data_manager.review_dir.exists():  
                try:  
                    cleared_review = self.data_manager.clear_review_queue()  
                    self.log_message(  
                        f"🗑 已自动清空待审核队列（共 {cleared_review} 个文件）",  
                        "WARNING"  
                    )  
                except Exception as ce:  
                    self.log_message(f"清空待审核队列失败: {str(ce)}", "ERROR")  

            # 自动清空：上次批量预测记录 & 人工审核记录  
            self.prediction_records = []  
            self.review_records     = {}  
            self.log_message("🗑 已自动清空上次预测记录与审核记录", "WARNING")  

            # 同步刷新审核队列 UI  
            self.review_queue          = []  
            self.current_review_index  = 0  
            if hasattr(self, 'review_count_label'):  
                self.review_count_label.config(text="0")  
            self.image_label.configure(image='')  
            self.review_index_label.config(text="")  
            self.review_prediction_label.config(text="")  
            self.confirm_button.config(state='disabled')  
            self.delete_button.config(state='disabled')  
            if hasattr(self, '_review_started'):  
                delattr(self, '_review_started')  
            for item in self.result_tree.get_children():  
                self.result_tree.delete(item)  

            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量预测进度")
            progress_window.geometry("400x150")
            ttk.Label(progress_window, text="正在处理...").pack(pady=10)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(image_files), length=350)
            progress_bar.pack(pady=10)
            status_label = ttk.Label(progress_window, text="")
            status_label.pack(pady=10)
            
            # ---- 同步 SelfLearner 策略参数（来自 GUI 控件）----
            if hasattr(self, 'sl_strategy_var'):
                self.self_learner.strategy = self.sl_strategy_var.get()
            self.self_learner.auto_threshold = self.confidence_threshold
            self.self_learner.review_threshold = \
                self.review_threshold if hasattr(self, 'review_threshold') else 0.0
            if hasattr(self, 'sl_ood_var') and self.sl_ood_var.get():
                if self.self_learner.ood_detector is None:
                    self.self_learner.ood_detector = OODDetector(
                        method='energy', percentile=95.0)
            else:
                self.self_learner.ood_detector = None

            # ---- 批量推理 ----
            paths_str = [str(p) for p in image_files]
            all_probs, errors_list = [], []
            for i, image_path in enumerate(image_files):
                try:
                    all_probs.append(self.classifier.predict(str(image_path)))
                except Exception as e:
                    all_probs.append(None)
                    errors_list.append(image_path.name)
                    self.log_message(f"推理失败 [{image_path.name}]: {e}", "WARNING")
                progress_var.set(i + 1)
                try:
                    if progress_window.winfo_exists():
                        status_label.config(text=f"推理中: {i+1}/{len(image_files)}")
                        progress_window.update()
                except Exception:
                    pass

            valid_paths = [p for p, pr in zip(paths_str, all_probs) if pr is not None]
            valid_probs = np.stack([pr for pr in all_probs if pr is not None]) \
                if any(pr is not None for pr in all_probs) else np.empty((0, 0))

            def _gui_progress(done, total):
                progress_var.set(done)
                try:
                    if progress_window.winfo_exists():
                        status_label.config(text=f"路由中: {done}/{total}")
                        progress_window.update()
                except Exception:
                    pass

            # ---- SelfLearner 路由 ----
            route_stats = self.self_learner.route_batch(
                image_paths=valid_paths,
                probs_batch=valid_probs,
                logits_batch=None,   # energy OOD 暂用 log(probs) 近似
                class_names=self.class_names,
                data_manager=self.data_manager,
                progress_cb=_gui_progress,
                log_cb=self.log_message,
            )

            auto_archived = route_stats['auto']
            for_review    = route_stats['review'] + route_stats['ood']
            errors        = len(errors_list)
            self.prediction_records = route_stats['records']
            # 为失败条目补充记录
            for name in errors_list:
                self.prediction_records.append({
                    "文件名": name, "文件路径": name,
                    "预测类别": "处理失败", "置信度": "-",
                    "处理方式": "处理失败",
                    "处理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "_path_key": name,
                })

            try:
                if progress_window.winfo_exists():
                    progress_window.destroy()
            except Exception:
                pass
            self._load_review_queue()
            self.update_statistics()
            total_processed = auto_archived + for_review + route_stats['discard'] + errors
            self._update_pred_stats(total_processed, auto_archived, for_review,
                                    route_stats['discard'])
            result_msg = (f"批量预测完成!\n"
                          f"策略: {self.self_learner.strategy}\n"
                          f"伪标注归档: {auto_archived}\n"
                          f"待审核: {for_review}\n"
                          f"丢弃: {route_stats['discard']}")
            if errors > 0:
                result_msg += f"\n处理失败: {errors} 张"
            messagebox.showinfo("完成", result_msg)  
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"批量预测失败: {str(e)}")
            if self.run_manager: 
                self.run_manager.log(f"批量预测失败: {str(e)}", "ERROR")
            self.update_status(f"批量预测失败: {str(e)}")

    def display_image(self, image_path):
        img = Image.open(image_path)
        display_size = (500, 500)
        img.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def display_predictions(self, predictions):
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices[:5]:
            class_name = self.class_names[idx]
            confidence = predictions[idx]
            if confidence >= self.confidence_threshold:
                status = "✓ 自动归档"
                tag = "high"
            elif confidence >= 0.7:
                status = "⚠ 需审核"
                tag = "medium"
            else:
                status = "✗ 低置信度"
                tag = "low"
            self.result_tree.insert("", tk.END, values=(class_name, f"{confidence:.4f} ({confidence*100:.2f}%)", status), tags=(tag,))
        
        self.result_tree.tag_configure("high", background="#90EE90")
        self.result_tree.tag_configure("medium", background="#FFD700")
        self.result_tree.tag_configure("low", background="#FFB6C1")
    
    def _load_review_queue(self):
        if not hasattr(self, 'review_dir_path'):
            messagebox.showwarning("警告", "请先选择审核数据集目录!")
            return
        self.review_queue = self.data_manager.get_review_queue()  
        if hasattr(self, 'review_count_label'):  
            self.review_count_label.config(text=str(len(self.review_queue)))  
        self.current_review_index = 0  
    
    def prev_review(self):
        if not self.review_queue:
            messagebox.showinfo("提示", "审核队列为空!")
            return
        if self.current_review_index == 0 and not hasattr(self, '_review_started'):
            self._review_started = True
            self._show_review_image()
            return
        self.current_review_index = (self.current_review_index - 1) % len(self.review_queue)
        self._show_review_image()

    def next_review(self):
        if not self.review_queue:
            messagebox.showinfo("提示", "审核队列为空!")
            return
        if self.current_review_index == 0 and not hasattr(self, '_review_started'):
            self._review_started = True
            self._show_review_image()
            return
        self.current_review_index = (self.current_review_index + 1) % len(self.review_queue)
        self._show_review_image()
    
    def _show_review_image(self):
        if not self.review_queue:
            self.review_index_label.config(text="")
            self.review_prediction_label.config(text="")
            self.confirm_button.config(state='disabled')
            self.delete_button.config(state='disabled')
            # 修复：队列为空时清空表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            return
        
        image_path = self.review_queue[self.current_review_index]
        self.current_image_path = str(image_path)
        self.display_image(self.current_image_path)
        self.review_index_label.config(text=f"第 {self.current_review_index + 1} / {len(self.review_queue)} 张")
        
        info_file = image_path.with_suffix('.json')
        predicted_class = "未知"
        confidence = 0.0
        
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                predicted_class = info.get('predicted_class', '未知')
                confidence = info.get('confidence', 0.0)
                predictions = info.get('predictions', None)
                if predictions is not None:
                    self.display_predictions(predictions)
                else:
                    # 修复：如果没有详细预测数据，清空表格
                    for item in self.result_tree.get_children():
                        self.result_tree.delete(item)
            except Exception as e:
                self.log_message(f"读取预测信息失败: {str(e)}", "WARNING")
        
        self.review_prediction_label.config(text=f"AI预测: {predicted_class}\n置信度: {confidence:.2%}")
        self.class_var.set(predicted_class)
        self.confirm_button.config(state='normal')
        self.delete_button.config(state='normal')
        self.update_status(f"审核模式 | {self.current_review_index + 1}/{len(self.review_queue)} | 预测: {predicted_class} ({confidence:.2%})")

    def create_new_class(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("新建类型")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text="创建新的类别", font=("Arial", 12, "bold")).pack(pady=15)
        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        ttk.Label(input_frame, text="类别名称:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        class_name_var = tk.StringVar()
        class_name_entry = ttk.Entry(input_frame, textvariable=class_name_var, width=25, font=("Arial", 10))
        class_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        class_name_entry.focus()
        
        error_label = ttk.Label(dialog, text="", font=("Arial", 9), foreground="red")
        error_label.pack(pady=5)
        
        def validate_and_create():
            class_name = class_name_var.get().strip()
            if not class_name:
                error_label.config(text="⚠ 类别名称不能为空!")
                return
            if class_name in self.class_names:
                error_label.config(text=f"⚠ 类别 '{class_name}' 已存在!")
                return
            
            try:
                archive_class_dir = self.data_manager.archive_dir / class_name
                archive_class_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(self, 'train_dir_path') and self.train_dir_path:
                    train_class_dir = Path(self.train_dir_path) / class_name
                    train_class_dir.mkdir(parents=True, exist_ok=True)
                
                self.class_names.append(class_name)
                self.class_names = sorted(self.class_names)
                self.class_combo['values'] = self.class_names
                if hasattr(self, '_review_started') and self._review_started:
                    self.class_var.set(class_name)
                    self.confirm_button.config(state='normal')
                
                if class_name not in self.data_manager.metadata["classes"]:
                    self.data_manager.metadata["classes"].append(class_name)
                    self.data_manager.metadata["classes"] = sorted(self.data_manager.metadata["classes"])
                    self.data_manager._save_metadata()
                
                self.update_statistics()
                self.update_status(f"新类别已创建: {class_name}")
                dialog.destroy()
                messagebox.showinfo("成功", f"✓ 新类别创建成功!\n\n类别名称: {class_name}")
                
            except Exception as e:
                error_label.config(text=f"⚠ 创建失败: {str(e)}")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        ttk.Button(button_frame, text="✓ 创建", command=validate_and_create, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✗ 取消", command=dialog.destroy, width=12).pack(side=tk.LEFT, padx=5)
        class_name_entry.bind('<Return>', lambda e: validate_and_create())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    # def confirm_and_archive(self):
    #     if not self.current_image_path:
    #         messagebox.showwarning("警告", "请先选择要归档的图像!")
    #         return
    #     selected_class = self.class_var.get()
    #     if not selected_class:
    #         messagebox.showwarning("警告", "请选择类别!")
    #         return
        
    #     try:
    #         image_path = Path(self.current_image_path)
    #         self.data_manager.add_to_training(
    #             self.current_image_path, selected_class, source="manual"
    #         )

    #         # 记录人工审核结果，以原始路径为 key 覆写
    #         self.review_records[self.current_image_path] = {
    #             "最终类别": selected_class,
    #             "审核时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         }
            
    #         if image_path.parent == self.data_manager.review_dir:
    #             image_path.unlink()
    #             info_file = image_path.with_suffix('.json')
    #             if info_file.exists(): info_file.unlink()
            
    #         self._load_review_queue()
    #         if self.review_queue:
    #             if self.current_review_index >= len(self.review_queue):
    #                 self.current_review_index = len(self.review_queue) - 1
    #             self._show_review_image()
    #             self.update_status(f"已归档 | 剩余 {len(self.review_queue)} 张待审核")
    #         else:
    #             self.image_label.configure(image='')
    #             self.review_index_label.config(text="")
    #             self.review_prediction_label.config(text="")
    #             self.confirm_button.config(state='disabled')
    #             self.delete_button.config(state='disabled')
    #             if hasattr(self, '_review_started'): delattr(self, '_review_started')
    #             # 修复：队列清空时清空表格
    #             for item in self.result_tree.get_children():
    #                 self.result_tree.delete(item)
    #             self.update_status("审核队列已清空")
            
    #         self.update_statistics()
    #     except Exception as e:
    #         messagebox.showerror("错误", f"归档失败: {str(e)}")

    def confirm_and_archive(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择要归档的图像!")
            return
        selected_class = self.class_var.get()
        if not selected_class:
            messagebox.showwarning("警告", "请选择类别!")
            return
        
        try:
            image_path = Path(self.current_image_path)
            self.data_manager.add_to_training(
                self.current_image_path, selected_class, source="manual"
            )

            # ✅ 关键修复：读取 JSON 中的 original_path 作为 key
            record_key = self.current_image_path   # 默认 fallback
            info_file = image_path.with_suffix('.json')
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    original_path = info.get('original_path', None)
                    if original_path:
                        record_key = original_path  # ← 用原始路径覆写 key
                except Exception:
                    pass

            # 以原始预测路径为 key 写入 review_records
            self.review_records[record_key] = {
                "最终类别": selected_class,
                "审核时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.log_message(
                f"✅ 已记录人工审核: {Path(record_key).name} → {selected_class}",
                "SUCCESS"
            )

            if image_path.parent == self.data_manager.review_dir:
                image_path.unlink()
                if info_file.exists():
                    info_file.unlink()

            self._load_review_queue()
            if self.review_queue:
                if self.current_review_index >= len(self.review_queue):
                    self.current_review_index = len(self.review_queue) - 1
                self._show_review_image()
                self.update_status(f"已归档 | 剩余 {len(self.review_queue)} 张待审核")
            else:
                self.image_label.configure(image='')
                self.review_index_label.config(text="")
                self.review_prediction_label.config(text="")
                self.confirm_button.config(state='disabled')
                self.delete_button.config(state='disabled')
                if hasattr(self, '_review_started'):
                    delattr(self, '_review_started')
                for item in self.result_tree.get_children():
                    self.result_tree.delete(item)
                self.update_status("审核队列已清空")

            self.update_statistics()
        except Exception as e:
            messagebox.showerror("错误", f"归档失败: {str(e)}")


    def delete_image(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择要删除的图像!")
            return
        
        image_path = Path(self.current_image_path)
        try:
            if image_path.exists(): image_path.unlink()
            info_file = image_path.with_suffix('.json')
            if info_file.exists(): info_file.unlink()
            
            self._load_review_queue()
            if self.review_queue:
                if self.current_review_index >= len(self.review_queue):
                    self.current_review_index = len(self.review_queue) - 1
                self._show_review_image()
                self.update_status(f"已删除 | 剩余 {len(self.review_queue)} 张待审核")
            else:
                self.image_label.configure(image='')
                self.review_index_label.config(text="")
                self.review_prediction_label.config(text="")
                self.confirm_button.config(state='disabled')
                self.delete_button.config(state='disabled')
                if hasattr(self, '_review_started'): delattr(self, '_review_started')
                # 修复：队列清空时清空表格
                for item in self.result_tree.get_children():
                    self.result_tree.delete(item)
                self.update_status("审核队列已清空")
            
            self.update_statistics()
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {str(e)}")

    def hyperparameter_tuning(self):
        if not hasattr(self, 'train_dir_path'):
            messagebox.showwarning("警告", "请先选择数据集目录!")
            return
        
        if self.classifier is None:
            try:
                self.classifier = PhytoplanktonClassifier(
                    num_classes=len(self.class_names), model_type=self.model_var.get(),
                    learning_rate=float(self.lr_var.get()), optimizer_type=self.optimizer_var.get(),
                    weight_decay=float(self.weight_decay_var.get()), scheduler_type=self.scheduler_var.get(),
                    augmentation_level=self.augmentation_var.get()
                )
                self.train_loader, self.val_loader, self.test_loader = \
                    self.classifier.prepare_dataloaders(self.train_dir_path, batch_size=self.batch_var.get(), train_ratio=0.1, val_ratio=0.1)
                self.classifier.model.unfreeze_backbone()
            except Exception as e:
                messagebox.showerror("错误", f"初始化分类器失败:\n{str(e)}")
                return
        
        tuning_window = tk.Toplevel(self.root)
        tuning_window.title("超参数自动调优")
        tuning_window.geometry("600x500")
        ttk.Label(tuning_window, text="Grid Search 超参数优化", font=("Arial", 14, "bold")).pack(pady=10)
        
        config_frame = ttk.LabelFrame(tuning_window, text="搜索空间", padding="10")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        search_space_text = tk.Text(config_frame, height=15, width=70)
        search_space_text.pack(pady=10)
        
        search_space = {
            "learning_rate": ["1e-3", "5e-4", "1e-4"],
            "batch_size": [16, 32, 64],
            "optimizer": ["Adam", "AdamW"],
            "weight_decay": ["1e-4", "1e-5"]
        }
        
        import itertools
        total_combinations = len(list(itertools.product(*search_space.values())))
        search_info = f"搜索空间配置:\n\n{search_space}\n\n总计: {total_combinations} 种组合\n预计训练时间: ~{total_combinations * self.epochs_var.get() * 2} 分钟\n"
        search_space_text.insert(1.0, search_info)
        search_space_text.config(state='disabled')
        
        button_frame = ttk.Frame(tuning_window)
        button_frame.pack(pady=10)
        def start_tuning():
            tuning_window.destroy()
            self._run_grid_search(search_space)
        ttk.Button(button_frame, text="开始调优", command=start_tuning).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=tuning_window.destroy).pack(side=tk.LEFT, padx=5)

    def _run_grid_search(self, search_space):
        import itertools
        self.tuning_manager = RunManager()
        
        keys = search_space.keys()
        values = search_space.values()
        combinations = list(itertools.product(*values))
        
        results = []
        best_acc = 0
        best_params = None
        
        self._disable_training_buttons()
        result_window = tk.Toplevel(self.root)
        result_window.title("超参数调优进度")
        result_window.geometry("700x500")
        
        progress_label = ttk.Label(result_window, text="", font=("Arial", 12))
        progress_label.pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(result_window, variable=progress_var, maximum=len(combinations), length=650)
        progress_bar.pack(pady=5, padx=20)
        result_text = tk.Text(result_window, height=20, width=85)
        result_text.pack(padx=10, pady=10)
        
        for idx, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            progress_label.config(text=f"测试组合 {idx+1}/{len(combinations)}: {params}")
            progress_var.set(idx + 1)
            result_text.insert(tk.END, f"\n{'='*80}\n组合 {idx+1}/{len(combinations)}: {params}\n")
            result_text.see(tk.END)
            result_window.update()
            
            try:
                learning_rate = float(params['learning_rate'])
                batch_size = int(params['batch_size'])
                optimizer_type = params['optimizer']
                weight_decay = float(params['weight_decay'])
                
                if optimizer_type == 'Adam':
                    self.classifier.optimizer = optim.Adam(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                elif optimizer_type == 'AdamW':
                    self.classifier.optimizer = optim.AdamW(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                elif optimizer_type == 'SGD':
                    self.classifier.optimizer = optim.SGD(self.classifier.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
                elif optimizer_type == 'RMSprop':
                    self.classifier.optimizer = optim.RMSprop(self.classifier.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                
                if batch_size != self.batch_var.get():
                    self.train_loader, self.val_loader, self.test_loader = \
                        self.classifier.prepare_dataloaders(self.train_dir_path, batch_size=batch_size, train_ratio=0.1, val_ratio=0.1)
                
                history = self.classifier.train(self.train_loader, self.val_loader, epochs=3, early_stop_patience=2, callback=None)
                
                val_acc = max(history['val_acc'])
                results.append({'params': params, 'val_acc': val_acc})
                result_text.insert(tk.END, f"✓ 验证准确率: {val_acc:.2f}%\n")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = params
                    result_text.insert(tk.END, "★ 当前最佳!\n")
            except Exception as e:
                result_text.insert(tk.END, f"✗ 失败: {str(e)}\n")
            
            result_text.see(tk.END)
            result_window.update()
        
        final_results = {'results': results, 'best_params': best_params, 'best_acc': best_acc, 'timestamp': datetime.now().isoformat()}
        results_json_path = self.tuning_manager.run_dir / "grid_search_results.json"
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        self._enable_training_buttons()
        result_text.insert(tk.END, f"\n{'='*80}\n调优完成! 最佳准确率: {best_acc:.2f}%\n")
        
        if best_params and messagebox.askyesno("应用最佳参数", f"是否应用最佳参数?\n\n{best_params}\n\n准确率: {best_acc:.2f}%"):
            self.lr_var.set(str(best_params['learning_rate']))
            self.batch_var.set(best_params['batch_size'])
            self.optimizer_var.set(best_params['optimizer'])
            self.weight_decay_var.set(str(best_params['weight_decay']))
            messagebox.showinfo("成功", "最佳参数已应用!")

    def select_train_dir(self):
        dir_path = filedialog.askdirectory(title="选择训练数据集目录", initialdir=self.train_dir_path if hasattr(self, 'train_dir_path') else "./class/training_data")
        if dir_path:
            self.train_dir_path = dir_path
            self.train_dir_label.config(text=dir_path, foreground="blue")
            self.data_manager.set_train_directory(self.train_dir_path)

    def select_review_dir(self):
        default_review = str(Path("./class/review_queue").resolve())  
        initial = self.review_dir_path if self.review_dir_path else default_review 
        Path(initial).mkdir(parents=True, exist_ok=True) 
        dir_path = filedialog.askdirectory(title="选择待审核图像目录", initialdir=initial) #self.review_dir_path if hasattr(self, 'review_dir_path') else "./class/review_queue")
        if dir_path:
            self.review_dir_path = dir_path
            self.review_dir_label.config(text=dir_path, foreground="blue")
            self.data_manager.set_review_directory(self.review_dir_path)

    def select_archive_dir(self):
        dir_path = filedialog.askdirectory(title="选择归档图像目录", initialdir=self.archive_dir_path if hasattr(self, 'archive_dir_path') else "./class/archived_data")
        if dir_path:
            self.archive_dir_path = dir_path
            self.archive_dir_label.config(text=dir_path, foreground="blue")
            self.data_manager.set_archive_directory(self.archive_dir_path)

    def select_style_dir(self):
        """选择风格来源目录（目标域 FlowCAM/显微镜图像文件夹）。"""
        dir_path = filedialog.askdirectory(title="选择风格来源目录（目标域图像）")
        if dir_path:
            self.style_dir_path = dir_path
            self.style_dir_label.config(text=dir_path, foreground="blue")
            self.log_message(f"风格来源目录: {dir_path}", "INFO")

    # ------------------------------------------------------------------
    def open_unsupervised_annotator(self):
        """弹出无监督标注工具（cluster_review_gui）。

        标注完成后若指定了输出目录，自动刷新训练数据目录。
        """
        try:
            from cluster_review_gui import run_cluster_review
        except ImportError:
            messagebox.showerror("错误", "未找到 cluster_review_gui.py，请确认文件存在于同目录。")
            return

        default_out = str(Path(self.train_dir_path).parent / "unsup_labeled") \
            if hasattr(self, 'train_dir_path') else "./class/unsup_labeled"

        def _on_done(out_dir, stats):
            total = sum(stats.values())
            self.log_message(
                f"无监督标注完成: {total} 张图 → {out_dir}", "SUCCESS")
            if messagebox.askyesno(
                "导入训练目录",
                f"已导出 {total} 张标注图像到:\n{out_dir}\n\n是否将其设为训练数据目录？"
            ):
                self.train_dir_path = out_dir
                self.train_dir_label.config(text=out_dir, foreground="blue")
                self.data_manager.set_train_directory(out_dir)
                self.class_names = self.data_manager.get_class_names()
                self.class_combo['values'] = self.class_names
                self.log_message(
                    f"训练目录已更新，类别: {self.class_names}", "INFO")
                self.update_statistics()

        run_cluster_review(
            parent=self.root,
            output_dir=default_out,
            done_callback=_on_done,
        )

    def export_to_excel(self):
        """将批量预测与人工审核结果合并导出为 Excel（单 Sheet）"""
        if not self.prediction_records:
            messagebox.showwarning(
                "提示",
                "暂无可导出的记录！\n请先执行批量预测后再导出。"
            )
            return

        # ── 文件保存对话框 ──────────────────────────────────────
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"预测审核记录_{timestamp}.xlsx"
        file_path = filedialog.asksaveasfilename(
            title="保存 Excel 文件",
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "预测与审核记录"

            # ── 样式定义 ────────────────────────────────────────
            thin       = Side(style="thin", color="BFBFBF")
            border     = Border(left=thin, right=thin, top=thin, bottom=thin)

            # 表头：深蓝底白字
            hdr_font   = Font(name="微软雅黑", bold=True, color="FFFFFF", size=11)
            hdr_fill   = PatternFill("solid", fgColor="2F5597")
            hdr_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # 数据行填充
            fill_auto    = PatternFill("solid", fgColor="C6EFCE")   # 浅绿  — 自动归档
            fill_review  = PatternFill("solid", fgColor="FFEB9C")   # 浅黄  — 待审核（未被人工处理）
            fill_manual  = PatternFill("solid", fgColor="F4B942")   # 橙色  — 人工审核（已修正）
            fill_discard = PatternFill("solid", fgColor="FFC7CE")   # 浅红  — 低置信度丢弃
            fill_error   = PatternFill("solid", fgColor="D9D9D9")   # 灰色  — 处理失败

            center_a   = Alignment(horizontal="center", vertical="center")
            left_a     = Alignment(horizontal="left",   vertical="center")

            # ── 表头 ────────────────────────────────────────────
            headers    = ["序号", "文件名", "文件路径", "预测类别", "置信度", "处理方式", "处理时间", "备注"]
            col_widths = [6,      28,       55,         18,         10,       14,         22,         20   ]

            for col_idx, (h, w) in enumerate(zip(headers, col_widths), start=1):
                cell            = ws.cell(row=1, column=col_idx, value=h)
                cell.font       = hdr_font
                cell.fill       = hdr_fill
                cell.alignment  = hdr_align
                cell.border     = border
                ws.column_dimensions[
                    openpyxl.utils.get_column_letter(col_idx)
                ].width = w

            ws.row_dimensions[1].height = 22
            ws.freeze_panes = "A2"

            # ── 数据行 ──────────────────────────────────────────
            manual_count = 0

            for row_idx, record in enumerate(self.prediction_records, start=2):
                path_key     = record.get("_path_key", record.get("文件路径", ""))
                reviewed     = path_key in self.review_records

                if reviewed:
                    # 人工审核：覆写类别、置信度、处理方式
                    rev          = self.review_records[path_key]
                    display_cls  = rev["最终类别"]
                    display_conf = "-"
                    display_type = "人工审核"
                    display_time = rev["审核时间"]
                    # 备注：标明原始预测
                    remark       = f"原预测: {record['预测类别']} ({record['置信度']:.4f})"
                    row_fill     = fill_manual
                    manual_count += 1
                else:
                    display_cls  = record["预测类别"]
                    display_conf = record["置信度"]
                    display_type = record["处理方式"]
                    display_time = record["处理时间"]
                    remark       = ""
                    # 按处理方式选色
                    fill_map = {
                        "自动归档":     fill_auto,
                        "待人工审核":   fill_review,
                        "低置信度丢弃": fill_discard,
                    }
                    row_fill = fill_map.get(display_type, fill_error)

                row_data = [
                    row_idx - 1,
                    record["文件名"],
                    record["文件路径"],
                    display_cls,
                    display_conf,
                    display_type,
                    display_time,
                    remark
                ]

                for col_idx, value in enumerate(row_data, start=1):
                    cell           = ws.cell(row=row_idx, column=col_idx, value=value)
                    cell.border    = border
                    cell.fill      = row_fill
                    cell.alignment = center_a if col_idx in (1, 4, 5, 6) else left_a

            # ── 汇总区 ──────────────────────────────────────────
            sum_row    = len(self.prediction_records) + 3
            sum_font   = Font(name="微软雅黑", bold=True, color="2F5597")

            auto_count    = sum(1 for r in self.prediction_records if r["处理方式"] == "自动归档")
            review_count  = sum(1 for r in self.prediction_records if r["处理方式"] == "待人工审核")
            discard_count = sum(1 for r in self.prediction_records if r["处理方式"] == "低置信度丢弃")
            error_count   = sum(1 for r in self.prediction_records if r["处理方式"] == "处理失败")

            summary_items = [
                ("总计图片数",       len(self.prediction_records)),
                ("自动归档",         auto_count),
                ("待审核（原始）",   review_count),
                ("低置信度丢弃",     discard_count),
                ("处理失败",         error_count),
                ("人工审核（已修正）", manual_count),
            ]
            for i, (label, value) in enumerate(summary_items):
                ws.cell(row=sum_row + i, column=1, value=label).font = sum_font
                ws.cell(row=sum_row + i, column=2, value=value).font = sum_font

            # ── 保存 ────────────────────────────────────────────
            wb.save(file_path)

            self.log_message(f"Excel 已导出: {file_path}", "SUCCESS")
            self.update_status(f"Excel 导出成功: {Path(file_path).name}")
            messagebox.showinfo(
                "导出成功",
                f"✅ Excel 文件已保存！\n\n"
                f"路径：{file_path}\n\n"
                f"📊 总记录数：{len(self.prediction_records)} 条\n"
                f"   ├ 🟢 自动归档：    {auto_count} 张\n"
                f"   ├ 🟡 待审核（原始）：{review_count} 张\n"
                f"   ├ 🟠 人工审核修正：  {manual_count} 张\n"
                f"   ├ 🔴 低置信度丢弃：{discard_count} 张\n"
                f"   └ ⚫ 处理失败：    {error_count} 张"
            )

        except ImportError:
            messagebox.showerror("错误", "请先安装 openpyxl：\npip install openpyxl")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"Excel 导出失败：\n{str(e)}")


    def clear_all_review_images(self):
        if not self.review_queue:
            messagebox.showinfo("提示", "待审核队列已经为空!")
            return
        if not messagebox.askyesno("确认清除", "确定要清除所有待审核图像吗?\n此操作不可恢复!", icon='warning'):
            return
        try:
            deleted_count = self.data_manager.clear_review_queue()
            self.review_queue = []
            self.current_review_index = 0
            self.review_count_label.config(text="0")
            self.image_label.configure(image='')
            self.review_index_label.config(text="")
            self.review_prediction_label.config(text="")
            self.confirm_button.config(state='disabled')
            self.delete_button.config(state='disabled')
            if hasattr(self, '_review_started'): delattr(self, '_review_started')
            # 修复：清空表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            self.update_statistics()
            messagebox.showinfo("完成", f"已删除 {deleted_count} 个文件")
        except Exception as e:
            messagebox.showerror("错误", f"清除失败: {str(e)}")
            
def main():
    """主函数"""
    data_dir = "./class"
    Path(data_dir).mkdir(exist_ok=True)
    root = tk.Tk()
    app = ActiveLearningGUI(root, data_dir)
    root.mainloop()

if __name__ == "__main__":
    main()