# -*- coding: utf-8 -*-
"""
models_registry.py — 统一模型注册表与构建接口

将分类骨干网络的选型从训练台代码中解耦，提供可插拔的 SOTA 模型支持。
所有模型均通过 timm 构建，便于后续扩展自监督(DINOv2)与新架构。

@author: brigchen@xmu.edu.cn
"""
import timm


# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------
# 每个条目字段：
#   model_name        : timm 模型名
#   default_size      : 推荐输入分辨率（与该预训练权重的原生分辨率匹配）
#   params            : 参数量（人类可读）
#   description       : 中文说明
#   family            : 架构系列（用于分组/特殊处理）
#   dynamic_img_size  : 是否在构建时向 timm 传入 img_size / dynamic_img_size
#                       （仅 ViT/DINOv2 等支持动态分辨率的骨干需要）
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # ===== EfficientNetV2（保持向后兼容，键名不变）=====
    'B0': {
        'model_name': 'tf_efficientnetv2_b0',
        'default_size': 224, 'params': '~7M',
        'description': '最快速度，适合原型开发',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },
    'B1': {
        'model_name': 'tf_efficientnetv2_b1',
        'default_size': 240, 'params': '~8M',
        'description': '平衡性能和速度',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },
    'B2': {
        'model_name': 'tf_efficientnetv2_b2',
        'default_size': 260, 'params': '~10M',
        'description': '推荐用于生产环境',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },
    'B3': {
        'model_name': 'tf_efficientnetv2_b3',
        'default_size': 300, 'params': '~14M',
        'description': '更高精度需求',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },
    'S': {
        'model_name': 'tf_efficientnetv2_s',
        'default_size': 384, 'params': '~24M',
        'description': '科研级精度',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },
    'M': {
        'model_name': 'tf_efficientnetv2_m',
        'default_size': 480, 'params': '~54M',
        'description': '最高精度，需强大算力',
        'family': 'efficientnetv2', 'dynamic_img_size': False,
    },

    # ===== ConvNeXt V2（SOTA 卷积骨干，强长尾表现）=====
    'ConvNeXtV2-T': {
        'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        'default_size': 224, 'params': '~28M',
        'description': 'ConvNeXtV2 Tiny，精度/速度兼顾',
        'family': 'convnext', 'dynamic_img_size': False,
    },
    'ConvNeXtV2-B': {
        'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'default_size': 224, 'params': '~89M',
        'description': 'ConvNeXtV2 Base，高精度卷积骨干',
        'family': 'convnext', 'dynamic_img_size': False,
    },

    # ===== Vision Transformer =====
    'ViT-B/16': {
        'model_name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'default_size': 224, 'params': '~86M',
        'description': 'ViT-Base，强全局建模能力',
        'family': 'vit', 'dynamic_img_size': False,
    },

    # ===== Swin Transformer =====
    'Swin-T': {
        'model_name': 'swin_tiny_patch4_window7_224.ms_in1k',
        'default_size': 224, 'params': '~28M',
        'description': 'Swin Tiny，层次化窗口注意力',
        'family': 'swin', 'dynamic_img_size': False,
    },
    'Swin-B': {
        'model_name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
        'default_size': 224, 'params': '~88M',
        'description': 'Swin Base，高精度层次化注意力',
        'family': 'swin', 'dynamic_img_size': False,
    },

    # ===== MaxViT（卷积 + 注意力混合）=====
    'MaxViT-T': {
        'model_name': 'maxvit_tiny_tf_224.in1k',
        'default_size': 224, 'params': '~31M',
        'description': 'MaxViT Tiny，局部+全局混合注意力',
        'family': 'maxvit', 'dynamic_img_size': False,
    },

    # ===== DINOv2（自监督预训练骨干，适合显微嵌入/微调）=====
    'DINOv2-S': {
        'model_name': 'vit_small_patch14_dinov2.lvd142m',
        'default_size': 224, 'params': '~22M',
        'description': 'DINOv2 ViT-S 自监督骨干（动态分辨率，适合显微特征）',
        'family': 'dinov2', 'dynamic_img_size': True,
    },
    'DINOv2-B': {
        'model_name': 'vit_base_patch14_dinov2.lvd142m',
        'default_size': 224, 'params': '~86M',
        'description': 'DINOv2 ViT-B 自监督骨干（动态分辨率，高表达力）',
        'family': 'dinov2', 'dynamic_img_size': True,
    },
}


def list_models():
    """返回注册表中所有可选模型键（保持插入顺序，用于 GUI 下拉框）"""
    return list(MODEL_REGISTRY.keys())


def get_config(model_key):
    """根据模型键获取配置；未注册时回退到 B2。"""
    return MODEL_REGISTRY.get(model_key, MODEL_REGISTRY['B2'])


def build_model(model_name, num_classes, pretrained=True, img_size=None,
                dynamic_img_size=False):
    """统一的模型构建入口。

    Args:
        model_name: timm 模型名（如 'tf_efficientnetv2_b0'）。
        num_classes: 分类类别数。
        pretrained: 是否加载预训练权重。
        img_size: 仅对支持动态分辨率的骨干生效。
        dynamic_img_size: 为 True 时向 timm 传入 img_size/dynamic_img_size，
                          使 ViT/DINOv2 可在非原生分辨率下工作。
    """
    create_kwargs = dict(pretrained=pretrained, num_classes=num_classes)
    if dynamic_img_size and img_size:
        create_kwargs['img_size'] = img_size
        create_kwargs['dynamic_img_size'] = True

    try:
        return timm.create_model(model_name, **create_kwargs)
    except Exception:
        # 离线或权重不可用时，退化为随机初始化，保证训练台仍可运行
        if pretrained:
            create_kwargs['pretrained'] = False
            return timm.create_model(model_name, **create_kwargs)
        raise


def build_embedding_extractor(model_key='DINOv2-S', pretrained=True, device='cpu'):
    """构建用于无监督标注的嵌入提取器（去掉分类头，输出池化特征向量）。

    返回 (model, config)：model 已置于 eval 模式并搬到 device。
    通过 timm 的 num_classes=0 获得全局池化后的特征。

    Args:
        model_key: 注册表键（推荐自监督骨干 'DINOv2-S'/'DINOv2-B'）。
        pretrained: 是否加载预训练权重（无监督标注几乎总是 True）。
        device: 'cpu' 或 'cuda'。
    """
    cfg = get_config(model_key)
    create_kwargs = dict(pretrained=pretrained, num_classes=0)
    if cfg.get('dynamic_img_size', False):
        create_kwargs['img_size'] = cfg['default_size']
        create_kwargs['dynamic_img_size'] = True
    try:
        model = timm.create_model(cfg['model_name'], **create_kwargs)
    except Exception:
        if pretrained:
            create_kwargs['pretrained'] = False
            model = timm.create_model(cfg['model_name'], **create_kwargs)
        else:
            raise
    model.eval()
    model.to(device)
    return model, cfg


def _head_param_ids(timm_model):
    """获取分类头(head/classifier)参数的 id 集合，用于通用冻结。"""
    head = None
    if hasattr(timm_model, 'get_classifier'):
        try:
            head = timm_model.get_classifier()
        except Exception:
            head = None
    if head is None:
        return set()
    return {id(p) for p in head.parameters()}


def freeze_backbone(timm_model):
    """通用冻结：保留分类头可训练，冻结骨干（适配任意 timm 架构）。"""
    head_ids = _head_param_ids(timm_model)
    for p in timm_model.parameters():
        p.requires_grad = id(p) in head_ids


def unfreeze_backbone(timm_model):
    """解冻：所有参数可训练。"""
    for p in timm_model.parameters():
        p.requires_grad = True
