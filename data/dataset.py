"""
Dataset utilities for anomaly detection
"""

import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    """
    載入指定路徑的圖像，進行裁切並縮放至 512x512 像素。

    Args:
        image_path (str or np.ndarray): 圖像檔案的路徑或已載入的 NumPy 陣列。
        left, right, top, bottom (int): 從圖像邊界向內裁切的像素值。

    Returns:
        np.ndarray: 處理完成的 512x512 圖像，格式為 NumPy 陣列。
    """
    if isinstance(image_path, str):
        # 如果是檔案路徑，則開啟圖像並轉換為 RGB
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        # 如果已經是 NumPy 陣列，則直接使用
        image = image_path
    
    h, w, c = image.shape
    # 確保裁切邊界不會超出圖像範圍
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - 1)
    bottom = min(bottom, h - top - 1)
    # 進行裁切
    image = image[top:h - bottom, left:w - right]
    
    h, w, c = image.shape
    # 為了保持圖像內容，將非正方形的圖像裁切成正方形
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
        
    # 將圖像縮放至 512x512
    image = np.array(Image.fromarray(image).resize((512, 512)))
    # 將 NumPy 圖像陣列轉換為 PyTorch 張量並標準化
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def generate_class_info(dataset_name):
    """
    Generate class information for different datasets
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (class_list, class_name_to_id_mapping)
    """
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'headct':
        obj_list = ['brain']
    else:
        # Default fallback for unknown datasets
        obj_list = ['default_class']
        
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id


class Dataset(data.Dataset):
    """
    Anomaly detection dataset loader
    """
    
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        
        # Load metadata
        meta_path = f'{self.root}/meta.json'
        if os.path.exists(meta_path):
            meta_info = json.load(open(meta_path, 'r'))
            name = self.root.split('/')[-1]
            meta_info = meta_info[mode]

            self.cls_names = list(meta_info.keys())
            for cls_name in self.cls_names:
                self.data_all.extend(meta_info[cls_name])
        else:
            # Fallback for datasets without meta.json
            print(f"Warning: meta.json not found at {meta_path}")
            self.cls_names = []
            self.data_all = []
            
        self.length = len(self.data_all)
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = (
            data['img_path'], data['mask_path'], data['cls_name'],
            data['specie_name'], data['anomaly']
        )
        
        # Load image
        img = Image.open(os.path.join(self.root, img_path))
        
        # Load mask
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            mask_full_path = os.path.join(self.root, mask_path)
            if os.path.isdir(mask_full_path):
                # For classification only, not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                if os.path.exists(mask_full_path):
                    img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                else:
                    # Fallback for missing masks
                    img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
                    
        # Apply transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        
        # Get class ID safely
        cls_id = self.class_name_map_class_id.get(cls_name, 0)  # Default to 0 if not found
        
        return {
            'img': img, 
            'img_mask': img_mask, 
            'cls_name': cls_name, 
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path), 
            "cls_id": cls_id
        }