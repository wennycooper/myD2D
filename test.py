#!/usr/bin/env python3
"""
Test script for Hybrid Anomaly Detection Model
混合異常檢測模型的測試腳本
"""

import torch
import torch.nn.functional as F
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
import sys

# Add project path
sys.path.append('/home/kkuei/my-proj')

from models.hybrid_model import HybridAnomalyDetector
from data.dataset import Dataset
from utils.transforms import get_transform
from utils.metrics import image_level_metrics, pixel_level_metrics
from config.base_config import get_default_config


def setup_seed(seed=42):
    """
    設定隨機種子以確保實驗的可重複性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_fallback_args():
    """Create fallback arguments for testing without command line"""
    class Args:
        def __init__(self):
            self.data_path = '/path/to/test/data'  # User needs to set this
            self.dataset = 'mvtec'
            self.resolution = 512
            self.save_path = './results'
            self.metrics = 'image-pixel-level'
            self.sigma = 4
            self.checkpoint_path = None
            self.use_gpu = True
            
    return Args()


def test(args):
    """
    Main testing function
    """
    # Setup
    setup_seed()
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize model
    config = get_default_config()
    model = HybridAnomalyDetector(config).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = model.load_checkpoint(args.checkpoint_path)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("Warning: No checkpoint provided, using randomly initialized model")
    
    model.eval()
    
    # Setup data
    try:
        # Get transforms compatible with args interface
        preprocess, target_transform = get_transform(args)
        
        # Create dataset
        test_data = Dataset(
            root=args.data_path, 
            transform=preprocess, 
            target_transform=target_transform, 
            dataset_name=args.dataset,
            mode='test'
        )
        
        # Create dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False
        )
        
        obj_list = test_data.obj_list
        print(f"Testing on {len(test_data)} samples from {len(obj_list)} classes")
        
    except Exception as e:
        print(f"Error setting up data: {e}")
        print("This might be due to missing dataset. Please check the data_path.")
        return
    
    # Initialize results storage
    results = {}
    for obj in obj_list:
        results[obj] = {
            'gt_sp': [],         # image-level ground truth
            'pr_sp': [],         # image-level predictions
            'imgs_masks': [],    # pixel-level ground truth
            'anomaly_maps': []   # pixel-level predictions
        }
    
    # Testing loop
    print("Starting inference...")
    with torch.no_grad():
        for idx, items in enumerate(tqdm(test_dataloader, desc="Testing")):
            try:
                # Get data
                image = items['img'].to(device)
                cls_name = items['cls_name'][0]  # Extract string from list
                gt_mask = items['img_mask']
                anomaly_label = items['anomaly'].item()
                
                # Binarize ground truth mask
                gt_mask[gt_mask > 0.5] = 1
                gt_mask[gt_mask <= 0.5] = 0
                
                # Store ground truth
                results[cls_name]['imgs_masks'].append(gt_mask)
                results[cls_name]['gt_sp'].append(anomaly_label)
                
                # Model inference
                model_outputs = model(image, mode='eval')
                
                # Extract results
                anomaly_maps = model_outputs['anomaly_maps']  # (1, 64, 64)
                image_scores = model_outputs['image_scores']  # (1,)
                
                # Resize anomaly map to match ground truth resolution
                anomaly_map_resized = F.interpolate(
                    anomaly_maps.unsqueeze(1), 
                    size=(args.resolution, args.resolution), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
                
                # Apply Gaussian smoothing
                anomaly_map_smoothed = torch.stack([
                    torch.from_numpy(gaussian_filter(am.cpu().numpy(), sigma=args.sigma)) 
                    for am in anomaly_map_resized
                ], dim=0)
                
                # Store predictions
                results[cls_name]['pr_sp'].append(image_scores.cpu().item())
                results[cls_name]['anomaly_maps'].append(anomaly_map_smoothed)
                
                # Optional: Print progress for some samples
                if idx % 50 == 0:
                    print(f"Processed {idx+1} samples, Current: {cls_name}, Score: {image_scores.item():.4f}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Calculate metrics
    print("\nCalculating metrics...")
    table_ls = []
    image_auroc_list, image_ap_list = [], []
    pixel_auroc_list, pixel_aupro_list = [], []
    
    for obj in tqdm(obj_list, desc="Computing metrics"):
        try:
            table = [obj]
            
            # Skip if no data for this class
            if len(results[obj]['gt_sp']) == 0:
                print(f"Warning: No data for class {obj}")
                continue
            
            # Concatenate results
            if results[obj]['imgs_masks']:
                results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
            if results[obj]['anomaly_maps']:
                results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
            
            # Calculate metrics based on args.metrics
            if 'pixel' in args.metrics and results[obj]['anomaly_maps'] is not None:
                try:
                    pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                    pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                    pixel_auroc_list.append(pixel_auroc)
                    pixel_aupro_list.append(pixel_aupro)
                except Exception as e:
                    print(f"Error calculating pixel metrics for {obj}: {e}")
                    pixel_auroc, pixel_aupro = 0.0, 0.0
                    pixel_auroc_list.append(pixel_auroc)
                    pixel_aupro_list.append(pixel_aupro)
            
            if 'image' in args.metrics:
                try:
                    image_auroc = image_level_metrics(results, obj, "image-auroc")
                    image_ap = image_level_metrics(results, obj, "image-ap")
                    image_auroc_list.append(image_auroc)
                    image_ap_list.append(image_ap)
                except Exception as e:
                    print(f"Error calculating image metrics for {obj}: {e}")
                    image_auroc, image_ap = 0.0, 0.0
                    image_auroc_list.append(image_auroc)
                    image_ap_list.append(image_ap)
            
            # Build result table
            if args.metrics == 'image-pixel-level':
                table.extend([f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}", 
                            f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"])
            elif args.metrics == 'pixel-level':
                table.extend([f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}"])
            elif args.metrics == 'image-level':
                table.extend([f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"])
            
            table_ls.append(table)
            
        except Exception as e:
            print(f"Error processing metrics for {obj}: {e}")
            continue
    
    # Calculate mean metrics
    mean_row = ['mean']
    headers = ['objects']
    
    if args.metrics == 'image-pixel-level':
        if pixel_auroc_list and pixel_aupro_list and image_auroc_list and image_ap_list:
            mean_row.extend([f"{np.mean(pixel_auroc_list)*100:.1f}", f"{np.mean(pixel_aupro_list)*100:.1f}", 
                           f"{np.mean(image_auroc_list)*100:.1f}", f"{np.mean(image_ap_list)*100:.1f}"])
        headers.extend(['pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'])
    elif args.metrics == 'pixel-level':
        if pixel_auroc_list and pixel_aupro_list:
            mean_row.extend([f"{np.mean(pixel_auroc_list)*100:.1f}", f"{np.mean(pixel_aupro_list)*100:.1f}"])
        headers.extend(['pixel_auroc', 'pixel_aupro'])
    elif args.metrics == 'image-level':
        if image_auroc_list and image_ap_list:
            mean_row.extend([f"{np.mean(image_auroc_list)*100:.1f}", f"{np.mean(image_ap_list)*100:.1f}"])
        headers.extend(['image_auroc', 'image_ap'])
    
    # Add mean row to table
    table_ls.append(mean_row)
    
    # Print results
    print("\n" + "="*60)
    print("HYBRID ANOMALY DETECTION TEST RESULTS")
    print("="*60)
    print(tabulate(table_ls, headers=headers, tablefmt="grid"))
    
    # Save results
    results_file = os.path.join(args.save_path, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("HYBRID ANOMALY DETECTION TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(tabulate(table_ls, headers=headers, tablefmt="grid"))
        f.write(f"\n\nTest completed on {len(test_data)} samples\n")
        f.write(f"Model checkpoint: {args.checkpoint_path or 'None (random weights)'}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("Test completed successfully!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Test Hybrid Anomaly Detection Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--dataset', type=str, default='mvtec',
                       choices=['mvtec', 'visa', 'btad', 'mpdd'],
                       help='Dataset name')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--resolution', type=int, default=512,
                       help='Input image resolution')
    
    # Evaluation arguments
    parser.add_argument('--metrics', type=str, default='image-pixel-level',
                       choices=['image-level', 'pixel-level', 'image-pixel-level'],
                       help='Metrics to evaluate')
    parser.add_argument('--sigma', type=float, default=4.0,
                       help='Gaussian smoothing sigma for anomaly maps')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default='./test_results',
                       help='Path to save test results')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Run test
    test(args)


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Fallback for testing without command line
        print("Running test with fallback arguments...")
        print("For full functionality, run with command line arguments:")
        print("python test.py --data_path /path/to/dataset --checkpoint_path /path/to/checkpoint.pth")
        
        args = create_fallback_args()
        try:
            test(args)
        except Exception as e:
            print(f"Test failed: {e}")
            print("This is expected when running without proper dataset setup.")
            print("Please provide proper --data_path and --checkpoint_path arguments.")