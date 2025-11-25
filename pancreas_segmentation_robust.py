import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import shutil
import nibabel as nib
import pydicom as dicomio  # pydicom 库

import torch
import torch.optim as optim

# 尝试导入辅助模块
try:
    from loss import TverskyLoss
    from net import UNet_2D, UNet_3D
    from volume_patch_composer import volume_composer, patch_creator
    from dataset import Pancreas_2D_dataset, Pancreas_3D_dataset, partitioning
    from metrics import performance_metrics
    from train import train_2D, train_3D
    from inference import (get_inference_performance_metrics_3D)
except ImportError as e:
    print(f"❌ 错误: 缺少必要的模块文件。\n详细信息: {e}")
    sys.exit(1)

# ================= ⚙️ 配置区域 =================
CONFIG = {
    'raw_ct_dir': './Pancreas-CT',              
    'raw_label_dir': './Pancreas-CT-Label',     
    'processed_2d_dir': './data',               
    'processed_3d_dir': './data3D',             
    
    'unet_2d': False,              
    'batch_size': 2,               
    'num_workers': 0,              
    'n_epochs': 1,                 
    'inference_only': False,       
    'train_on_gpu': torch.cuda.is_available(),
    'seed': 51
}

# ================= 🛠️ 工具函数 =================

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def prepare_directories():
    for p in [CONFIG['processed_2d_dir'], CONFIG['processed_3d_dir']]:
        if not os.path.exists(p):
            os.makedirs(p)

def preprocess_data_robust():
    """
    鲁棒的数据预处理函数 (v4 - 智能缓存版)：
    如果检测到数据已存在，直接跳过耗时的生成步骤。
    """
    print("--- 检查数据状态 ---")
    
    # 🚀 优化点 1：检查是否已有数据，有则跳过
    # 检查最后一个病人文件夹是否存在且不为空
    check_patient = os.path.join(CONFIG['processed_2d_dir'], 'Patient0082', 'CT')
    if os.path.exists(check_patient) and len(os.listdir(check_patient)) > 0:
        print("✅ 检测到本地已有预处理数据 (./data)，跳过 PNG 生成步骤。")
        return

    print("🔄 未找到完整数据，开始执行预处理 (这可能需要几分钟)...")
    prepare_directories()

    # 检查 pydicom 版本兼容性
    try:
        if not hasattr(dicomio, 'dcmread'):
            dicomio.dcmread = dicomio.read_file
    except:
        pass

    for i in range(1, 83):
        patient_id = '{:04d}'.format(i)
        
        # 路径准备
        nifti_filename = f"label{patient_id}.nii.gz"
        nifti_path = os.path.join(CONFIG['raw_label_dir'], nifti_filename)
        ct_folder_pattern = os.path.join(CONFIG['raw_ct_dir'], f"PANCREAS_{patient_id}", "**", "*.dcm")
        
        # 1. 检查源文件
        if not os.path.exists(nifti_path):
            # print(f"⚠️  [Patient {patient_id}] 跳过: 找不到标签文件")
            continue
        
        dcm_files = glob.glob(ct_folder_pattern, recursive=True)
        if not dcm_files:
            # print(f"⚠️  [Patient {patient_id}] 跳过: 找不到 DICOM 文件")
            continue

        # 2. 读取并排序 DICOM
        try:
            slices = []
            for f in dcm_files:
                try:
                    ds = dicomio.dcmread(f)
                    slices.append(ds)
                except Exception:
                    pass
            
            if not slices:
                continue

            # 按 Z 轴位置排序
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            
        except Exception as e:
            print(f"❌ [Patient {patient_id}] 处理崩溃: {e}")
            continue

        # 3. 读取 Mask
        try:
            mask_obj = nib.load(nifti_path)
            mask_data = mask_obj.get_fdata()
        except Exception as e:
            print(f"❌ [Patient {patient_id}] NIfTI 读取失败: {e}")
            continue

        # 4. 对齐层数
        num_dcm = len(slices)
        num_mask = mask_data.shape[2]
        valid_slices = min(num_dcm, num_mask)
        
        if valid_slices < 10:
            continue
            
        # 5. 保存 PNG
        save_dir_ct = os.path.join(CONFIG['processed_2d_dir'], 'Patient' + patient_id, 'CT')
        save_dir_mask = os.path.join(CONFIG['processed_2d_dir'], 'Patient' + patient_id, 'Masks')
        os.makedirs(save_dir_ct, exist_ok=True)
        os.makedirs(save_dir_mask, exist_ok=True)

        try:
            for s in range(valid_slices):
                mask_slice = mask_data[:, :, s]
                ct_slice = slices[s].pixel_array.transpose(1, 0) 
                filename = f"{s:04d}.png"
                cv2.imwrite(os.path.join(save_dir_mask, filename), mask_slice)
                cv2.imwrite(os.path.join(save_dir_ct, filename), ct_slice)
            
            # print(f"✅ [Patient {patient_id}] 完成")
            
        except Exception as e:
            print(f"❌ [Patient {patient_id}] 保存出错: {e}")

    print("--- 数据预处理完成 ---")

def main():
    set_seed(CONFIG['seed'])
    
    print(f"CUDA 是否可用: {CONFIG['train_on_gpu']}")
    if CONFIG['train_on_gpu']:
        print(f"使用设备: {torch.cuda.get_device_name(0)}")

    # 1. 智能预处理
    preprocess_data_robust()

    # 2. 构建数据索引
    print("构建文件索引...")
    patient_path_list = {'CT': {}, 'Masks': {}}
    patient_image_cnt_CT = {}
    patient_image_cnt_Mask = {}

    valid_patients = []
    patient_dirs = sorted(glob.glob(os.path.join(CONFIG['processed_2d_dir'], 'Patient*')))
    
    for p_dir in patient_dirs:
        p_key = os.path.basename(p_dir)
        ct_files = sorted(glob.glob(os.path.join(p_dir, 'CT', '*.png')))
        mask_files = sorted(glob.glob(os.path.join(p_dir, 'Masks', '*.png')))
        
        if len(ct_files) > 0 and len(ct_files) == len(mask_files):
            patient_path_list['CT'][p_key] = ct_files
            patient_path_list['Masks'][p_key] = mask_files
            patient_image_cnt_CT[p_key] = len(ct_files)
            patient_image_cnt_Mask[p_key] = len(mask_files)
            valid_patients.append(p_key)

    print(f"有效病例数: {len(valid_patients)}")
    if len(valid_patients) == 0:
        print("❌ 错误: 没有有效病例。请检查数据。")
        return

    # 3. 体积重采样 (智能跳过)
    print("检查 3D 数据缓存...")
    d1 = torch.linspace(-1, 1, 256)
    d2 = torch.linspace(-1, 1, 256)
    d3 = torch.linspace(-1, 1, 128)
    meshx, meshy, meshz = torch.meshgrid((d1, d2, d3), indexing='ij')
    grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0)

    # 🚀 优化点 2：如果 .pt 文件存在，直接跳过生成
    new_pt_count = 0
    for patient in valid_patients:
        out_ct_path = os.path.join(CONFIG['processed_3d_dir'], patient + '_CT.pt')
        if not os.path.exists(out_ct_path):
            try:
                # 只有文件不存在时才调用
                volume_composer(patient, patient_image_cnt_CT, patient_path_list, grid)
                new_pt_count += 1
            except Exception as e:
                print(f"Resizing {patient} error: {e}")
    
    if new_pt_count == 0:
        print("✅ 所有 3D 数据 (.pt) 已存在，跳过重采样步骤。")
    else:
        print(f"🔄 新生成了 {new_pt_count} 个 3D 数据文件。")

    # 4. 训练准备
    print("准备 Dataset...")
    part = partitioning(valid_patients, split_ratio=[0.7, 0.1, 0.2])
    
    kc, kh, kw = 32, 64, 64
    dc, dh, dw = 32, 64, 64

    CT_patches = {}
    mask_patches = {}
    
    print("加载 Patches (这步需要一点内存)...")
    for p in ['train', 'valid']:
        CT_patches[p], mask_patches[p] = patch_creator(part[p], kw, kh, kc, dw, dh, dc)

    dataset_train = Pancreas_3D_dataset(CT_patches['train'], mask_patches['train'], augment=True)
    dataset_valid = Pancreas_3D_dataset(CT_patches['valid'], mask_patches['valid'], augment=False)

    loaders = {
        'train': torch.utils.data.DataLoader(dataset_train, batch_size=CONFIG['batch_size'], 
                                             shuffle=True, num_workers=CONFIG['num_workers']),
        'valid': torch.utils.data.DataLoader(dataset_valid, batch_size=CONFIG['batch_size'], 
                                             shuffle=False, num_workers=CONFIG['num_workers'])
    }

    # 5. 模型训练
    print("初始化模型...")
    model = UNet_3D(1, 1, 32, 0.2)
    if CONFIG['train_on_gpu']:
        model.cuda()

    criterion = TverskyLoss(1e-8, 0.3, 0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    if len(loaders['train']) == 0:
        print("❌ 训练集为空，无法训练。")
        return

    if not CONFIG['inference_only']:
        print(f"开始训练...")
        model = train_3D(CONFIG['n_epochs'], loaders, model, optimizer, criterion, 
                         CONFIG['train_on_gpu'], performance_metrics, 'model.pt', 0.5)

    print("脚本运行结束。")

if __name__ == '__main__':
    main()