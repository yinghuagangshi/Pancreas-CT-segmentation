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
    from loss import TverskyLoss, MixedLoss
    from net import UNet_2D, UNet_3D
    from volume_patch_composer import volume_composer, patch_creator
    from dataset import Pancreas_2D_dataset, Pancreas_3D_dataset, partitioning
    from metrics import performance_metrics
    from train import train_2D, train_3D
    from inference import (get_inference_performance_metrics_3D)
except ImportError as e:
    print(f"❌ 错误: 缺少必要的模块文件。\n详细信息: {e}")
    sys.exit(1)

def process_ct_window(ct_array, w_level=40, w_width=400):
    """
    对 CT 数据进行窗宽窗位调整和归一化。
    胰腺/软组织推荐: WL=40, WW=350~400
    """
    # 1. 应用窗宽窗位
    min_val = w_level - w_width / 2
    max_val = w_level + w_width / 2
    
    ct_clipped = np.clip(ct_array, min_val, max_val)
    
    # 2. 归一化到 [0, 255]
    ct_norm = (ct_clipped - min_val) / (max_val - min_val)
    ct_norm = ct_norm * 255.0
    
    return ct_norm.astype(np.uint8)


# ================= ⚙️ 配置区域 =================
CONFIG = {
    'raw_ct_dir': './Pancreas-CT',              
    'raw_label_dir': './Pancreas-CT-Label',     
    'processed_2d_dir': './data',               
    'processed_3d_dir': './data3D',             
    
    'unet_2d': False,              
    'batch_size': 4,               
    'num_workers': 0,              
    'n_epochs': 50,                # 🚀 修改：正式训练建议设为 50。如果想快速测试，可改回 1 或 5
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

                    # 先转为 float 避免计算溢出
                    image = ds.pixel_array.astype(np.float32)
                    
                    # 应用斜率和截距 (如果存在)
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope)
                        intercept = float(ds.RescaleIntercept)
                        image = image * slope + intercept

                    # slices.append(ds)
                    slices.append((float(ds.ImagePositionPatient[2]), image))
                except Exception:
                    pass
            
            if not slices:
                continue

            # 按 Z 轴位置排序
            # slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            slices.sort(key=lambda x: x[0])
            
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

        # (前面的代码保持不变)
        try:
            for s in range(valid_slices):
                mask_slice = mask_data[:, :, s]
                
                # 获取原始 CT 数据
                # raw_ct_slice = slices[s].pixel_array.transpose(1, 0)
                raw_ct_slice = slices[s][1].transpose(1, 0)
                
                # --- 🔥 修改开始 🔥 ---
                # 1. 对 CT 进行窗位调整和归一化 (关键修复!)
                processed_ct_slice = process_ct_window(raw_ct_slice, w_level=40, w_width=400)
                
                # 2. 确保 Mask 也是 uint8 格式 (0 和 255, 或者 0 和 1)
                # 建议将 Mask 乘以 255 以便肉眼观察，但在读取时要除回来
                mask_slice = (mask_slice * 255).astype(np.uint8)
                # 这里为了兼容你现有的 dataset 代码(假设它读取0/1)，我们保持 0/1 但转为 uint8
                # mask_slice = mask_slice.astype(np.uint8)
                
                # --- 🔥 修改结束 🔥 ---

                filename = f"{s:04d}.png"
                cv2.imwrite(os.path.join(save_dir_mask, filename), mask_slice)
                cv2.imwrite(os.path.join(save_dir_ct, filename), processed_ct_slice)
            
        except Exception as e:
            print(f"❌ [Patient {patient_id}] 保存出错: {e}")      

    print("--- 数据预处理完成 ---")

def main():
    set_seed(CONFIG['seed'])
    
    # ================= 📁 1. 设置结果目录和时间戳 (修改部分) =================
    # 创建 results 文件夹
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 生成时间戳，例如: "20251126-1030"
    import time
    timestamp = time.strftime("%Y%m%d-%H%M")
    experiment_name = f"run_{timestamp}"
    
    print(f"🚀 本次实验ID: {experiment_name}")
    print(f"📂 结果将保存在: {results_dir}/")

    # 定义带路径的保存文件名
    model_save_path = os.path.join(results_dir, f"{experiment_name}_model.pt")
    loss_plot_path = os.path.join(results_dir, f"{experiment_name}_loss_curve.png")
    metric_save_path = os.path.join(results_dir, f"{experiment_name}_metrics.csv")
    test_save_path = os.path.join(results_dir, f"{experiment_name}_inference_results.csv")
    # ====================================================================

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

    new_pt_count = 0
    for patient in valid_patients:
        out_ct_path = os.path.join(CONFIG['processed_3d_dir'], patient + '_CT.pt')
        if not os.path.exists(out_ct_path):
            try:
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
    
    # kc: Kernel Depth (切块的深度/层数)
    # kh: Kernel Height (切块的高度)
    # kw: Kernel Width (切块的宽度)
    kc, kh, kw = 32, 64, 64
    # dc, dh, dw: Stride (滑动窗口的步长，通常设为和上面一样，表示不重叠)
    dc, dh, dw = 32, 64, 64

    CT_patches = {}
    mask_patches = {}
    
    print("加载 Patches (这步需要一点内存)...")
    for p in ['train', 'valid']:
        CT_patches[p], mask_patches[p] = patch_creator(part[p], kw, kh, kc, dw, dh, dc)

    dataset_train = Pancreas_3D_dataset(CT_patches['train'], mask_patches['train'], augment=True, is_train=True)
    dataset_valid = Pancreas_3D_dataset(CT_patches['valid'], mask_patches['valid'], augment=False , is_train=False)

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

    # 修改为你实际的模型文件名
    checkpoint_path = './results/run_20251126-1659_model.pt' 
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 正在加载预训练模型: {checkpoint_path}")
        # 加载权重
        model.load_state_dict(torch.load(checkpoint_path))
        print("✅ 加载成功！将在现有基础上继续训练。")
    else:
        print("⚠️ 未找到预训练模型，将从头开始训练。")   

    # ✅ 使用新的混合 Loss
    # alpha=0.7 强调召回，bce_weight=0.5 提供梯度平滑
    # criterion = MixedLoss(alpha=0.7, beta=0.3, bce_weight=0.5) 

    criterion = TverskyLoss(1e-6, 0.7, 0.3)
    # 1. 定义基础优化器 (LR 会被 Scheduler 覆盖，所以这里初始 LR 可以随意，但建议设为 max_lr 的 1/10 或 1/25)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 2.定义 OneCycleLR
    # max_lr: 最大学习率，可以尝试 1e-3 或 5e-4
    # steps_per_epoch: 每个 epoch 的 batch 数量
    # epochs: 总 epoch 数
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(loaders['train']), 
        epochs=CONFIG['n_epochs']
    )
    
    if len(loaders['train']) == 0:
        print("❌ 训练集为空，无法训练。")
        return

    if not CONFIG['inference_only']:
        print(f"开始训练 ({CONFIG['n_epochs']} epochs)...")

        # 3. 把 scheduler 传进去
        model = train_3D(CONFIG['n_epochs'], loaders, model, optimizer, criterion, 
                         CONFIG['train_on_gpu'], performance_metrics, model_save_path,metric_save_path, 0.5, 
                         scheduler=scheduler) # 传入 scheduler       
        
        # 处理 Loss 曲线和 Metrics
        if os.path.exists(metric_save_path):
            try:
                df = pd.read_csv(metric_save_path)
                
                # 绘图并保存到 results 文件夹
                plt.figure()
                plt.plot(df['epoch'], df['Training Loss'], label='Train')
                plt.plot(df['epoch'], df['Validation Loss'], label='Valid')
                plt.legend()
                plt.title(f'Training Process ({experiment_name})')
                plt.savefig(loss_plot_path) # 修改保存路径
                print(f"✅ Loss 曲线已保存: {loss_plot_path}")
                plt.close() # 关闭图表释放内存
                
            except Exception as e:
                print(f"保存曲线出错: {e}")

    # 6. 测试集推理 (Evaluation)
    print("\n--- 开始测试集评估 ---")
    # 修改：从新的 model_save_path 加载模型
    if os.path.exists(model_save_path):
        print(f"加载模型权重: {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        print(f"正在测试 {len(part['test'])} 个测试集病例...")
        df_test = get_inference_performance_metrics_3D(model, part['test'], Pancreas_3D_dataset, 
                                                  CONFIG['batch_size'], CONFIG['train_on_gpu'], 
                                                  0.5, kw, kh, kc, dw, dh, dc)
        print("\n📊 测试集结果统计:")
        print(df_test.describe())
        
        # 保存到 results 文件夹
        df_test.to_csv(test_save_path, index=False)
        print(f"✅ 详细测试结果已保存至: {test_save_path}")

    else:
        print(f"⚠️ 未找到模型文件 {model_save_path}，跳过测试。")

    print("脚本全部运行结束。")

if __name__ == '__main__':
    main() # python -u "e:\Pancreas-CT-segmentation\pancreas_segmentation_robust.py"