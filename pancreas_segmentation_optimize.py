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
import pydicom as dicomio

import torch
import torch.optim as optim
from torchsummary import summary
# 如果不想用 torch_lr_finder，可以将下面这行注释掉，并在配置中把 lr_find 设为 False
try:
    from torch_lr_finder import LRFinder
except ImportError:
    print("未找到 torch_lr_finder，将跳过 LR 搜索功能。")

# ================= 导入本地模块 =================
# 确保 dataset.py, net.py 等文件就在同一级目录下
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
    print("请确保 dataset.py, net.py, loss.py 等文件在当前目录中。")
    sys.exit(1)

# ================= ⚙️ 配置区域 (根据你的截图调整) =================
CONFIG = {
    # 原始数据路径 (根据截图)
    'raw_ct_dir': './Pancreas-CT',              # 存放 DICOM 的文件夹
    'raw_label_dir': './Pancreas-CT-Label',     # 存放 .nii.gz 的文件夹
    
    # 预处理输出路径 (脚本自动生成)
    'processed_2d_dir': './data',               # 转换后的 PNG 存放处
    'processed_3d_dir': './data3D',             # Resize 后的 Tensor 存放处
    
    # 训练参数
    'unet_2d': False,              # 默认为 3D 分割
    'batch_size': 2,               # 本地显存通常较小，建议设为 2 或 4
    'num_workers': 0,              # Windows 下建议设为 0，避免多进程报错
    'n_epochs': 1,                 # 演示用 1 个 epoch，实际训练可改为 50+
    'lr_find': False,
    'inference_only': False,       # 如果只想测试，设为 True
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
    """确保输出目录存在"""
    for p in [CONFIG['processed_2d_dir'], CONFIG['processed_3d_dir']]:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"创建目录: {p}")

def preprocess_data():
    """
    核心预处理逻辑：
    1. 读取 Pancreas-CT-Label 中的 .nii.gz -> 转为 PNG 存入 ./data/PatientXXX/Masks
    2. 读取 Pancreas-CT 中的 DICOM -> 转为 PNG 存入 ./data/PatientXXX/CT
    """
    print("--- 开始数据预处理 ---")
    
    # 检查是否已经处理过 (检查 Patient0001 是否存在)
    check_path = os.path.join(CONFIG['processed_2d_dir'], 'Patient0001', 'CT')
    if os.path.exists(check_path) and len(os.listdir(check_path)) > 0:
        print("检测到 ./data 目录已有数据，跳过 PNG 转换步骤。")
        return

    # 初始化病人文件夹
    for i in range(1, 83):
        patient_id = '{:04d}'.format(i)
        p_folder = os.path.join(CONFIG['processed_2d_dir'], 'Patient' + patient_id)
        os.makedirs(os.path.join(p_folder, 'Masks'), exist_ok=True)
        os.makedirs(os.path.join(p_folder, 'CT'), exist_ok=True)

    # 1. 处理 Masks (NIfTI -> PNG)
    print("正在处理 Masks (NIfTI -> PNG)...")
    for i in range(1, 83):
        patient_id = '{:04d}'.format(i)
        # 假设文件名格式为 label0001.nii.gz
        nifti_filename = f"label{patient_id}.nii.gz"
        nifti_path = os.path.join(CONFIG['raw_label_dir'], nifti_filename)
        
        if not os.path.exists(nifti_path):
            print(f"⚠️ 跳过: 找不到标签文件 {nifti_path}")
            continue

        try:
            img = nib.load(nifti_path)
            img_data = img.get_fdata()
            
            # 保存每一层切片
            for s in range(img_data.shape[2]):
                slice_label = '{:03d}'.format(s + 1)
                slice_img = img_data[:, :, s]
                save_path = os.path.join(CONFIG['processed_2d_dir'], 'Patient' + patient_id, 
                                         'Masks', f"M_{slice_label}.png")
                cv2.imwrite(save_path, slice_img)
        except Exception as e:
            print(f"处理 Mask {patient_id} 出错: {e}")

    # 2. 处理 CT (DICOM -> PNG)
    print("正在处理 CT (DICOM -> PNG)... 这可能需要几分钟")
    for i in range(1, 83):
        patient_id = '{:04d}'.format(i)
        # 搜索 DICOM 文件，结构通常是 Pancreas-CT/PANCREAS_0001/.../*.dcm
        # 使用 recursive=True 来穿透多层子文件夹
        search_pattern = os.path.join(CONFIG['raw_ct_dir'], f"PANCREAS_{patient_id}", "**", "*.dcm")
        dcm_files = glob.glob(search_pattern, recursive=True)

        if not dcm_files:
            print(f"⚠️ 跳过: 找不到 Patient {patient_id} 的 DICOM 文件")
            continue

        for f in dcm_files:
            try:
                # 文件名通常包含切片序号，例如 1-001.dcm
                file_name = os.path.basename(f)
                # 尝试提取中间的数字部分作为序号
                parts = file_name.replace('.dcm', '').split('-')
                if len(parts) > 1:
                    slice_idx = parts[-1] 
                else:
                    slice_idx = parts[0] # fallback

                save_path = os.path.join(CONFIG['processed_2d_dir'], 'Patient' + patient_id, 
                                         'CT', f"CT_{slice_idx}.png")
                
                dcm = dicomio.read_file(f)
                img_array = dcm.pixel_array
                # 根据原代码逻辑，需要转置 (Transpose)
                cv2.imwrite(save_path, img_array.transpose(1, 0))
            except Exception as e:
                pass # 忽略单个文件错误
    print("数据转换完成。")

# ================= 🚀 主程序逻辑 =================

def main():
    set_seed(CONFIG['seed'])
    prepare_directories()
    
    print(f"CUDA 是否可用: {CONFIG['train_on_gpu']}")
    if CONFIG['train_on_gpu']:
        print(f"使用设备: {torch.cuda.get_device_name(0)}")

    # 1. 执行数据预处理
    preprocess_data()

    # 2. 构建数据索引字典 (这是 volume_composer 需要的格式)
    print("构建文件索引...")
    patient_path_list = {'CT': {}, 'Masks': {}}
    patient_image_cnt_CT = {}
    patient_image_cnt_Mask = {}

    valid_patients = []
    # 扫描生成的 data 目录
    patient_dirs = sorted(glob.glob(os.path.join(CONFIG['processed_2d_dir'], 'Patient*')))
    
    for p_dir in patient_dirs:
        p_key = os.path.basename(p_dir) # e.g., "Patient0001"
        
        ct_files = sorted(glob.glob(os.path.join(p_dir, 'CT', '*.png')))
        mask_files = sorted(glob.glob(os.path.join(p_dir, 'Masks', '*.png')))
        
        if len(ct_files) > 0 and len(ct_files) == len(mask_files):
            patient_path_list['CT'][p_key] = ct_files
            patient_path_list['Masks'][p_key] = mask_files
            patient_image_cnt_CT[p_key] = len(ct_files)
            patient_image_cnt_Mask[p_key] = len(mask_files)
            valid_patients.append(p_key)
        else:
            # print(f"跳过不完整数据: {p_key} (CT: {len(ct_files)}, Mask: {len(mask_files)})")
            pass

    print(f"有效病例数: {len(valid_patients)}")

    # 3. 体积重采样 (Volume Resize -> 3D Tensor)
    print("执行 3D 体积重采样 (生成 .pt 文件)...")
    d1 = torch.linspace(-1, 1, 256)
    d2 = torch.linspace(-1, 1, 256)
    d3 = torch.linspace(-1, 1, 128)
    meshx, meshy, meshz = torch.meshgrid((d1, d2, d3), indexing='ij')
    grid = torch.stack((meshx, meshy, meshz), 3).unsqueeze(0)

    # 调用 volume_patch_composer.py 中的函数
    # 注意：需要确保 volume_patch_composer.py 里的保存路径也是指向 CONFIG['processed_3d_dir']
    # 如果原文件写死了 '/content/data3D/'，需要你手动去改一下那个文件，或者我们这里 monkey patch 一下
    # 这里假设我们传递正确的字典进去
    
    # 为了避免修改 volume_patch_composer.py，我们在这里手动检查并生成
    # 原函数 volume_composer 内部路径可能写死了，建议去修改 volume_patch_composer.py:
    # 将 '/content/data3D/' 替换为 './data3D/'
    
    for patient in valid_patients:
        out_ct_path = os.path.join(CONFIG['processed_3d_dir'], patient + '_CT.pt')
        if not os.path.exists(out_ct_path):
            try:
                # 尝试调用，如果 volume_patch_composer 内部写死了路径可能会存错地方
                # 建议打开 volume_patch_composer.py 把所有 /content/data3D 改为 ./data3D
                volume_composer(patient, patient_image_cnt_CT, patient_path_list, grid)
                
                # Hack: 如果它存到了默认位置 (例如根目录)，移动它
                if os.path.exists(f'/content/data3D/{patient}_CT.pt'):
                    shutil.move(f'/content/data3D/{patient}_CT.pt', out_ct_path)
            except Exception as e:
                print(f"Resizing {patient} error: {e}")
                # 可以在这里重写简单的 resize 逻辑，但为了利用原代码暂且如此

    # 检查 data3D 文件夹是否有内容，如果没有，提示用户修改 volume_patch_composer.py
    if not os.listdir(CONFIG['processed_3d_dir']):
        print("❌ 警告: data3D 文件夹为空。")
        print("请打开 'volume_patch_composer.py' 文件，将里面所有的 '/content/data3D/' 替换为 './data3D/'，然后重新运行。")
        return

    # 4. 数据划分与加载
    print("准备 Dataset 和 DataLoader...")
    part = partitioning(valid_patients, split_ratio=[0.7, 0.1, 0.2])

    # 3D 参数
    kc, kh, kw = 32, 64, 64
    dc, dh, dw = 32, 64, 64

    CT_patches = {}
    mask_patches = {}
    
    # 同样，patch_creator 内部可能也有路径硬编码，请检查 volume_patch_composer.py
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

    # 5. 模型与训练
    print("初始化 3D UNet 模型...")
    model = UNet_3D(1, 1, 32, 0.2)
    if CONFIG['train_on_gpu']:
        model.cuda()

    criterion = TverskyLoss(1e-8, 0.3, 0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.08, 
                                                    steps_per_epoch=len(loaders['train']), 
                                                    epochs=CONFIG['n_epochs'])

    if not CONFIG['inference_only']:
        print(f"开始训练 ({CONFIG['n_epochs']} epochs)...")
        # 调用 train.py 中的 train_3D
        model = train_3D(CONFIG['n_epochs'], loaders, model, optimizer, criterion, 
                         CONFIG['train_on_gpu'], performance_metrics, 'model.pt', 0.5)
        
        # 保存 Loss 曲线
        if os.path.exists('performance_metrics.csv'):
            df = pd.read_csv('performance_metrics.csv')
            plt.figure()
            plt.plot(df['epoch'], df['Training Loss'], label='Train')
            plt.plot(df['epoch'], df['Validation Loss'], label='Valid')
            plt.legend()
            plt.title('Training Process')
            plt.savefig('loss_curve.png')
            print("训练完成，Loss 曲线已保存为 loss_curve.png")

    # 6. 推理测试
    print("开始测试集推理...")
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt'))
    
    # 同样需要注意 inference.py 内部是否也有路径硬编码
    df = get_inference_performance_metrics_3D(model, part['test'], Pancreas_3D_dataset, 
                                              CONFIG['batch_size'], CONFIG['train_on_gpu'], 
                                              0.5, kw, kh, kc, dw, dh, dc)
    print("\n测试结果统计:")
    print(df.describe())
    df.to_csv('inference_results.csv')

if __name__ == '__main__':
    main()