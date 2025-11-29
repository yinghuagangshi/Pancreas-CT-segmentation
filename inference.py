#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from volume_patch_composer import  patch_creator
from metrics import performance_metrics
import nibabel as nib
import os



#-----------------------------------------------------------------------#
#             get_inference_performance_metrics_2D                      #
#  Performs prediction on the test dataset, return the performance      #
#  metrics for each patient                                             #
#-----------------------------------------------------------------------#
# Returns inference metrics table                                       #
#-----------------------------------------------------------------------#
# model:         Trained model                                          #  
# part:          A list of patients in the test partition               #
# dataset_test:  Test dataset which is grouped per patient              #
# threshold:     Threshold value to create binary image                 #
#-----------------------------------------------------------------------#
def get_inference_performance_metrics_2D(model, part, dataset_test,
                                         batch_size, train_on_gpu, threshold):
    # Initialize a list to keep track of test performance metrics    
    test_metrics =[]
    
    # Set the model to inference mode
    model.eval()
    for p in part:
        # Test dataloader per patient
        loaders = torch.utils.data.DataLoader(dataset_test[p], 
                                                      batch_size=batch_size, 
                                                      shuffle=False, 
                                                      num_workers= 0)
        
        # Initialize variables to monitor performance metrics
        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        # initialize the number of test instances
        test_cnt = 0
        
        for batch_idx, (data, target) in enumerate(loaders):
            # Move image & mask Pytorch Tensor to GPU if CUDA is available.
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass (inference) to get the output
            output = model(data)
            output = output.cpu().detach().numpy()
            # Binarize the output
            output_b = (output>threshold)*1
            output_b = np.squeeze(output_b)
            batch_l = output_b.size
            # Update the total number of inference pairs 
            test_cnt += batch_l
            t1 = transforms.ToTensor()
            # Transform output back to Pytorch Tensor and move it to GPU
            output_b = t1(output_b)
            output_b = output_b.cuda()
            m = performance_metrics(smooth = 1e-6)
            # Get average metrics per batch
            specificity, sensitivity, precision, F1_score, F2_score, DSC = m(
                output_b, target)    
            specificity_val += specificity * batch_l
            sensitivity_val += sensitivity * batch_l
            precision_val += precision * batch_l
            F1_score_val += F1_score * batch_l
            F2_score_val += F2_score * batch_l 
            DSC_val += DSC * batch_l 
            
       
    # Calculate the overall average metrics   
    specificity_val, sensitivity_val, precision_val, F1_score_val, 
    F2_score_val, DSC_val = specificity_val/test_cnt, sensitivity_val/test_cnt,
    precision_val/test_cnt, F1_score_val/test_cnt, F2_score_val/test_cnt, 
    DSC_val/test_cnt
    # Add each patient's prediction metrics to the list
    test_metrics.append((p, specificity_val, sensitivity_val, precision_val,
                         F1_score_val, F2_score_val, DSC_val ))
    #save the test metrics as a table
    df=pd.DataFrame.from_records(test_metrics, columns=[
        'Patient','specificity', 'sensitivity', 'precision', 'F1_score',
        'F2_score', 'DSC'])
    df.to_csv('test_metrics.csv', index=False)       
    #return the inference metrics table
    return df


#-----------------------------------------------------------------------#
#             get_inference_performance_metrics_3D                      #
#  Builds a test dataset and dataloader per patient and performs        #
#  prediction on the test dataset, return the permormance metrics for   #
#  each patient.                                                        #
#-----------------------------------------------------------------------#
# Returns inference metrics table                                       #
#-----------------------------------------------------------------------#
# model:                Trained model                                   #  
# part:                 A list of patients in the test partition        #
# Pancreas_3D_dataset:  3D dataset which is grouped per patient         #
# threshold:            Threshold value to create binary image          #
#-----------------------------------------------------------------------#
def get_inference_performance_metrics_3D(model, part, Pancreas_3D_dataset, 
                                    batch_size, train_on_gpu, threshold,
                                    kw, kh, kc, dw, dh, dc):
    test_metrics = []
    for patient in part:
        # Set the model to inference mode
        model.eval()
        # Create subvolumes (patches) for patient's CT and mask
        CT_patches = []
        mask_patches =[]
        CT_patches, mask_patches = patch_creator([patient], kw, kh, kc, 
                                                  dw, dh, dc) 
        dataset_test= Pancreas_3D_dataset (CT_patches, mask_patches,
                                            augment= False)
        loaders_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=0)
        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        # initialize the number of test instances
        valid_cnt = 0
        for batch_idx, (data, target) in enumerate(loaders_test):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass
            output = model(data)

            output = torch.sigmoid(output) # 🔥 必须加！把 Logits 转回概率
            output = output.cpu().detach().numpy()

            # Binarize the output
            output_b = (output>threshold)*1
            output_b = np.squeeze(output_b)
            batch_l = output_b.size
            # update the total number of validation pairs
            valid_cnt += batch_l
            #t1 = transforms.ToTensor()
            # Transform output back to Pytorch Tensor and move it to GPU
            #output_b = t1(output_b)
            output_b = torch.as_tensor(output_b)
            output_b = output_b.cuda()
            # calculate average performance metrics per batches
            m = performance_metrics(smooth = 1e-6)
            specificity, sensitivity, precision, F1_score, F2_score, DSC =  m(output_b, target)    
            
            specificity_val += specificity * batch_l
            sensitivity_val += sensitivity * batch_l
            precision_val += precision * batch_l
            F1_score_val += F1_score * batch_l
            F2_score_val += F2_score * batch_l 
            DSC_val += DSC * batch_l 
            # Calculate the overall average metrics    
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val/valid_cnt, sensitivity_val/valid_cnt, precision_val/valid_cnt, F1_score_val/valid_cnt, F2_score_val/valid_cnt, DSC_val/valid_cnt

        # Add each patient's prediction metrics to the list
        test_metrics.append((patient,specificity_val, sensitivity_val, 
                             precision_val, F1_score_val, 
                             F2_score_val, DSC_val ))
        #save the test metrics as a table
    df=pd.DataFrame.from_records(test_metrics, 
                                 columns=['Patient', 'specificity', 
                                          'sensitivity', 'precision', 
                                          'F1_score', 'F2_score', 'DSC' ])
    # df.to_csv('test_metrics.csv', index=False)       
    #return the inference metrics table
    return df

#-----------------------------------------------------------------------#
#                    visualize_patient_prediction_2D                    #
#  Performs prediction on a specific patient of the test dataset,       # 
#  Plot the image trio: image, mask and prediction                      #
#-----------------------------------------------------------------------#
# model:         Trained model                                          #  
# patient:       patient ID/label                                       #
# dataset_test:  Test dataset which is grouped per patient              #
# threshold:     Threshold value to create binary image                 #
#-----------------------------------------------------------------------#
def visualize_patient_prediction_2D(model, patient, dataset_test, batch_size, 
                                 train_on_gpu, threshold):
    loaders_test = torch.utils.data.DataLoader(dataset_test[patient], 
                                                  batch_size=batch_size, 
                                                  shuffle=False,
                                                  num_workers=0)
    # Set the model to inference mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders_test):
        # move to GPU
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output = output.cpu().detach().numpy()
        # Binarize the output
        output_b = (output>threshold)*1
        output_b = np.squeeze(output_b)
        # Plot the image trio: image, mask and prediction
        for gt, pred in zip(target.cpu().numpy(), output_b):
            gt = np.squeeze(gt)
            pred = np.squeeze(pred)
            plt.figure(figsize=(3,6))
            plt.subplot(1,2,1)
            plt.imshow(gt, cmap="gray", interpolation= None)
            plt.subplot(1,2,2)
            plt.imshow(pred, cmap="gray", interpolation= None)
        

#-----------------------------------------------------------------------#
#                              volume                                   #
#  Creates a volume with the the same depth as patch depth. Its height  #
#  and width is the same as the height and width of the resized volumes.#
#-----------------------------------------------------------------------#
# Returns the rth subvolume of image, mask and prediction               #
#-----------------------------------------------------------------------#
# num_patch_width:     number of patches in the width direction         # 
# num_patch_height:    number of patches in the height direction        #
# num_patch_depth:     number of patches in the depth direction         #
# num_batches:         total number of batches                          #
# r:                   the number of the subvolume to be built          #
# CT_subvol:           dictionaries CT patches per batch                #
# mask_subvol:         dictionaries mask patches per batch              #
# predict_subvol:      dictionaries prediction patches per batch        #
# image_vol:           rth subvolume of CT image                        #
# mask_vol:            rth subvolume of mask                            #
# prediction_vol:      rth subvolume of prediction                      #
# kc:                  kernel size in depth direction
#-----------------------------------------------------------------------#
# def volume(num_patch_width, num_patch_height, num_patch_depth, num_batches,
#            r, CT_subvol, mask_subvol, predict_subvol, kc):
#     image_vol = []
#     mask_vol =[]
#     prediction_vol = [] 
#     #sweep in the depth direction
#     for k in range(kc):    
#         idx= 0
#         image = {}
#         mask = {}
#         prediction = {}
#         # sweep in the width and height direction to create layer k of the rth 
#         # subvolume horizontally stack the layer k of patches of each bach and
#         # then sweep in the height direction and create an array for the kth
#         # layer of the final 3D image. Then vertically stack all layers
#         # to build a subvolume. Vertically stacking the subvolumes results
#         # in a 3D image.
#         for q in range(num_batches):
#             for j, (im, m, pred)  in enumerate(zip(CT_subvol[q], mask_subvol[q],
#                                                    predict_subvol[q])):
#                 if j%num_patch_depth == r:
#                     # im = np.squeeze(im).transpose(0,2,1)
#                     # m = np.squeeze(m).transpose(0,2,1)
#                     # pred= pred.transpose(0,2,1)
                                                            
#                     image[idx] = im[k,:,:]
#                     mask[idx] = m[k,:,:]
#                     prediction[idx] = pred[k,:,:]                    
                  
#                     idx+=1
             
#         image_vol.append(np.vstack(tuple([np.hstack(tuple([image[num_patch_width*i + j] 
#                                                            for j in range(num_patch_height)])) 
#                                           for i in range(num_patch_width)])))
#         mask_vol.append(np.vstack(tuple([np.hstack(tuple([mask[num_patch_width*i + j]  
#                                                       for j in range(num_patch_height)])) 
#                                          for i in range(num_patch_width)])))
#         prediction_vol.append(np.vstack(tuple([np.hstack(tuple([prediction[num_patch_width*i + j] 
#                                                                 for j in range(num_patch_height)])) 
#                                                for i in range(num_patch_width)])))
        
#     return image_vol, mask_vol, prediction_vol



#-----------------------------------------------------------------------#
#                              volume                                   #
#-----------------------------------------------------------------------#
def volume(num_patch_width, num_patch_height, num_patch_depth, num_batches,
           r, CT_subvol, mask_subvol, predict_subvol, kc, batch_size):
    image_vol, mask_vol, prediction_vol = [], [], []
    
    # 1. Identify actual available batches to prevent KeyErrors
    available_batches = sorted(CT_subvol.keys())

    for k in range(kc):    
        idx, image, mask, prediction = 0, {}, {}, {}
        
        # 2. Iterate only over existing batches
        for q in available_batches:
            # Iterate over samples in the current batch
            for j, (im, m, pred)  in enumerate(zip(CT_subvol[q], mask_subvol[q], predict_subvol[q])):
                
                # 3. Calculate Global Index safely
                global_idx = q * batch_size + j
                
                # 4. Check if this patch belongs to the current depth row 'r'
                if global_idx % num_patch_depth == r:
                    image[idx] = im[k, :, :]
                    mask[idx] = m[k, :, :]
                    prediction[idx] = pred[k, :, :]
                    idx+=1
        
        # 5. Robust Stacking (Handling potential missing indices if necessary, though ideally idx aligns)
        # Note: This logic assumes 'idx' increments perfectly to match num_patch_width * num_patch_height
        try:
            current_image_layer = np.vstack(tuple([np.hstack(tuple([image[num_patch_width*i + j] 
                                                               for j in range(num_patch_height)])) 
                                              for i in range(num_patch_width)]))
            
            current_mask_layer = np.vstack(tuple([np.hstack(tuple([mask[num_patch_width*i + j]  
                                                          for j in range(num_patch_height)])) 
                                             for i in range(num_patch_width)]))
            
            current_pred_layer = np.vstack(tuple([np.hstack(tuple([prediction[num_patch_width*i + j] 
                                                                    for j in range(num_patch_height)])) 
                                                   for i in range(num_patch_width)]))
            
            image_vol.append(current_image_layer)
            mask_vol.append(current_mask_layer)
            prediction_vol.append(current_pred_layer)
            
        except KeyError as e:
            print(f"Error reconstruction layer {k}: Missing patch index {e}")
            # Optional: Append zero-arrays or handle error appropriately
            return [], [], []

        
    return image_vol, mask_vol, prediction_vol


#-----------------------------------------------------------------------#
#               visualize_patient_prediction_3D                         #
#  Creates 3D images of patient's CT, mask, and prediction, and save    #
#  them as nibabel file. Plot sample of cross sections (slices.)        #
#-----------------------------------------------------------------------#
# model:                Trained model                                   #  
# patient:              patient ID/label                                #
# Pancreas_3D_dataset:  3D dataset which is grouped per patient         #
# threshold:            Threshold value to create binary image          #
# kc, kh, kw:           kernel size (patch parameters for volumetric    #
#                       segmentation)                                   #
# dc, dh, dw:           stride (patch parameters for volumetric         #
#                       segmentation)                                   #
# num_patch_width:      number of patches in the width direction        # 
# num_patch_height:     number of patches in the height direction       #
# num_patch_depth:      number of patches in the depth direction        #
# num_batches:          total number of batches                         #
# r:                    the number of the subvolume to be built         #
# CT_subvol:            dictionaries CT patches per batch               #
# mask_subvol:          dictionaries mask patches per batch             #
# predict_subvol:       dictionaries prediction patches per batch       #
# image_vol:            rth subvolume of CT image                       #
# mask_vol:             rth subvolume of mask                           #
# prediction_vol:       rth subvolume of prediction                     #
# image_volume:         3D CT image                                     #
# mask_volume:          3D annotation mask image                        #
# prediction_volume:    3D prediction image                             #
#-----------------------------------------------------------------------#
# def visualize_patient_prediction_3D(model, patient, Pancreas_3D_dataset, 
#                                     batch_size, train_on_gpu, threshold,
#                                     kw, kh, kc, dw, dh, dc):
#     # Set the model to inference mode
#     model.eval()
#     # Create subvolumes (patches) for patient's CT and mask
#     CT_patches = []
#     mask_patches =[]
#     CT_patches, mask_patches = patch_creator([patient], kw, kh, kc, 
#                                               dw, dh, dc) 
#     dataset_test= Pancreas_3D_dataset (CT_patches, mask_patches,
#                                         augment= False)
#     loaders_test = torch.utils.data.DataLoader(dataset_test, 
#                                                 batch_size=batch_size, 
#                                                 shuffle=False, 
#                                                 num_workers=0)
#     # Create dictionaries of prediction, CT and mask subvolumes per batch
#     predict_subvol= {}
#     CT_subvol = {}
#     mask_subvol ={}

#     for batch_idx, (data, target) in enumerate(loaders_test):
#         # move to GPU
#         if train_on_gpu:
#             data, target = data.cuda(), target.cuda()
#         # forward pass
#         output = model(data)
#         output = output.cpu().detach().numpy()
#         # Binarize the output
#         output_b = (output>threshold)*1
        
#         # --- 核心修改开始 ---
#         # 不要使用无差别的 squeeze。
#         # data 形状是 (Batch, Channel, D, H, W)，我们要去掉 Channel(第1维)，保留 Batch(第0维)
#         # 即使 Batch=1，也要保留它！
        
#         # 处理 Prediction (假设输出是 Batch, 1, D, H, W 或 Batch, D, H, W)
#         # 如果 output_b 有 5 维，去掉第 1 维；如果是 4 维就不动
#         if output_b.ndim == 5:
#             predict_subvol[batch_idx] = np.squeeze(output_b, axis=1)
#         else:
#             predict_subvol[batch_idx] = output_b

#         # 处理 CT Image (Batch, 1, D, H, W) -> (Batch, D, H, W)
#         ct_numpy = data.cpu().detach().numpy()
#         CT_subvol[batch_idx] = np.squeeze(ct_numpy, axis=1)

#         # 处理 Mask (Batch, 1, D, H, W) -> (Batch, D, H, W)
#         target_numpy = target.cpu().detach().numpy()
#         mask_subvol[batch_idx] = np.squeeze(target_numpy, axis=1)
#         # --- 核心修改结束 ---
        
#         # predict_subvol[batch_idx] = np.squeeze(output_b)
#         # CT_subvol[batch_idx] = np.squeeze(data.cpu().detach().numpy())
#         # mask_subvol[batch_idx] = np.squeeze(target.cpu().detach().numpy())

#     num_batches = 256*256*128 // (kc*kh*kw*batch_size)
#     num_patch_depth = 128//kc
#     num_patch_width = 256//kw
#     num_patch_height = 256//kh
#     image_volume = []
#     mask_volume =[]
#     prediction_volume =[]
#     #sweep along the depth direction, create subvolumes and merge them to build 
#     #the final 3D image
#     for r in range(num_patch_depth):
#         image_vol, mask_vol, prediction_vol = volume(num_patch_width, num_patch_height, 
#                                                      num_patch_depth, num_batches, 
#                                                      r, CT_subvol, mask_subvol,
#                                                      predict_subvol, kc)
#         image_volume.extend(image_vol)
#         mask_volume.extend(mask_vol)
#         prediction_volume.extend(prediction_vol)

#     nifti_image_np=np.array(image_volume)
#     nifti_image = nib.Nifti1Image(nifti_image_np, np.eye(4))  # Save axis for data (just identity)
#     nifti_mask_np=np.array(mask_volume)
#     nifti_mask = nib.Nifti1Image(nifti_mask_np, np.eye(4))  # Save axis for data (just identity)
#     nifti_prediction_np=np.array(prediction_volume).astype('int32')
#     nifti_prediction = nib.Nifti1Image(nifti_prediction_np, np.eye(4))  # Save axis for data (just identity)

#     nifti_image.header.get_xyzt_units()
#     nifti_image.to_filename('results/image.nii.gz')  # Save as NiBabel file
#     nifti_mask.header.get_xyzt_units()
#     nifti_mask.to_filename('results/mask.nii.gz')  # Save as NiBabel file
#     nifti_prediction.header.get_xyzt_units()
#     nifti_prediction.to_filename('results/prediction.nii.gz')  # Save as NiBabel file
    
#     #plot sample of image cross sections: CT, mask and predictions
#     for k in range(0,128,8):
#         plt.figure(figsize=(16,16))

#         # plt.subplot(1,4,1)
#         # plt.imshow(nifti_image_np[k,:,:])
#         # plt.title('CT')
#         # plt.subplot(1,4,2)
#         # plt.imshow(nifti_image_np[k,:,:])
#         # plt.imshow(nifti_mask_np[k,:,:], cmap="jet", alpha = 0.3, interpolation= None)  
#         # plt.title('CT and mask')
#         # plt.subplot(1,4,3)
#         # plt.imshow(nifti_image_np[k,:,:])
#         # plt.imshow(nifti_prediction_np[k,:,:], cmap="jet", alpha = 0.3, interpolation= None)  
#         # plt.title('CT and prediction')
#         # plt.subplot(1,4,4)
#         # plt.imshow(nifti_prediction_np[k,:,:])
#         # plt.imshow(nifti_mask_np[k,:,:], cmap="jet", alpha = 0.7, interpolation= None)
#         # plt.title('mask and prediction')

#         # --- 1. 纯 CT 图像 ---
#         plt.subplot(1, 4, 1)
#         # 修改点：加上 cmap='gray'
#         plt.imshow(nifti_image_np[k, :, :], cmap='gray')
#         plt.title('CT')

#         # --- 2. CT + 真实 Mask ---
#         plt.subplot(1, 4, 2)
#         plt.imshow(nifti_image_np[k, :, :], cmap='gray') # 先画黑白底图
        
#         # 修改点：处理 Mask，把值为0的背景变透明
#         mask_data = nifti_mask_np[k, :, :]
#         masked_mask = np.ma.masked_where(mask_data == 0, mask_data)
#         # 使用红色显示 Mask，背景完全透明
#         plt.imshow(masked_mask, cmap='Reds', alpha=0.6, interpolation='none')
#         plt.title('CT and Ground Truth')

#         # --- 3. CT + 预测 Prediction ---
#         plt.subplot(1, 4, 3)
#         plt.imshow(nifti_image_np[k, :, :], cmap='gray') # 先画黑白底图
        
#         # 修改点：同样处理 Prediction 的背景
#         pred_data = nifti_prediction_np[k, :, :]
#         masked_pred = np.ma.masked_where(pred_data == 0, pred_data)
#         # 使用橙黄色显示预测，方便区分
#         plt.imshow(masked_pred, cmap='autumn', alpha=0.6, interpolation='none')
#         plt.title('CT and Prediction')

#         # --- 4. Mask 和 Prediction 对比 ---
#         plt.subplot(1, 4, 4)
#         # 这里不需要画 CT 底图，直接对比两个 Mask
#         # 画真实 Mask (红色)
#         plt.imshow(masked_mask, cmap='Reds', alpha=0.5, interpolation='none')
#         # 画预测 Mask (绿色或蓝色，用于区分)
#         masked_pred_only = np.ma.masked_where(pred_data == 0, pred_data)
#         plt.imshow(masked_pred_only, cmap='cool', alpha=0.5, interpolation='none') 
#         plt.title('GT(Red) vs Pred(Cyan)')
        
#         plt.show() # 确保在循环里展示出来


#-----------------------------------------------------------------------#
#               visualize_patient_prediction_3D                         #
#-----------------------------------------------------------------------#
def visualize_patient_prediction_3D(model, patient, Pancreas_3D_dataset, 
                                    batch_size, train_on_gpu, threshold,
                                    kw, kh, kc, dw, dh, dc):
    print(f"🚀 正在处理病人: {patient} ...")
    
    # --- 1. 模型推理 ---
    model.eval()
    CT_patches, mask_patches = patch_creator([patient], kw, kh, kc, dw, dh, dc) 
    dataset_test = Pancreas_3D_dataset(CT_patches, mask_patches, augment=False)
    loaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predict_subvol, CT_subvol, mask_subvol = {}, {}, {}

    for batch_idx, (data, target) in enumerate(loaders_test):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)

        output = torch.sigmoid(output) # 🔥 必须加！把 Logits 转回概率
        output = output.cpu().detach().numpy()
        output_b = (output > threshold) * 1
        
        # 兼容性修复：防止 squeeze 掉 batch 维度
        if output_b.ndim == 5:
            predict_subvol[batch_idx] = np.squeeze(output_b, axis=1)
        else:
            predict_subvol[batch_idx] = output_b
            
        CT_subvol[batch_idx] = np.squeeze(data.cpu().detach().numpy(), axis=1)
        mask_subvol[batch_idx] = np.squeeze(target.cpu().detach().numpy(), axis=1)

    # --- 2. 拼图还原 ---
    num_batches = 256*256*128 // (kc*kh*kw*batch_size)
    num_patch_depth = 128//kc
    num_patch_width = 256//kw
    num_patch_height = 256//kh
    image_volume, mask_volume, prediction_volume = [], [], []

    for r in range(num_patch_depth):
        # 调用上面修复好的 volume 函数
        image_vol, mask_vol, prediction_vol = volume(num_patch_width, num_patch_height, 
                                                     num_patch_depth, num_batches, 
                                                     r, CT_subvol, mask_subvol,
                                                     predict_subvol, kc, batch_size)
        image_volume.extend(image_vol)
        mask_volume.extend(mask_vol)
        prediction_volume.extend(prediction_vol)

    nifti_image_np = np.array(image_volume)
    nifti_mask_np = np.array(mask_volume)
    nifti_prediction_np = np.array(prediction_volume).astype('int32')

    # --- 3. 保存文件 ---
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.Nifti1Image(nifti_image_np, np.eye(4)).to_filename(os.path.join(save_dir, 'image.nii.gz'))
    nib.Nifti1Image(nifti_mask_np, np.eye(4)).to_filename(os.path.join(save_dir, 'mask.nii.gz'))
    nib.Nifti1Image(nifti_prediction_np, np.eye(4)).to_filename(os.path.join(save_dir, 'prediction.nii.gz'))
    print(f"✅ 文件已保存到 {save_dir}/")
    
    # --- 4. 智能筛选展示层 ---
    # 找出所有包含真值(Mask)的层
    z_indices = np.any(nifti_mask_np, axis=(1, 2))
    valid_slices = np.where(z_indices)[0]
    
    if len(valid_slices) == 0:
        print("❌ 警告：该病人的 Mask 全是空的！(没有红色)")
        # 兜底：画中间层
        plot_indices = [64]
    else:
        # 找出胰腺面积最大的 4 层
        pixel_counts = [np.sum(nifti_mask_np[i]) for i in valid_slices]
        sorted_indices = [x for _, x in sorted(zip(pixel_counts, valid_slices), reverse=True)]
        plot_indices = sorted(sorted_indices[:4]) 
        print(f"🎯 正在展示胰腺面积最大的层: {plot_indices}")

    # --- 5. 绘图 ---
    for k in plot_indices:
        plt.figure(figsize=(16, 4))

        # Subplot 1: CT
        plt.subplot(1, 4, 1)
        plt.imshow(nifti_image_np[k, :, :], cmap='gray')
        plt.title(f'Slice {k} CT')
        plt.axis('off')

        # Subplot 2: CT + GT
        plt.subplot(1, 4, 2)
        plt.imshow(nifti_image_np[k, :, :], cmap='gray')
        mask_data = nifti_mask_np[k, :, :]
        if np.sum(mask_data) > 0:
            masked_mask = np.ma.masked_where(mask_data == 0, mask_data)
            plt.imshow(masked_mask, cmap='Reds', alpha=0.7, interpolation='none')
        plt.title('Ground Truth (Red)')
        plt.axis('off')

        # Subplot 3: CT + Pred
        plt.subplot(1, 4, 3)
        plt.imshow(nifti_image_np[k, :, :], cmap='gray')
        pred_data = nifti_prediction_np[k, :, :]
        if np.sum(pred_data) > 0:
            masked_pred = np.ma.masked_where(pred_data == 0, pred_data)
            plt.imshow(masked_pred, cmap='autumn', alpha=0.7, interpolation='none')
        plt.title('Prediction (Orange)')
        plt.axis('off')

        # Subplot 4: Contrast (Contour Style)
        plt.subplot(1, 4, 4)
        if np.sum(pred_data) > 0:
            masked_pred_only = np.ma.masked_where(pred_data == 0, pred_data)
            plt.imshow(masked_pred_only, cmap='cool', alpha=0.5, interpolation='none')
        
        if np.sum(mask_data) > 0:
            plt.contour(mask_data, colors='red', linewidths=2, linestyles='--')
            
        plt.title('Pred(Cyan) vs GT(Red Line)')
        plt.axis('off')
        
        plt.show()