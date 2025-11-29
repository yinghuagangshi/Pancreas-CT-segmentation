#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

#-----------------------------------------------------------------------#
#                                train_2D                               #
#              Train 2D UNet for some number of epochs                  #
#-----------------------------------------------------------------------#
def train_2D(n_epochs, loaders, model, optimizer, criterion, train_on_gpu, performance_metrics, path, threshold):
    #keep track of loss and performance merics
    loss_and_metrics =[]
    # initialize tracker for max DSC 
    DSC_max = 0
    show_every = 50
    # epoch training loop
    for epoch in tqdm( range(1, n_epochs+1), total = n_epochs+1):
        print(f'=== Epoch #{epoch} ===')
        # initialize variables to monitor training and validation loss, and performance metrics
        train_loss = 0.0
        valid_loss = 0.0
        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        valid_cnt = 0
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        # batch training loop
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # === (BEGIN) ===
            # 仅在第一个 batch 检查，避免刷屏
            if batch_idx == 0:
                print(f"\nDEBUG: Data Shape: {data.shape}")
                print(f"DEBUG: Target Shape: {target.shape}")

                print(f"\nDEBUG: Target Max Value: {target.max().item()}")
                if target.max().item() > 1:
                    print("⚠️ 警告: Mask 值大于 1 (可能是 255)，会导致 Loss 计算错误！")
                else:
                    print("✅ Mask 值正常 (0-1)。")
            # ===  (END) ===

            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            # clear the gradients of all optimized variable
            optimizer.zero_grad() 
            # forward pass (inference) to get the output
            output = model(data) 
            # calculate the batch loss
            loss = criterion(output, target) 
            # backpropagation
            loss.backward() 
            # Update weights
            optimizer.step() 
            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
            
                         
        ######################    
        # validate the model #
        ######################
        print('=== Validation ===')
        # Set the model to inference mode
        model.eval()
        with torch.no_grad():
            # batch training loop
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["valid"])}...')
                # move to GPU
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass (inference) to get the output
                output = model(data)
                # calculate the batch loss
                loss = criterion (output, target)
                # update validation loss
                valid_loss +=  ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                 
                # convert output probabilities to predicted class
                output = output.cpu().detach().numpy()
                # Binarize the output
                output_b = (output>threshold)*1
                output_b = np.squeeze(output_b)
                batch_l = output_b.size
                # update the total number of validation pairs
                valid_cnt += batch_l
                t1 = torch.as_tensor # 使用 torch.as_tensor 替代可能的 transforms.ToTensor()
                # Transform output back to Pytorch Tensor and move it to GPU
                output_b = torch.as_tensor(output_b) # 修正原来的 t1 调用
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

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print('Specificity: {:.6f} \tSensitivity: {:.6f} \tF2_score: {:.6f} \tDSC: {:.6f}'.format(
            specificity_val,
            sensitivity_val, 
            F2_score_val, 
            DSC_val
        ))
        
        
        if DSC_val > DSC_max:
            print('Validation DSC increased.  Saving model ...')            
            torch.save(model.state_dict(), path)
            DSC_max = DSC_val

        loss_and_metrics.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(), specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val))

    #save the loss_epoch and performance metrics history
    df=pd.DataFrame.from_records(loss_and_metrics, columns=['epoch', 'Training Loss', 'Validation Loss', 'specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score', 'DSC' ])
    df.to_csv('performance_metrics.csv', index=False)      
    
    return model


#-----------------------------------------------------------------------#
#                                train_3D                               #
#              Train 3D UNet for some number of epochs                  #
#-----------------------------------------------------------------------#
def train_3D(n_epochs, loaders, model, optimizer, criterion, train_on_gpu, performance_metrics, path, metric_save_path,threshold, scheduler=None):
    #train 3D UNet for some number of epochs
    #keep track of loss and performance merics
    loss_and_metrics =[]
    # initialize tracker for max DSC 
    DSC_max = 0
    show_every = 50
    for epoch in tqdm( range(1, n_epochs+1), total = n_epochs+1):
        print(f'=== Epoch #{epoch} ===')
        # initialize variables to monitor training and validation loss, and performance metrics
        train_loss = 0.0
        valid_loss = 0.0

        specificity_val = 0
        sensitivity_val = 0
        precision_val = 0
        F1_score_val = 0
        F2_score_val = 0
        DSC_val = 0
        valid_cnt = 0
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # === (BEGIN) ===
            # 仅在第一个 batch 检查，避免刷屏
            if batch_idx == 0:
                print(f"\nDEBUG: Data Shape: {data.shape}")
                print(f"DEBUG: Target Shape: {target.shape}")
                
            if batch_idx == 0:
                print(f"\nDEBUG: Target Max Value: {target.max().item()}")
                if target.max().item() > 1:
                    print("⚠️ 警告: Mask 值大于 1 (可能是 255)，会导致 Loss 计算错误！")
                else:
                    print("✅ Mask 值正常 (0-1)。")
            # ===  (END) ===

            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            # clear the gradients of all optimized variable
            optimizer.zero_grad() 
            # forward pass (inference)
            output = model(data) 
            # calculate the batch loss
            loss = criterion(output, target) 
            # backpropagation
            loss.backward() 
            # Update weights
            optimizer.step() 

            # OneCycleLR 需要在每个 batch 结束后 step，而不是 epoch 结束
            # ReduceLROnPlateau 不能在这里运行
            # if scheduler is not None:
            #     scheduler.step()

            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
            
                         
        ######################    
        # validate the model #
        ######################
        print('=== Validation ===')
        # Set the model to inference mode
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["valid"])}...')
                # move to GPU
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass (inference) to get the output
                output = model(data)
                # calculate the batch loss
                loss = criterion (output, target)
                # update validation loss
                valid_loss +=  ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                                
                # convert output probabilities to predicted class
                output = output.cpu().detach().numpy()
                # Binarize the output
                output_b = (output>threshold)*1
                output_b = np.squeeze(output_b)
                batch_l = output_b.size
                # update the total number of validation pairs
                valid_cnt += batch_l
                # Transform output back to Pytorch Tensor and move it to GPU
                output_b = torch.as_tensor(output_b)
                output_b = output_b.cuda()
                # calculate average performance metrics per batches
                m = performance_metrics(smooth = 1e-6)
                specificity, sensitivity, precision, F1_score, F2_score, DSC =  m(output_b,target)    
                specificity_val += specificity * batch_l
                sensitivity_val += sensitivity * batch_l
                precision_val += precision * batch_l
                F1_score_val += F1_score * batch_l
                F2_score_val += F2_score * batch_l
                DSC_val += DSC * batch_l 
        # Calculate the overall average metrics
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val = specificity_val/valid_cnt, sensitivity_val/valid_cnt, precision_val/valid_cnt, F1_score_val/valid_cnt, F2_score_val/valid_cnt, DSC_val/valid_cnt

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        # ✅ 只有 ReduceLROnPlateau 需要这一行传入验证集 Los
        if scheduler is not None:
            scheduler.step(valid_loss) 
                
        print('Specificity: {:.6f} \tSensitivity: {:.6f} \tF2_score: {:.6f} \tDSC: {:.6f}'.format(
            specificity_val,
            sensitivity_val, 
            F2_score_val, 
            DSC_val
        ))
              
        
        if DSC_val > DSC_max:
            print('Validation DSC increased.  Saving model ...')            
            torch.save(model.state_dict(), path)
            DSC_max = DSC_val

        loss_and_metrics.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(), specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val))

    #save the loss_epoch as well as the performance metrics history
    df=pd.DataFrame.from_records(loss_and_metrics, columns=['epoch', 'Training Loss', 'Validation Loss', 'specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score', 'DSC' ])
    df.to_csv(metric_save_path, index=False)      
    
    return model