#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn

#-----------------------------------------------------------------------#
#                             TverskyLoss                               #
#                 Calculate and return the Tversky Loss                 #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# Salehi al. “Tversky Loss Function for Image Segmentation              #
# Using 3D Fully Convolutional Deep Networks.”                          #
# ArXiv abs/1706.05721 (2017)                                           #
#-----------------------------------------------------------------------#
# alpha and beta control the magnitude of penalties on false            #
# positives and false negatives, respectively.                          #
# alpha = beta = 0.5 is equilvalent to Dice coefficient                 #
# alpha = beta = 1 is equilvalent to Tanimoto index/Jaccard coefficient #
# alpha + beta = 1 leads to the set of F_beta scores                    #
# smooth: a very small number added to the denomiators to               #
#         prevent the division by zero                                  #
# tp: number of true positives                                          #
# fp: number of false positives                                         #
# fn: number of false negatives                                         #
#-----------------------------------------------------------------------#
class TverskyLoss(nn.Module):
    #returns the Tversky loss per batch
    def __init__(self, smooth = 0.000001, alpha = 0.5, beta = 0.5):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        # calculate the number of true positives, false positives and false negatives
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # calculate the Tversky index
        tversky = tp/(tp + self.alpha * fn + self.beta * fp + self.smooth)
        # return the loss
        return 1 - tversky
    

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, bce_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss() 

    def forward(self, y_pred, y_true):
        # Flatten
        y_pred_flat = torch.flatten(y_pred)
        
        # 🔥【关键修改】在这里加上 .float() 
        # 把 Int 类型的 Mask 转为 Float 类型，否则 BCE 会报错
        y_true_flat = torch.flatten(y_true).float()
        
        # --- 1. Tversky Loss ---
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss_tversky = 1 - tversky_index

        # --- 2. BCE Loss ---
        y_pred_clamped = torch.clamp(y_pred_flat, 1e-7, 1 - 1e-7)
        loss_bce = self.bce(y_pred_clamped, y_true_flat)

        # --- 3. 混合 ---
        return loss_tversky + (self.bce_weight * loss_bce)

