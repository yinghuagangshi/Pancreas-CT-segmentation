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
        
        # ✅ 区别1：使用 BCEWithLogitsLoss (自动包含 Sigmoid + BCE，数值更稳)
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, y_pred, y_true):
        # y_pred: Logits (无 Sigmoid, 范围 -inf ~ inf)
        # y_true: Labels (0 或 1)

        # Flatten
        y_pred_flat = torch.flatten(y_pred)
        # ✅ 关键修改：转 float 防止报错
        y_true_flat = torch.flatten(y_true).float()
        
        # --- 1. Tversky Loss (需要手动转概率) ---
        # ✅ 区别2：计算 Tversky 前必须手动加 Sigmoid
        y_pred_prob = torch.sigmoid(y_pred_flat)
        
        tp = (y_pred_prob * y_true_flat).sum()
        fp = (y_pred_prob * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_prob) * y_true_flat).sum()
        
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss_tversky = 1 - tversky_index

        # --- 2. BCE Loss (直接用 Logits) ---
        # ✅ 区别3：直接传 Logits 给 BCEWithLogitsLoss，不需要 clamp
        loss_bce = self.bce(y_pred, y_true.float()) # 注意这里 y_pred 不用 flatten 也可以，只要维度匹配

        # --- 3. 混合 ---
        return loss_tversky + (self.bce_weight * loss_bce)


#-----------------------------------------------------------------------#
#                       MixedLossWithLogits                             #
#         适用于模型输出没有经过 Sigmoid 激活函数的情况 (Logits)        #
#-----------------------------------------------------------------------#
class MixedLossWithLogits(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, bce_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_weight = bce_weight
        # BCEWithLogitsLoss 内部集成了 Sigmoid + BCELoss，数值更稳定
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, y_pred_logits, y_true):
        """
        y_pred_logits: 模型的原始输出 (Logits), 未经过 Sigmoid
        y_true: 真实标签 (Mask)
        """
        # 展平
        y_pred_flat_logits = torch.flatten(y_pred_logits)
        y_true_flat = torch.flatten(y_true).float()

        # --- 1. Tversky Loss (需要概率值) ---
        # 必须先手动对 logits 做 sigmoid 才能计算 Tversky
        y_pred_flat_probs = torch.sigmoid(y_pred_flat_logits)
        
        tp = (y_pred_flat_probs * y_true_flat).sum()
        fp = (y_pred_flat_probs * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat_probs) * y_true_flat).sum()
        
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss_tversky = 1 - tversky_index

        # --- 2. BCEWithLogitsLoss (直接吃 Logits) ---
        # 这里直接传入 logits，不需要手动 clamp，PyTorch 内部处理了
        loss_bce = self.bce_logits(y_pred_flat_logits, y_true_flat)

        # --- 3. 混合 ---
        return loss_tversky + (self.bce_weight * loss_bce)
