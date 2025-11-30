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
        # ✅ 修复：必须先过 Sigmoid 将 Logits 转为 0-1 概率
        y_pred_prob = torch.sigmoid(y_pred_flat)
        
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # calculate the Tversky index
        tversky = tp/(tp + self.alpha * fn + self.beta * fp + self.smooth)
        # return the loss
        return 1 - tversky



class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, bce_weight=0.5, pos_weight=None):
        """
        pos_weight (float): 正样本权重。建议设为 10.0 到 20.0 之间。
                            如果不传 (None)，则退化为普通的 BCE。
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_weight = bce_weight
        
        # ✅ 关键修改：处理 pos_weight
        if pos_weight is not None:
            # 创建 tensor
            pw = torch.tensor([pos_weight])
            # 这里的处理技巧：
            # 我们不在这里 .cuda()，因为初始化时可能还没上 GPU。
            # 我们把它注册为 buffer，这样它会自动跟随模型移动到 GPU。
            self.register_buffer('pos_weight', pw)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        # y_pred: Logits (无 Sigmoid) [B, 1, H, W]
        # y_true: Labels (0 或 1) [B, 1, H, W]

        # Flatten
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true).float()
        
        # --- 1. Tversky Loss (需要手动转概率) ---
        y_pred_prob = torch.sigmoid(y_pred_flat)
        
        tp = (y_pred_prob * y_true_flat).sum()
        fp = (y_pred_prob * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_prob) * y_true_flat).sum()
        
        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss_tversky = 1 - tversky_index

        # --- 2. BCE Loss (直接用 Logits) ---
        # BCEWithLogitsLoss 不需要 Flatten，它会自动处理广播，只要维度对得上
        loss_bce = self.bce(y_pred, y_true.float()) 

        # --- 3. 混合 ---
        return loss_tversky + (self.bce_weight * loss_bce)


