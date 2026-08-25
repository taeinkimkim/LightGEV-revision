from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
import gc
import time
from .feature_extraction import *
from .hourglass import *
from .loss import *


class ACVNet(nn.Module):
    def __init__(self, cfgs):
        super(ACVNet, self).__init__()
        self.maxdisp = cfgs.MAX_DISP
        self.attn_weights_only = cfgs.ATTN_WEIGHTS_ONLY
        self.freeze_attn_weights = cfgs.FREEZE_ATTN_WEIGHTS
        self.num_groups = cfgs.NUM_GROUPS  
        self.concat_channels = cfgs.CONCAT_CHANNELS  
        self.feature_extraction = feature_extraction()
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.patch = nn.Conv3d(40, 40, kernel_size=(1,3,3), stride=1, dilation=1, groups=40, padding=(0,1,1), bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=3, groups=16, padding=(0,3,3), bias=False)

        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))        
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, data):
        left = data['left']
        right = data['right']

        if self.freeze_attn_weights:
            with torch.no_grad():
                features_left = self.feature_extraction(left)
                features_right = self.feature_extraction(right)
                gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
                gwc_volume = self.patch(gwc_volume)
                patch_l1 = self.patch_l1(gwc_volume[:, :8])
                patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
                patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
                patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)
                cost_attention = self.dres1_att_(patch_volume)
                cost_attention = self.dres2_att_(cost_attention)
                att_weights = self.classif_att_(cost_attention)

        else:

            features_left = self.feature_extraction(left)
            features_right = self.feature_extraction(right)
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
            gwc_volume = self.patch(gwc_volume)
            patch_l1 = self.patch_l1(gwc_volume[:, :8])
            patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
            patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
            patch_volume = torch.cat((patch_l1,patch_l2,patch_l3), dim=1)
            cost_attention = self.dres1_att_(patch_volume)
            cost_attention = self.dres2_att_(cost_attention)
            att_weights = self.classif_att_(cost_attention)

        if not self.attn_weights_only:
            concat_feature_left = self.concatconv(features_left["gwc_feature"])
            concat_feature_right = self.concatconv(features_right["gwc_feature"])  
            concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)
            ac_volume = F.softmax(att_weights, dim=2) * concat_volume   ### ac_volume = att_weights * concat_volume 
            cost0 = self.dres0(ac_volume)
            cost0 = self.dres1(cost0) + cost0
            out1 = self.dres2(cost0)
            out2 = self.dres3(out1)

        if self.training:

            if not self.freeze_attn_weights:

                cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = torch.squeeze(cost_attention, 1)
                pred_attention = F.softmax(cost_attention, dim=1)
                pred_attention = disparity_regression(pred_attention, self.maxdisp)

            if not self.attn_weights_only:

                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)    

                cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost0 = torch.squeeze(cost0, 1)
                pred0 = F.softmax(cost0, dim=1)
                pred0 = disparity_regression(pred0, self.maxdisp)    

                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparity_regression(pred1, self.maxdisp)    

                cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparity_regression(pred2, self.maxdisp)

                if self.freeze_attn_weights:
                    disp_preds = [pred0, pred1, pred2]
                    return {'disp_preds': disp_preds,
                            'disp_pred': disp_preds[-1]}
                
                disp_preds = [pred_attention, pred0, pred1, pred2]
                return {'disp_preds': disp_preds,
                        'disp_pred': disp_preds[-1]}
            
            disp_preds = [pred_attention]
            return {'disp_preds': disp_preds,
                    'disp_pred': disp_preds[-1]}

        else:

            if self.attn_weights_only:

                cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost_attention = torch.squeeze(cost_attention, 1)
                pred_attention = F.softmax(cost_attention, dim=1)
                pred_attention = disparity_regression(pred_attention, self.maxdisp)
                return [pred_attention]

            cost2 = self.classif2(out2)
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return {'disp_pred': pred2}
        
    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]
        mask = (disp_gt < self.maxdisp) & (disp_gt > 0)
        disp_pred = model_preds['disp_pred']

        loss = 0.0

        if self.attention_weights_only:
            loss = model_loss_train_attn_only(disp_pred, disp_gt, mask)
        elif self.freeze_attention_weights:
            loss = model_loss_train_freeze_attn(disp_pred, disp_gt, mask)
        else:
            loss = model_loss_train(disp_pred, disp_gt, mask)

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info

def acv(d):
    return ACVNet(d)
