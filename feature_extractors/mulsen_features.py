import torch
import torch.nn.functional as F
from feature_extractors.features import Features
from utils_loc.mvtec3d_util import *
import numpy as np
import math
import os
import random
from tqdm import tqdm
from utils_loc.cpu_knn import fill_missing_values

try:
    from models.pointnet2_utils import farthest_point_sample
except Exception:
    farthest_point_sample = None



# -------------------------------------------------------------------------
# PC token -> 2D feature map helpers (M3DM-style soft splatting)
# -------------------------------------------------------------------------
def normalize_xy_to_grid(centers, eps=1e-6):
    """
    centers: [B, G, 3]
    return:  [B, G, 2] in [-1, 1]
    """
    xy = centers[..., :2]
    xy_min = xy.amin(dim=1, keepdim=True)
    xy_max = xy.amax(dim=1, keepdim=True)
    xy_norm = (xy - xy_min) / (xy_max - xy_min + eps)
    return xy_norm * 2.0 - 1.0


def splat_tokens_to_2d(tokens, centers, out_h=28, out_w=28, eps=1e-6):
    """
    Soft-splat point-group tokens onto a regular 2D grid using x-y coordinates.

    tokens:  [B, G, C]
    centers: [B, G, 3]
    return:
        feat_2d: [B, C, out_h, out_w]
        weight:  [B, 1, out_h, out_w]
    """
    B, G, C = tokens.shape
    device, dtype = tokens.device, tokens.dtype

    xy = normalize_xy_to_grid(centers)
    x = (xy[..., 0] + 1.0) * 0.5 * (out_w - 1)
    y = (xy[..., 1] + 1.0) * 0.5 * (out_h - 1)

    x0 = torch.floor(x).long().clamp(0, out_w - 1)
    y0 = torch.floor(y).long().clamp(0, out_h - 1)
    x1 = (x0 + 1).clamp(0, out_w - 1)
    y1 = (y0 + 1).clamp(0, out_h - 1)

    wx = x - x0.float()
    wy = y - y0.float()
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    feat = torch.zeros(B, C, out_h * out_w, device=device, dtype=dtype)
    weight = torch.zeros(B, 1, out_h * out_w, device=device, dtype=dtype)

    def scatter(ix, iy, w):
        idx = iy * out_w + ix  # [B, G]
        idx_feat = idx.unsqueeze(1).expand(B, C, G)
        val = tokens.permute(0, 2, 1) * w.unsqueeze(1)
        feat.scatter_add_(2, idx_feat, val)
        weight.scatter_add_(2, idx.unsqueeze(1), w.unsqueeze(1))

    scatter(x0, y0, w00)
    scatter(x0, y1, w01)
    scatter(x1, y0, w10)
    scatter(x1, y1, w11)

    feat = feat / (weight + eps)
    return feat.view(B, C, out_h, out_w), weight.view(B, 1, out_h, out_w)


def sample_2d_score_to_points(score_2d, pts, eps=1e-6):
    """
    Project a 2D PC anomaly score map back to original point-level scores.

    score_2d: [1, 1, H, W], [1, H, W], or [H, W]
    pts:      [B, N, 3], [B, 3, N], [N, 3], or [3, N]
    return:   numpy [N]
    """
    if score_2d.dim() == 2:
        score_2d = score_2d[None, None]
    elif score_2d.dim() == 3:
        score_2d = score_2d[:, None]

    if not torch.is_floating_point(score_2d):
        score_2d = score_2d.float()

    device = score_2d.device
    pts = pts.to(device=device, dtype=score_2d.dtype)

    # Be tolerant to cached point layouts. MulSen-AD normally uses [B, N, 3],
    # but cached tensors may appear as [N, 3], [3, N], or [B, 3, N].
    if pts.dim() == 2:
        if pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.transpose(0, 1)
        pts = pts.unsqueeze(0)
    elif pts.dim() == 3:
        if pts.shape[1] == 3 and pts.shape[2] != 3:
            pts = pts.transpose(1, 2)

    xy = pts[..., :2]
    xy_min = xy.amin(dim=1, keepdim=True)
    xy_max = xy.amax(dim=1, keepdim=True)
    xy_norm = (xy - xy_min) / (xy_max - xy_min + eps)
    xy_norm = xy_norm * 2.0 - 1.0
    grid = xy_norm.unsqueeze(2)  # [B, N, 1, 2]

    sampled = F.grid_sample(
        score_2d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)[0].detach().cpu().numpy()
##Single modality
class RGBFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        self.patch_rgb_lib.append(rgb_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)



   
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
  

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
      
        

        
        s = torch.tensor([s_rgb])
        


        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

    
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
 
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
  

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')


        s = torch.tensor([s_rgb])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra

        ## PC
     
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
     
        
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0) 

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)))) 
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
      
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
      

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)


        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

            
class InfraFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([s_infra])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_infra])
 
        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
         # pixel-level preds  or labels
        ## RGB


        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())
        ## PC

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):

        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib)

        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  

class PCFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T  
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.patch_xyz_lib.append(xyz_patch)

        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask,center_idx,pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
    
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')

             
        s = torch.tensor([s_xyz])
        
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask,center_idx,pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')


        s = torch.tensor([s_xyz])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        # pixel-level preds  or labels
        ## RGB


        ## Infra

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx,pts, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        if modal=='xyz':
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map


        

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)      
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std


        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
 


## Decision level fusion
class PCRGBGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

        
    def predict(self, sample, label,pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
   
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
 



        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx, pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
    
        

        
        s = torch.tensor([[s_xyz, s_rgb]])

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

      
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
       
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, s_rgb]])
        s = torch.tensor(self.detect_fuser.score_samples(s))

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

        #-------------------------------------------------------------
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

   
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class PCInfraGatingFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _= self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        

        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_infra_lib.append(infra_patch)
        

        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

            
        

        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

    
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        
        
        s = torch.tensor([[s_xyz, s_infra]])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

    
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
       
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_infra]])
 

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB

        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000


        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            # segmentation map
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)

        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_infra_lib)

        self.infra_mean = torch.mean(self.patch_xyz_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std


        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]

class RGBInfraGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], sample[1], sample[2])
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)

        
    def predict(self, sample, label, pixel_mask):
 
        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
 

        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
   
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
          
        s = torch.tensor([[s_rgb, s_infra]])

        self.s_lib.append(s)
 

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_rgb, s_infra]])

        s = torch.tensor(self.detect_fuser.score_samples(s))
        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        ## PC
    
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000


        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            # segmentation map
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        

        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_infra_lib)
        self.infra_mean = torch.mean(self.patch_rgb_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]



class TripleRGBInfraPointFeatures(Features):
    """
    Three independent memory banks version:
      - RGB memory bank
      - Infra memory bank
      - PC memory bank

    Difference from the original MulSen-AD implementation:
      - Point-cloud group tokens are first projected to a 28x28 2D feature map
        by x-y soft splatting, then converted to [28*28, C] patch tokens.
      - RGB / Infra / PC keep three independent memory banks and coreset steps.
      - Object score uses the original late-fusion OCSVM over [PC, RGB, Infra] scores
        when memory_bank='multiple'.
    """

    def _xyz_feature_maps_to_2d_patch(self, xyz_feature_maps, center, out_h=28, out_w=28):
        xyz_tokens = torch.cat(xyz_feature_maps, 1).to(self.device)  # [B, C, G]
        xyz_tokens = xyz_tokens.permute(0, 2, 1).contiguous()        # [B, G, C]
        xyz_2d_map, _ = splat_tokens_to_2d(
            tokens=xyz_tokens,
            centers=center.to(self.device),
            out_h=out_h,
            out_w=out_w,
        )  # [B, C, out_h, out_w]
        xyz_patch = xyz_2d_map.reshape(xyz_2d_map.shape[1], -1).T.contiguous()  # [H*W, C]
        return xyz_patch

    def _rgb_patch(self, rgb_feature_maps):
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        return rgb_patch.reshape(rgb_patch.shape[1], -1).T

    def _infra_patch(self, infra_feature_maps):
        infra_patch = torch.cat(infra_feature_maps, 1)
        return infra_patch.reshape(infra_patch.shape[1], -1).T

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(
            sample[0], sample[1], sample[2]
        )

        xyz_patch = self._xyz_feature_maps_to_2d_patch(xyz_feature_maps, center, out_h=28, out_w=28).detach().cpu()
        rgb_patch = self._rgb_patch(rgb_feature_maps).detach().cpu()
        infra_patch = self._infra_patch(infra_feature_maps).detach().cpu()

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)

    def predict(self, sample, label, pixel_mask):
        if label[0] == 1 or label[1] == 1 or label[2] == 1:
            label_s = 1
        else:
            label_s = 0

        rgb = sample[0].to(self.device)
        infra = sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(
            rgb, infra, pc
        )

        xyz_patch = self._xyz_feature_maps_to_2d_patch(xyz_feature_maps, center, out_h=28, out_w=28)
        rgb_patch = self._rgb_patch(rgb_feature_maps).to(self.device)
        infra_patch = self._infra_patch(infra_feature_maps).to(self.device)

        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        rgb = sample[0].to(self.device)
        infra = sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(
            rgb, infra, pc
        )

        xyz_patch = self._xyz_feature_maps_to_2d_patch(xyz_feature_maps, center, out_h=28, out_w=28)
        rgb_patch = self._rgb_patch(rgb_feature_maps).to(self.device)
        infra_patch = self._infra_patch(infra_feature_maps).to(self.device)

        xyz_patch = ((xyz_patch - self.xyz_mean) / self.xyz_std).detach().cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean) / self.rgb_std).detach().cpu()
        infra_patch = ((infra_patch - self.infra_mean) / self.infra_std).detach().cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_xyz, _ = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, center_idx, pts, modal='xyz_2d')
        s_rgb, _ = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, _ = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[float(s_xyz), float(s_rgb), float(s_infra)]])
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        xyz_patch = ((xyz_patch - self.xyz_mean) / self.xyz_std).detach().cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean) / self.rgb_std).detach().cpu()
        infra_patch = ((infra_patch - self.infra_mean) / self.infra_std).detach().cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_xyz, s_map_xyz_2d = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, center_idx, pts, modal='xyz_2d')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        # Object-level decision. Keep the old OCSVM fusion if trained; otherwise use max of 3 banks.
        score_vec = torch.tensor([[float(s_xyz), float(s_rgb), float(s_infra)]])
        if self.args.memory_bank == 'multiple' and hasattr(self.detect_fuser, 'coef_'):
            s = torch.tensor(self.detect_fuser.score_samples(score_vec))
        else:
            s = score_vec.max(dim=1).values

        # Object-level preds / labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # RGB pixel metrics
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        # Infra pixel metrics
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        # PC pixel metrics: sample the 2D PC score map back to original points.
        s_map_xyz_points = sample_2d_score_to_points(s_map_xyz_2d.to(self.device), pts)
        self.pc_pixel_preds.extend(s_map_xyz_points.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        self.pc_pts.append(pts[0, :])
        self.pc_predictions.append(s_map_xyz_points.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz_2d'):
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000

        m_test = patch[s_idx].unsqueeze(0)

        if modal in ['xyz', 'xyz_2d']:
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)
        elif modal == 'rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_infra_lib)

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)

        if modal in ['xyz', 'xyz_2d']:
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) / 1000
        elif modal == 'rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1) / 1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1], dtype=patch.dtype))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-8))
        s = w * s_star

        # For xyz_2d/RGB/Infra, make a 2D map and upsample to 224.
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear', align_corners=True)
        s_map = self.blur(s_map)
        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0).detach().cpu()
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0).detach().cpu()
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0).detach().cpu()

        # Normalize each memory bank independently.
        eps = 1e-8
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib) + eps
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib) + eps
        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + eps

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean) / self.infra_std

        if self.f_coreset < 1:
            n_xyz = max(1, int(self.f_coreset * self.patch_xyz_lib.shape[0]))
            n_rgb = max(1, int(self.f_coreset * self.patch_rgb_lib.shape[0]))
            n_infra = max(1, int(self.f_coreset * self.patch_infra_lib.shape[0]))

            coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib, n=n_xyz, eps=self.coreset_eps)
            self.patch_xyz_lib = self.patch_xyz_lib[coreset_idx]

            coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib, n=n_rgb, eps=self.coreset_eps)
            self.patch_rgb_lib = self.patch_rgb_lib[coreset_idx]

            coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib, n=n_infra, eps=self.coreset_eps)
            self.patch_infra_lib = self.patch_infra_lib[coreset_idx]


class TripleRGBInfraPointFusionMemoryFeatures(TripleRGBInfraPointFeatures):
    """
    MulSen-AD three-memory-bank baseline plus one learned fusion memory bank.

    Pipeline:
      1) Extract frozen RGB / Infra / PC-to-2D patch features once and cache them.
      2) Pretrain a tri-modal high-order contrastive fusion block from the cached patches.
      3) Build RGB / Infra / PC / Fusion memory banks from cached patches.
      4) Evaluate with MulSen-AD's original metric protocol.
    """

    def __init__(self, args):
        super().__init__(args)
        from models.feature_fusion_high_order import TriModalHighOrderFusionBlock

        self.fusion_embed_dim = int(getattr(args, 'fusion_embed_dim', 512))
        self.fusion_temperature = float(getattr(args, 'fusion_temperature', 0.07))
        self.fusion_pretrained = False
        self.fusion_block = TriModalHighOrderFusionBlock(
            xyz_dim=1152,
            rgb_dim=768,
            infra_dim=768,
            embed_dim=self.fusion_embed_dim,
            temperature=self.fusion_temperature,
            dropout=float(getattr(args, 'fusion_dropout', 0.0)),
            compactness_weight=float(getattr(args, 'fusion_compactness_weight', 0.1)),
            centroid_weight=float(getattr(args, 'fusion_centroid_weight', 0.25)),
            # Kept for backward compatibility. The current fusion block ignores DP.
            use_dp=False,
            dp_alpha=float(getattr(args, 'fusion_dp_alpha', 1.0)),
            dp_lambda=float(getattr(args, 'fusion_dp_lambda', 0.01)),
            num_prototypes=int(getattr(args, 'fusion_dp_num_prototypes', 64)),
        ).to(self.device)

        ckpt_path = getattr(args, 'fusion_module_path', '')
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_fusion_block(ckpt_path)

        # Object-level score buffers for single banks and bank combinations.
        # These are evaluated in calculate_metrics() as extra sample-level AUROC columns.
        self.object_score_dict = {
            'PC': [],
            'RGB': [],
            'Infra': [],
            'Fusion': [],
            'PC+RGB': [],
            'PC+Infra': [],
            'RGB+Infra': [],
            'PC+Fusion': [],
            'RGB+Fusion': [],
            'Infra+Fusion': [],
            'PC+RGB+Fusion': [],
            'PC+Infra+Fusion': [],
            'RGB+Infra+Fusion': [],
            'PC+RGB+Infra': [],
            'PC+RGB+Infra+Fusion': [],
        }

        # Pixel-level buffers for learned fusion memory and D_s late-fusion maps.
        self.fusion_pixel_preds = []
        self.fusion_pixel_labels = []
        self.fusion_predictions = []
        self.fusion_gts = []
        self.m3dm_dlf_pixel_preds = []
        self.m3dm_dlf_pixel_labels = []
        self.m3dm_dlf_predictions = []
        self.m3dm_dlf_gts = []


        # Pixel-level score buffers for all requested modality combinations.
        # 2D-only combinations are evaluated on the 224 x 224 image grid.
        # Combinations containing PC are evaluated on the point grid, with
        # RGB/Infra/Fusion 2D score maps sampled back to points using the same
        # x-y projection used for PC score maps.
        self.pixel_combo_raw = {
            'RGB+Infra': {'preds': [], 'labels': []},
            'RGB+Fusion': {'preds': [], 'labels': []},
            'Infra+Fusion': {'preds': [], 'labels': []},
            'RGB+Infra+Fusion': {'preds': [], 'labels': []},
            'PC+RGB': {'preds': [], 'labels': []},
            'PC+Infra': {'preds': [], 'labels': []},
            'PC+Fusion': {'preds': [], 'labels': []},
            'PC+RGB+Infra': {'preds': [], 'labels': []},
            'PC+RGB+Fusion': {'preds': [], 'labels': []},
            'PC+Infra+Fusion': {'preds': [], 'labels': []},
            'PC+RGB+Infra+Fusion': {'preds': [], 'labels': []},
        }
        self.pixel_combo_metrics = {}

    # ------------------------------------------------------------------
    # Checkpoints and feature cache
    # ------------------------------------------------------------------
    def _fusion_checkpoint_path(self, class_name=None):
        """
        Fusion checkpoint path.

        By default we use ONE global checkpoint trained with all selected classes'
        normal training caches. This is the intended M3DM-style fusion-memory
        workflow:

            global fusion block = trained once on all class train caches
            memory banks         = built per class
            metrics              = computed per class

        If --no-fusion_global_pretrain is passed, this falls back to the old
        per-class checkpoint naming.
        """
        explicit = getattr(self.args, 'fusion_module_path', '')
        if explicit:
            return explicit

        base = getattr(self.args, 'fusion_pretrain_dir', '') or os.path.join(self.args.output_dir, 'fusion_pretrain')
        os.makedirs(base, exist_ok=True)

        if bool(getattr(self.args, 'fusion_global_pretrain', True)):
            name = getattr(self.args, 'fusion_global_ckpt_name', 'global_tri_modal_high_order_fusion.pth')
            return os.path.join(base, name)

        cname = class_name or 'default'
        return os.path.join(base, f'{cname}_tri_modal_high_order_fusion.pth')

    def _feature_cache_root(self, class_name, split):
        base = getattr(self.args, 'fusion_feature_cache_dir', '') or './cache/fusion_high_order_features'
        use_multiscale = bool(getattr(self.args, 'use_multiscale', True))
        scale_tag = 'ms3' if use_multiscale else 'ss1'
        key = (
            f"img{getattr(self.args, 'img_size', 224)}_"
            f"ng{getattr(self.args, 'num_group', 1024)}_"
            f"gs{getattr(self.args, 'group_size', 128)}_"
            f"rgb{getattr(self.args, 'rgb_backbone_name', 'rgb')}_"
            f"xyz{getattr(self.args, 'xyz_backbone_name', 'xyz')}_"
            f"{scale_tag}"
        )
        return os.path.join(base, class_name, split, key)

    def _feature_cache_index_path(self, class_name, split):
        return os.path.join(self._feature_cache_root(class_name, split), 'index.pt')

    def load_fusion_block(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        self.fusion_block.load_state_dict(state, strict=False)
        self.fusion_block.to(self.device).eval()
        self.fusion_pretrained = True
        print(f'Loaded tri-modal high-order fusion block: {ckpt_path}')

    def save_fusion_block(self, ckpt_path, epoch=None):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({'model': self.fusion_block.state_dict(), 'epoch': epoch}, ckpt_path)
        print(f'Saved tri-modal high-order fusion block: {ckpt_path}')

    def _reduce_multiscale_patch_for_bank(self, patch):
        """
        Convert cached multi-scale patch features to single-scale memory-bank features.

        Contrastive pretraining must keep the full input-level multi-scale tensor:
            [num_patches, 3, dim]

        But PatchCore/MulSen-AD memory banks, coreset, cdist, OCSVM/DLF, and
        score-map generation expect one feature vector per spatial patch:
            [num_patches, dim]

        Therefore, only the memory-bank / scoring path uses the scale mean.
        The cached tensors are left unchanged, and fusion_block.encode(...) still
        receives [N,3,D] during contrastive pretraining and fusion feature creation.
        """
        if not torch.is_tensor(patch):
            patch = torch.as_tensor(patch)

        if patch.dim() == 3:
            # Preferred cache format: [N, S, D], where S=3.
            if patch.shape[1] in (2, 3, 4):
                return patch.float().mean(dim=1)
            # Also tolerate [S, N, D].
            if patch.shape[0] in (2, 3, 4):
                return patch.float().mean(dim=0)
            raise ValueError(
                f'Cannot reduce ambiguous multi-scale patch shape {tuple(patch.shape)}. '
                'Expected [N,S,D] or [S,N,D] with S in {2,3,4}.'
            )

        if patch.dim() == 2:
            return patch.float()

        raise ValueError(f'Expected patch [N,D] or [N,S,D], got {tuple(patch.shape)}')

    def _make_image_input_scales(self, image):
        """
        Build input-level image scales before feature extraction.

        image: [B, C, H, W]
        return:
          full image
          image downsampled to 1/2 then upsampled back to H x W
          image downsampled to 1/4 then upsampled back to H x W

        This follows the requested pipeline:
          [224,224,3] -> [112,112,3] -> [224,224,3] -> backbone
          [224,224,3] -> [56,56,3]   -> [224,224,3] -> backbone
        """
        if image.dim() != 4:
            raise ValueError(f'Expected image tensor [B,C,H,W], got {tuple(image.shape)}')

        _, _, h, w = image.shape
        scales = [image]
        for factor in (2, 4):
            dh = max(1, h // factor)
            dw = max(1, w // factor)
            down = F.interpolate(image, size=(dh, dw), mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
            scales.append(up)
        return scales

    def _normalize_pointcloud_layout(self, pts):
        """Return point cloud as [B, N, 3]."""
        if not torch.is_tensor(pts):
            pts = torch.as_tensor(pts)
        pts = pts.to(self.device).float()

        if pts.dim() == 2:
            # [N,3] or [3,N]
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.t().contiguous()
            pts = pts.unsqueeze(0)
        elif pts.dim() == 3:
            # [B,N,3] or [B,3,N]
            if pts.shape[1] == 3 and pts.shape[2] != 3:
                pts = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f'Expected point cloud [N,3], [3,N], [B,N,3], or [B,3,N], got {tuple(pts.shape)}')

        if pts.shape[-1] != 3:
            raise ValueError(f'Point cloud last dimension must be 3, got {tuple(pts.shape)}')
        return pts

    def _to_point_model_layout(self, pts):
        """
        Return point cloud in the layout expected by the Point-MAE backbone: [B, 3, N].

        Internally, helper functions use [B, N, 3] because FPS/KNN style utilities
        usually operate on this layout. However, this project's Point-MAE wrapper
        expects [B, 3, N] and transposes it inside the backbone. Passing [B, N, 3]
        directly will make the backbone treat N as the channel dimension and causes
        the huge invalid reshape seen in group_divider.
        """
        pts = self._normalize_pointcloud_layout(pts)  # [B, N, 3]
        return pts.transpose(1, 2).contiguous()       # [B, 3, N]

    def _fps_pointcloud(self, pts, ratio):
        """
        FPS downsample point cloud at input level.

        pts: [B, N, 3], [B, 3, N], [N, 3], or [3, N]
        ratio: 0.5 or 0.25
        return: [B, max(1, int(N*ratio)), 3]

        It uses the project-provided farthest_point_sample implementation when
        available. If FPS is unavailable, it falls back to deterministic strided
        sampling. The output deliberately stays [B, M, 3]; conversion to the
        Point-MAE layout [B, 3, M] happens right before calling the backbone.
        """
        pts = self._normalize_pointcloud_layout(pts)
        b, n, _ = pts.shape
        n_sample = max(1, int(n * float(ratio)))

        if farthest_point_sample is not None:
            idx = farthest_point_sample(pts.contiguous(), n_sample)
            # Some FPS helpers return (idx, extra) or sampled coordinates. Keep only indices.
            if isinstance(idx, tuple):
                idx = idx[0]
            # Expected index shape: [B, n_sample]. If an implementation returns
            # sampled coordinates [B, n_sample, 3], fall back to strided sampling
            # because those cannot be used as integer indices safely.
            if not torch.is_tensor(idx) or idx.dim() != 2:
                base = torch.linspace(0, n - 1, steps=n_sample, device=pts.device).long()
                idx = base.unsqueeze(0).repeat(b, 1)
            else:
                idx = idx.long().to(pts.device)
        else:
            base = torch.linspace(0, n - 1, steps=n_sample, device=pts.device).long()
            idx = base.unsqueeze(0).repeat(b, 1)

        idx = idx.clamp_(0, n - 1)
        batch_idx = torch.arange(b, device=pts.device).view(b, 1).expand_as(idx)
        return pts[batch_idx, idx, :].contiguous()

    def _make_point_input_scales(self, pointcloud):
        """
        Build input-level point-cloud scales before point feature extraction.

        pointcloud: [B,N,3], [B,3,N], [N,3], or [3,N]
        return:
          original point cloud
          FPS to N/2
          FPS to N/4

        No interpolation back to N is performed. Each scale is passed directly
        into the point backbone, then its tokens are splatted to the aligned
        28 x 28 grid afterwards.
        """
        pts = self._normalize_pointcloud_layout(pointcloud)
        return [pts, self._fps_pointcloud(pts, 0.5), self._fps_pointcloud(pts, 0.25)]

    def _extract_aligned_patches_single_scale(self, rgb, infra, pointcloud):
        # Point-MAE in this project expects [B, 3, N], while the multi-scale
        # point-cloud helpers keep points as [B, N, 3]. Convert here and only here.
        pointcloud_for_backbone = self._to_point_model_layout(pointcloud)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(
            rgb, infra, pointcloud_for_backbone
        )
        xyz_patch = self._xyz_feature_maps_to_2d_patch(xyz_feature_maps, center, out_h=28, out_w=28).to(self.device)
        rgb_patch = self._rgb_patch(rgb_feature_maps).to(self.device)
        infra_patch = self._infra_patch(infra_feature_maps).to(self.device)
        return xyz_patch, rgb_patch, infra_patch, pts

    def _extract_aligned_patches(self, sample):
        """
        Extract input-level multi-scale aligned patch features.

        RGB / Infra:
          full image -> backbone -> [784, C]
          1/2 input downsample then upsample -> backbone -> [784, C]
          1/4 input downsample then upsample -> backbone -> [784, C]

        Point cloud:
          full point cloud -> point backbone -> splat -> [784, C]
          FPS N/2 -> point backbone -> splat -> [784, C]
          FPS N/4 -> point backbone -> splat -> [784, C]

        Cached output layout:
          xyz_patch:   [784, 3, 1152]
          rgb_patch:   [784, 3, 768]
          infra_patch: [784, 3, 768]
        """
        rgb, infra, pointcloud = sample[0], sample[1], sample[2]
        use_multiscale = bool(getattr(self.args, 'use_multiscale', True))

        # Switch OFF: exactly one original input scale.
        # Returned patches are [N, D]. Fusion pretraining still works, but
        # intra-modal compactness becomes zero because there is only one scale.
        if not use_multiscale:
            xyz_patch, rgb_patch, infra_patch, pts = self._extract_aligned_patches_single_scale(
                rgb.to(self.device, non_blocking=True),
                infra.to(self.device, non_blocking=True),
                pointcloud,
            )
            return xyz_patch, rgb_patch, infra_patch, pts

        # Switch ON: input-level multi-scale extraction.
        # RGB/Infra: 224 -> 112/56 -> 224, then backbone.
        # Point cloud: N -> N/2/N/4 by FPS, then point backbone directly.
        rgb_scales = self._make_image_input_scales(rgb.to(self.device, non_blocking=True))
        infra_scales = self._make_image_input_scales(infra.to(self.device, non_blocking=True))
        pc_scales = self._make_point_input_scales(pointcloud)

        xyz_patches = []
        rgb_patches = []
        infra_patches = []
        pts_full = None

        for scale_idx in range(3):
            xyz_patch, rgb_patch, infra_patch, pts = self._extract_aligned_patches_single_scale(
                rgb_scales[scale_idx],
                infra_scales[scale_idx],
                pc_scales[scale_idx],
            )
            xyz_patches.append(xyz_patch)
            rgb_patches.append(rgb_patch)
            infra_patches.append(infra_patch)
            if scale_idx == 0:
                pts_full = pts

        # [S, N, D] -> [N, S, D], where S=3: full, half, quarter.
        # This is the tensor that enters contrastive pretraining.
        xyz_patch = torch.stack(xyz_patches, dim=0).permute(1, 0, 2).contiguous()
        rgb_patch = torch.stack(rgb_patches, dim=0).permute(1, 0, 2).contiguous()
        infra_patch = torch.stack(infra_patches, dim=0).permute(1, 0, 2).contiguous()

        return xyz_patch, rgb_patch, infra_patch, pts_full

    def build_feature_cache(self, data_loader, class_name, split='train'):
        """Extract frozen features once and save a tensor dataset to disk."""
        root = self._feature_cache_root(class_name, split)
        index_path = self._feature_cache_index_path(class_name, split)
        force = bool(getattr(self.args, 'fusion_force_recache', False))

        if os.path.exists(index_path) and not force:
            index = torch.load(index_path, map_location='cpu')
            print(f'Loaded cached {split} fusion features: {index_path} ({len(index["files"])} samples)')
            return index

        os.makedirs(root, exist_ok=True)
        files = []
        self.deep_feature_extractor.eval()
        for p in self.deep_feature_extractor.parameters():
            p.requires_grad_(False)

        pbar = tqdm(data_loader, desc=f'Caching {split} fusion features for {class_name}')
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                if split == 'train':
                    sample, _ = batch
                    label = None
                    pixel_mask = None
                    paths = None
                else:
                    sample, label, pixel_mask, paths = batch

                xyz_patch, rgb_patch, infra_patch, pts = self._extract_aligned_patches(sample)
                item = {
                    'xyz_patch': xyz_patch.detach().cpu(),
                    'rgb_patch': rgb_patch.detach().cpu(),
                    'infra_patch': infra_patch.detach().cpu(),
                    'pts': pts.detach().cpu() if torch.is_tensor(pts) else pts,
                }
                if label is not None:
                    item['label'] = label.detach().cpu() if torch.is_tensor(label) else label
                if pixel_mask is not None:
                    item['pixel_mask'] = [m.detach().cpu() for m in pixel_mask]
                if paths is not None:
                    item['paths'] = paths

                path = os.path.join(root, f'{i:06d}.pt')
                torch.save(item, path)
                files.append(path)

        index = {
            'files': files,
            'class_name': class_name,
            'split': split,
            'cache_root': root,
            'format': 'MulSenADTriModalPatchCacheV1',
        }
        torch.save(index, index_path)
        print(f'Saved cached {split} fusion features: {index_path} ({len(files)} samples)')
        return index

    def iter_feature_cache(self, cache_index, shuffle=False):
        files = list(cache_index["files"])
        if shuffle:
            random.shuffle(files)
    
        # 根据 use_multiscale 选择后缀
        if bool(getattr(self.args, "use_multiscale", True)):
            suffix = "_ms3"
        else:
            suffix = "_ss1"
    
        # 取 index.pt 所在目录，如果存在，则优先用它
        cache_root = cache_index.get("cache_root", None)
        if cache_root is None:
            cache_root = os.path.dirname(files[0])  # fallback
    
        for path in files:
            real_path = path
    
            # 1. 先尝试原路径
            if not os.path.exists(real_path):
                # 2. 尝试用 cache_root + 文件名
                candidate = os.path.join(cache_root, os.path.basename(path))
                if os.path.exists(candidate):
                    real_path = candidate
    
            # 3. 尝试加后缀 (_ms3 或 _ss1)
            if not os.path.exists(real_path):
                dirname = os.path.dirname(path)
                basename = os.path.basename(path)
                candidate = os.path.join(dirname + suffix, basename)
                if os.path.exists(candidate):
                    real_path = candidate
    
            # 4. 兜底报错
            if not os.path.exists(real_path):
                raise FileNotFoundError(
                    f"Cached feature file not found.\n"
                    f"Stored path in index.pt: {path}\n"
                    f"Resolved real_path: {real_path}\n"
                    f"Suffix used: {suffix}"
                )
    
            yield torch.load(real_path, map_location="cpu")

    # ------------------------------------------------------------------
    # High-order tri-modal fusion pretraining from cached features
    # ------------------------------------------------------------------
    def _sample_cached_patches_for_pretrain(self, item, max_patches):
        """Load one cached sample and optionally subsample aligned patches."""
        xyz_patch = item['xyz_patch'].to(self.device, non_blocking=True)
        rgb_patch = item['rgb_patch'].to(self.device, non_blocking=True)
        infra_patch = item['infra_patch'].to(self.device, non_blocking=True)

        n = xyz_patch.shape[0]
        if max_patches > 0 and n > max_patches:
            idx = torch.randperm(n, device=self.device)[:max_patches]
            xyz_patch = xyz_patch[idx]
            rgb_patch = rgb_patch[idx]
            infra_patch = infra_patch[idx]

        return xyz_patch, rgb_patch, infra_patch

    def ensure_fusion_block_ready(self, train_cache=None, class_name=None):
        """
        Load the global fusion checkpoint if it exists.

        Normal fit() should not accidentally train a separate class-specific
        fusion block when the intended workflow is global pretraining. If the
        checkpoint is missing and fusion_pretrain_epochs > 0, we keep a fallback
        path for debugging/single-class runs.
        """
        ckpt_path = self._fusion_checkpoint_path(class_name)
        if os.path.exists(ckpt_path):
            self.load_fusion_block(ckpt_path)
            return

        epochs = int(getattr(self.args, 'fusion_pretrain_epochs', 0))
        if epochs <= 0:
            print(f'[FusionMemory] Fusion checkpoint not found and pretraining disabled: {ckpt_path}')
            return

        if bool(getattr(self.args, 'fusion_global_pretrain', True)):
            raise FileNotFoundError(
                f'Global fusion checkpoint not found: {ckpt_path}\n'
                f'Run --fusion_pretrain_only first, or pass --fusion_module_path.'
            )

        if train_cache is None:
            raise RuntimeError('train_cache is required for fallback per-class fusion pretraining.')
        self.pretrain_fusion_block_from_cache(train_cache, class_name=class_name)

    def pretrain_fusion_block_from_caches(self, train_caches, class_names=None):
        """
        Pretrain ONE global tri-modal high-order fusion block from multiple
        classes' cached training features.

        This is the intended workflow: all selected classes contribute normal
        training patches to the same contrastive pretraining stage.
        """
        epochs = int(getattr(self.args, 'fusion_pretrain_epochs', 0))
        if epochs <= 0:
            print('[FusionMemory] fusion_pretrain_epochs <= 0; skip global fusion pretraining.')
            return

        ckpt_path = self._fusion_checkpoint_path('global')
        if os.path.exists(ckpt_path) and not bool(getattr(self.args, 'fusion_force_pretrain', False)):
            self.load_fusion_block(ckpt_path)
            return

        if not train_caches:
            raise ValueError('train_caches is empty; cannot pretrain global fusion block.')

        lr = float(getattr(self.args, 'fusion_pretrain_lr', 1e-4))
        wd = float(getattr(self.args, 'fusion_pretrain_weight_decay', 1e-5))
        max_patches = int(getattr(self.args, 'fusion_pretrain_max_patches', 512))

        optimizer = torch.optim.AdamW(
            self.fusion_block.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.95),
        )

        # Flatten all cache files into one global training list.
        all_files = []
        for cache in train_caches:
            cname = cache.get('class_name', 'unknown')
            for f in cache.get('files', []):
                all_files.append((cname, f))

        if not all_files:
            raise RuntimeError('No cached train feature files found for global fusion pretraining.')

        print(f'[FusionMemory] Global fusion pretraining classes: {class_names}')
        print(f'[FusionMemory] Number of cached train samples: {len(all_files)}')
        print(f'[FusionMemory] Saving global fusion checkpoint to: {ckpt_path}')

        for epoch in range(epochs):
            self.fusion_block.train()
            random.shuffle(all_files)
            total_loss, steps = 0.0, 0

            pbar = tqdm(all_files, desc=f'Global fusion pretrain epoch {epoch + 1}/{epochs}')
            for cname, path in pbar:
                item = torch.load(path, map_location='cpu')
                xyz_patch, rgb_patch, infra_patch = self._sample_cached_patches_for_pretrain(item, max_patches)

                loss = self.fusion_block(xyz_patch, rgb_patch, infra_patch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite global fusion pretrain loss: {float(loss.detach().cpu())}')

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_block.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                steps += 1
                pbar.set_postfix(cls=cname, loss=f'{total_loss / max(steps, 1):.6f}')

            print(f'Global fusion pretrain epoch {epoch + 1}: loss={total_loss / max(steps, 1):.6f}')

        self.fusion_block.eval()
        self.fusion_pretrained = True
        self.save_fusion_block(ckpt_path, epoch=epochs - 1)

    def pretrain_fusion_block_from_cache(self, train_cache, class_name=None):
        """Pretrain fusion block with cached high-order tri-modal patch contrastive learning."""
        epochs = int(getattr(self.args, 'fusion_pretrain_epochs', 0))
        if epochs <= 0:
            return

        ckpt_path = self._fusion_checkpoint_path(class_name or 'default')
        if os.path.exists(ckpt_path) and not bool(getattr(self.args, 'fusion_force_pretrain', False)):
            self.load_fusion_block(ckpt_path)
            return

        lr = float(getattr(self.args, 'fusion_pretrain_lr', 1e-4))
        wd = float(getattr(self.args, 'fusion_pretrain_weight_decay', 1e-5))
        max_patches = int(getattr(self.args, 'fusion_pretrain_max_patches', 512))
        optimizer = torch.optim.AdamW(self.fusion_block.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

        for epoch in range(epochs):
            self.fusion_block.train()
            total_loss, steps = 0.0, 0
            pbar = tqdm(self.iter_feature_cache(train_cache, shuffle=True), total=len(train_cache['files']),
                        desc=f'Pretraining fusion block {class_name} epoch {epoch + 1}/{epochs}')
            for item in pbar:
                xyz_patch = item['xyz_patch'].to(self.device, non_blocking=True)
                rgb_patch = item['rgb_patch'].to(self.device, non_blocking=True)
                infra_patch = item['infra_patch'].to(self.device, non_blocking=True)

                n = xyz_patch.shape[0]
                if max_patches > 0 and n > max_patches:
                    idx = torch.randperm(n, device=self.device)[:max_patches]
                    xyz_patch = xyz_patch[idx]
                    rgb_patch = rgb_patch[idx]
                    infra_patch = infra_patch[idx]

                loss = self.fusion_block(xyz_patch, rgb_patch, infra_patch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite fusion pretrain loss: {float(loss.detach().cpu())}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_block.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                steps += 1
                pbar.set_postfix(loss=f'{total_loss / max(steps, 1):.6f}')

            print(f'Fusion pretrain {class_name} epoch {epoch + 1}: loss={total_loss / max(steps, 1):.6f}')

        self.fusion_block.eval()
        self.fusion_pretrained = True
        self.save_fusion_block(ckpt_path, epoch=epochs - 1)

    # Keep the old method for compatibility, but prefer the cached version.
    def pretrain_fusion_block(self, train_loader, class_name=None):
        train_cache = self.build_feature_cache(train_loader, class_name or 'default', split='train')
        return self.pretrain_fusion_block_from_cache(train_cache, class_name=class_name)

    def _make_fusion_patch(self, xyz_patch, rgb_patch, infra_patch):
        """Return learned tri-modal fusion embedding [num_patches, fusion_embed_dim]."""
        if hasattr(self, 'fusion_block'):
            xyz_patch = xyz_patch.to(self.device)
            rgb_patch = rgb_patch.to(self.device)
            infra_patch = infra_patch.to(self.device)
            self.fusion_block.eval()
            with torch.no_grad():
                fusion = self.fusion_block.encode(xyz_patch, rgb_patch, infra_patch)
            return fusion

        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch)
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch)
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch)
        xyz_n = F.normalize(xyz_patch.float(), p=2, dim=1, eps=1e-12)
        rgb_n = F.normalize(rgb_patch.float(), p=2, dim=1, eps=1e-12)
        infra_n = F.normalize(infra_patch.float(), p=2, dim=1, eps=1e-12)
        fusion = torch.cat([xyz_n, rgb_n, infra_n], dim=1)
        fusion = F.normalize(fusion, p=2, dim=1, eps=1e-12)
        return fusion

    # ------------------------------------------------------------------
    # Memory-bank construction from cached features
    # ------------------------------------------------------------------
    def add_cached_item_to_mem_bank(self, item):
        # Keep cached tensors multi-scale for fusion encoding, but store
        # single [N,D] features in the three modality-specific memory banks.
        xyz_patch_ms = item['xyz_patch']
        rgb_patch_ms = item['rgb_patch']
        infra_patch_ms = item['infra_patch']

        fusion_patch = self._make_fusion_patch(
            xyz_patch_ms, rgb_patch_ms, infra_patch_ms
        ).detach().cpu()

        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch_ms).detach().cpu()
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch_ms).detach().cpu()
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch_ms).detach().cpu()

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        self.patch_fusion_lib.append(fusion_patch)

    def add_sample_to_mem_bank(self, sample, class_name=None):
        # Fallback path without cache. The extractor returns multi-scale
        # [N,3,D] tensors; only memory banks use the scale mean.
        xyz_patch_ms, rgb_patch_ms, infra_patch_ms, _ = self._extract_aligned_patches(sample)
        fusion_patch = self._make_fusion_patch(
            xyz_patch_ms, rgb_patch_ms, infra_patch_ms
        ).detach().cpu()

        self.patch_xyz_lib.append(
            self._reduce_multiscale_patch_for_bank(xyz_patch_ms).detach().cpu()
        )
        self.patch_rgb_lib.append(
            self._reduce_multiscale_patch_for_bank(rgb_patch_ms).detach().cpu()
        )
        self.patch_infra_lib.append(
            self._reduce_multiscale_patch_for_bank(infra_patch_ms).detach().cpu()
        )
        self.patch_fusion_lib.append(fusion_patch)

    def _dlf_score_weights(self):
        """
        M3DM-style DLF object-score weights.

        The vector order is [PC/XYZ, RGB, Infra, Fusion]. These weights scale
        each memory-bank object score before it is fed into D_a/detect_fuser.
        Defaults: first three modalities = 1.0, fusion = 0.1.
        """
        return torch.tensor([
            float(getattr(self.args, 'xyz_s_lambda', getattr(self.args, 'pc_s_lambda', 1.0))),
            float(getattr(self.args, 'rgb_s_lambda', 1.0)),
            float(getattr(self.args, 'infra_s_lambda', 1.0)),
            float(getattr(self.args, 'fusion_s_lambda', 0.1)),
        ], dtype=torch.float32)

    def _dlf_smap_weights(self):
        """
        M3DM-style DLF segmentation-map weights.

        The vector order is [PC/XYZ, RGB, Infra, Fusion]. These weights scale
        each memory-bank score map before it is fed into D_s/seg_fuser.
        Defaults: first three modalities = 1.0, fusion = 0.1.
        """
        return torch.tensor([
            float(getattr(self.args, 'xyz_smap_lambda', getattr(self.args, 'pc_smap_lambda', 1.0))),
            float(getattr(self.args, 'rgb_smap_lambda', 1.0)),
            float(getattr(self.args, 'infra_smap_lambda', 1.0)),
            float(getattr(self.args, 'fusion_smap_lambda', 0.1)),
        ], dtype=torch.float32)

    def add_cached_item_to_late_fusion_mem_bank(self, item):
        xyz_patch_ms = item['xyz_patch'].to(self.device)
        rgb_patch_ms = item['rgb_patch'].to(self.device)
        infra_patch_ms = item['infra_patch'].to(self.device)

        fusion_patch = self._make_fusion_patch(
            xyz_patch_ms, rgb_patch_ms, infra_patch_ms
        ).to(self.device)

        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch_ms).to(self.device)
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch_ms).to(self.device)
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch_ms).to(self.device)

        xyz_patch = ((xyz_patch - self.xyz_mean) / self.xyz_std).detach().cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean) / self.rgb_std).detach().cpu()
        infra_patch = ((infra_patch - self.infra_mean) / self.infra_std).detach().cpu()
        fusion_patch = ((fusion_patch - self.fusion_mean) / self.fusion_std).detach().cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        s_xyz, s_map_xyz_2d = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz_2d')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        # M3DM Decision Layer Fusion (DLF), detection branch D_a:
        # collect one score vector per normal training sample.
        s = torch.tensor([[float(s_xyz), float(s_rgb), float(s_infra), float(s_fusion)]], dtype=torch.float32)
        s = s * self._dlf_score_weights().view(1, -1)
        self.s_lib.append(s)

        # M3DM Decision Layer Fusion (DLF), segmentation branch D_s:
        # collect one vector per spatial location. Original M3DM trains a separate
        # novelty classifier for segmentation maps, not only image-level scores.
        map_vec = torch.stack([
            s_map_xyz_2d.detach().cpu().flatten(),
            s_map_rgb.detach().cpu().flatten(),
            s_map_infra.detach().cpu().flatten(),
            s_map_fusion.detach().cpu().flatten(),
        ], dim=1).float()
        map_vec = map_vec * self._dlf_smap_weights().view(1, -1)

        max_seg = int(getattr(self.args, 'm3dm_dlf_max_seg_pixels', 0))
        if max_seg > 0 and map_vec.shape[0] > max_seg:
            perm = torch.randperm(map_vec.shape[0])[:max_seg]
            map_vec = map_vec[perm]
        self.s_map_lib.append(map_vec)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        # Fallback path without cache.
        xyz_patch, rgb_patch, infra_patch, _ = self._extract_aligned_patches(sample)
        item = {'xyz_patch': xyz_patch.detach().cpu(), 'rgb_patch': rgb_patch.detach().cpu(), 'infra_patch': infra_patch.detach().cpu()}
        self.add_cached_item_to_late_fusion_mem_bank(item)

    def run_late_fusion(self):
        """
        M3DM original Decision Layer Fusion (DLF) style.

        The official M3DM code uses two novelty classifiers in the decision
        layer: D_a for object-level anomaly detection and D_s for segmentation.
        In code these are SGDOneClassSVM modules named detect_fuser and
        seg_fuser. Here we reuse that same mechanism, but extend the input
        score vector from M3DM's three memory banks to our four memory banks:
        [PC, RGB, Infra, Fusion].
        """
        if len(self.s_lib) == 0:
            raise RuntimeError('DLF training failed: self.s_lib is empty.')

        self.s_lib = torch.cat(self.s_lib, 0).detach().cpu()
        self.detect_fuser.fit(self.s_lib)
        print(f'[M3DM-DLF] fitted D_a/detect_fuser on score vectors: {tuple(self.s_lib.shape)}')

        if len(getattr(self, 's_map_lib', [])) > 0:
            self.s_map_lib = torch.cat(self.s_map_lib, 0).detach().cpu()
            self.seg_fuser.fit(self.s_map_lib)
            print(f'[M3DM-DLF] fitted D_s/seg_fuser on score-map vectors: {tuple(self.s_map_lib.shape)}')

    # ------------------------------------------------------------------
    # Prediction from cached features
    # ------------------------------------------------------------------
    def predict_cached_item(self, item):
        label = item['label']
        pixel_mask = item['pixel_mask']
        if torch.is_tensor(label):
            label_s = 1 if torch.any(label > 0).item() else 0
        else:
            label_s = 1 if any([int(x) > 0 for x in label]) else 0

        xyz_patch_ms = item['xyz_patch'].to(self.device)
        rgb_patch_ms = item['rgb_patch'].to(self.device)
        infra_patch_ms = item['infra_patch'].to(self.device)
        pts = item['pts'].to(self.device) if torch.is_tensor(item['pts']) else item['pts']

        # Fusion encoder sees the correct multi-scale tensors [N,3,D].
        fusion_patch = self._make_fusion_patch(
            xyz_patch_ms, rgb_patch_ms, infra_patch_ms
        ).to(self.device)

        # Memory-bank scoring sees one vector per patch [N,D].
        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch_ms).to(self.device)
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch_ms).to(self.device)
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch_ms).to(self.device)

        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, fusion_patch, label_s, pixel_mask, pts)

    def predict(self, sample, label, pixel_mask):
        # Fallback path without cache.
        if label[0] == 1 or label[1] == 1 or label[2] == 1:
            label_s = 1
        else:
            label_s = 0
        xyz_patch_ms, rgb_patch_ms, infra_patch_ms, pts = self._extract_aligned_patches(sample)
        fusion_patch = self._make_fusion_patch(
            xyz_patch_ms, rgb_patch_ms, infra_patch_ms
        ).to(self.device)
        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch_ms).to(self.device)
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch_ms).to(self.device)
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch_ms).to(self.device)
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, fusion_patch, label_s, pixel_mask, pts)

    def _append_object_combination_scores(self, s_xyz, s_rgb, s_infra, s_fusion):
        """Store sample-level scores for single banks and requested combinations.

        Each score is distance-like, so larger means more anomalous. For simple
        decision ablations we use max pooling across the selected memory banks.
        The trained M3DM D_a score is still stored in self.image_preds separately.
        """
        pc = float(s_xyz)
        rgb = float(s_rgb)
        infra = float(s_infra)
        fusion = float(s_fusion)

        combos = {
            'PC': (pc,),
            'RGB': (rgb,),
            'Infra': (infra,),
            'Fusion': (fusion,),
            'PC+RGB': (pc, rgb),
            'PC+Infra': (pc, infra),
            'RGB+Infra': (rgb, infra),
            'PC+Fusion': (pc, fusion),
            'RGB+Fusion': (rgb, fusion),
            'Infra+Fusion': (infra, fusion),
            'PC+RGB+Fusion': (pc, rgb, fusion),
            'PC+Infra+Fusion': (pc, infra, fusion),
            'RGB+Infra+Fusion': (rgb, infra, fusion),
            'PC+RGB+Infra': (pc, rgb, infra),
            'PC+RGB+Infra+Fusion': (pc, rgb, infra, fusion),
        }

        if not hasattr(self, 'object_score_dict'):
            self.object_score_dict = {name: [] for name in combos}

        for name, values in combos.items():
            self.object_score_dict.setdefault(name, []).append(max(values))



    def _minmax_np(self, arr):
        """Per-sample min-max normalization for pixel-map fusion."""
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            return arr
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - mn) / (mx - mn)

    def _make_2d_pixel_gt(self, pixel_mask, target_shape, indices):
        """Union selected 2D modality masks on the image grid."""
        masks = []
        for idx in indices:
            try:
                m = pixel_mask[idx]
            except Exception:
                continue
            if torch.is_tensor(m):
                m = m.detach().cpu().squeeze()
                if tuple(m.shape) == tuple(target_shape):
                    masks.append((m > 0).float().numpy())
            else:
                m = np.asarray(m).squeeze()
                if tuple(m.shape) == tuple(target_shape):
                    masks.append((m > 0).astype(np.float32))

        if len(masks) == 0:
            return np.zeros(target_shape, dtype=np.float32)
        return np.maximum.reduce(masks).astype(np.float32)

    def _sample_2d_numpy_to_points(self, map_2d, pts):
        """Sample a 2D numpy score/mask map to the original point grid."""
        t = torch.as_tensor(map_2d, dtype=torch.float32, device=self.device)
        return sample_2d_score_to_points(t, pts)

    def _append_pixel_combo(self, name, pred, label):
        if not hasattr(self, 'pixel_combo_raw'):
            self.pixel_combo_raw = {}
        self.pixel_combo_raw.setdefault(name, {'preds': [], 'labels': []})
        self.pixel_combo_raw[name]['preds'].extend(np.asarray(pred).reshape(-1))
        self.pixel_combo_raw[name]['labels'].extend(np.asarray(label).reshape(-1))

    def _append_all_pixel_combination_scores(self, s_map_xyz_points, s_map_rgb, s_map_infra, s_map_fusion, pixel_mask, pts):
        """
        Store pixel-level scores for modality combinations.

        Score fusion rule: max over per-sample min-max normalized modality score maps.
        Ground-truth rule: union of the selected modalities' binary masks.
        For combinations with PC, all 2D maps/masks are sampled to the point grid.
        """
        rgb_map = s_map_rgb.detach().cpu().squeeze().numpy()
        infra_map = s_map_infra.detach().cpu().squeeze().numpy()
        fusion_map = s_map_fusion.detach().cpu().squeeze().numpy()
        target_shape = rgb_map.shape

        rgb_gt = self._make_2d_pixel_gt(pixel_mask, target_shape, [0])
        infra_gt = self._make_2d_pixel_gt(pixel_mask, target_shape, [1])
        fusion_gt = self._make_fusion_pixel_gt(pixel_mask, target_shape)

        def combine_2d(name, score_maps, gt_maps):
            score = np.maximum.reduce([self._minmax_np(x) for x in score_maps])
            gt = np.maximum.reduce([(np.asarray(x) > 0).astype(np.float32) for x in gt_maps])
            self._append_pixel_combo(name, score, gt)

        combine_2d('RGB+Infra', [rgb_map, infra_map], [rgb_gt, infra_gt])
        combine_2d('RGB+Fusion', [rgb_map, fusion_map], [rgb_gt, fusion_gt])
        combine_2d('Infra+Fusion', [infra_map, fusion_map], [infra_gt, fusion_gt])
        combine_2d('RGB+Infra+Fusion', [rgb_map, infra_map, fusion_map], [rgb_gt, infra_gt, fusion_gt])

        # Point-domain combinations for anything containing PC.
        pc_score = np.asarray(s_map_xyz_points).reshape(-1)
        pc_label = np.asarray(pixel_mask[2]).reshape(-1).astype(np.float32)

        rgb_score_p = self._sample_2d_numpy_to_points(rgb_map, pts)
        infra_score_p = self._sample_2d_numpy_to_points(infra_map, pts)
        fusion_score_p = self._sample_2d_numpy_to_points(fusion_map, pts)

        rgb_gt_p = (self._sample_2d_numpy_to_points(rgb_gt, pts) > 0.5).astype(np.float32)
        infra_gt_p = (self._sample_2d_numpy_to_points(infra_gt, pts) > 0.5).astype(np.float32)
        fusion_gt_p = (self._sample_2d_numpy_to_points(fusion_gt, pts) > 0.5).astype(np.float32)

        n = min(
            pc_score.size, pc_label.size,
            rgb_score_p.size, infra_score_p.size, fusion_score_p.size,
            rgb_gt_p.size, infra_gt_p.size, fusion_gt_p.size,
        )
        if n <= 0:
            return

        pc_score = pc_score[:n]
        pc_label = pc_label[:n]
        rgb_score_p = rgb_score_p[:n]
        infra_score_p = infra_score_p[:n]
        fusion_score_p = fusion_score_p[:n]
        rgb_gt_p = rgb_gt_p[:n]
        infra_gt_p = infra_gt_p[:n]
        fusion_gt_p = fusion_gt_p[:n]

        def combine_point(name, score_arrays, gt_arrays):
            score = np.maximum.reduce([self._minmax_np(x) for x in score_arrays])
            gt = np.maximum.reduce([(np.asarray(x) > 0).astype(np.float32) for x in gt_arrays])
            self._append_pixel_combo(name, score, gt)

        combine_point('PC+RGB', [pc_score, rgb_score_p], [pc_label, rgb_gt_p])
        combine_point('PC+Infra', [pc_score, infra_score_p], [pc_label, infra_gt_p])
        combine_point('PC+Fusion', [pc_score, fusion_score_p], [pc_label, fusion_gt_p])
        combine_point('PC+RGB+Infra', [pc_score, rgb_score_p, infra_score_p], [pc_label, rgb_gt_p, infra_gt_p])
        combine_point('PC+RGB+Fusion', [pc_score, rgb_score_p, fusion_score_p], [pc_label, rgb_gt_p, fusion_gt_p])
        combine_point('PC+Infra+Fusion', [pc_score, infra_score_p, fusion_score_p], [pc_label, infra_gt_p, fusion_gt_p])
        combine_point('PC+RGB+Infra+Fusion', [pc_score, rgb_score_p, infra_score_p, fusion_score_p], [pc_label, rgb_gt_p, infra_gt_p, fusion_gt_p])

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, fusion_patch, label, pixel_mask, pts):
        # Be defensive: this function expects [N,D] for modality banks. If a
        # caller passes cached [N,3,D], reduce it here. Contrastive learning is
        # unaffected because it happens before this scoring stage.
        xyz_patch = self._reduce_multiscale_patch_for_bank(xyz_patch).to(self.device)
        rgb_patch = self._reduce_multiscale_patch_for_bank(rgb_patch).to(self.device)
        infra_patch = self._reduce_multiscale_patch_for_bank(infra_patch).to(self.device)

        xyz_patch = ((xyz_patch - self.xyz_mean) / self.xyz_std).detach().cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean) / self.rgb_std).detach().cpu()
        infra_patch = ((infra_patch - self.infra_mean) / self.infra_std).detach().cpu()
        fusion_patch = ((fusion_patch - self.fusion_mean) / self.fusion_std).detach().cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        s_xyz, s_map_xyz_2d = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz_2d')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size, modal='fusion')

        self._append_object_combination_scores(s_xyz, s_rgb, s_infra, s_fusion)

        score_vec = torch.tensor([[float(s_xyz), float(s_rgb), float(s_infra), float(s_fusion)]], dtype=torch.float32)
        score_vec = score_vec * self._dlf_score_weights().view(1, -1)
        use_m3dm_dlf = self.args.memory_bank == 'multiple' and hasattr(self.detect_fuser, 'coef_')
        if use_m3dm_dlf:
            # M3DM D_a: object-level novelty classifier over memory-bank scores.
            s = torch.tensor(self.detect_fuser.score_samples(score_vec))
        else:
            s = score_vec.max(dim=1).values

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # M3DM D_s: pixel-level novelty classifier over memory-bank score maps.
        # This is the official M3DM decision-layer idea adapted from 3 banks to 4.
        fused_seg_map = None
        if use_m3dm_dlf and hasattr(self.seg_fuser, 'coef_'):
            map_vec = torch.stack([
                s_map_xyz_2d.detach().cpu().flatten(),
                s_map_rgb.detach().cpu().flatten(),
                s_map_infra.detach().cpu().flatten(),
                s_map_fusion.detach().cpu().flatten(),
            ], dim=1).float()
            map_vec = map_vec * self._dlf_smap_weights().view(1, -1)
            fused_seg = torch.tensor(self.seg_fuser.score_samples(map_vec))
            fused_seg_map = fused_seg.view_as(s_map_rgb.detach().cpu())

        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        s_map_xyz_points = sample_2d_score_to_points(s_map_xyz_2d.to(self.device), pts)
        self.pc_pixel_preds.extend(s_map_xyz_points.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        if torch.is_tensor(pts):
            self.pc_pts.append(pts[0, :].detach().cpu())
        self.pc_predictions.append(s_map_xyz_points.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

        # Fusion-memory pixel metrics. The fusion score map is a 2D 224 x 224 map,
        # so its default ground truth is the union of available 2D masks
        # (RGB and Infra). If only one compatible 2D mask is available, use it.
        fusion_map_np = s_map_fusion.detach().cpu().squeeze().numpy()
        fusion_gt_np = self._make_fusion_pixel_gt(pixel_mask, target_shape=fusion_map_np.shape)

        if not hasattr(self, 'fusion_pixel_preds'):
            self.fusion_pixel_preds = []
        if not hasattr(self, 'fusion_pixel_labels'):
            self.fusion_pixel_labels = []
        if not hasattr(self, 'fusion_predictions'):
            self.fusion_predictions = []
        if not hasattr(self, 'fusion_gts'):
            self.fusion_gts = []

        self.fusion_pixel_preds.extend(fusion_map_np.flatten())
        self.fusion_pixel_labels.extend(fusion_gt_np.flatten())
        self.fusion_predictions.append(fusion_map_np)
        self.fusion_gts.append(fusion_gt_np)

        self._append_all_pixel_combination_scores(
            s_map_xyz_points=s_map_xyz_points,
            s_map_rgb=s_map_rgb,
            s_map_infra=s_map_infra,
            s_map_fusion=s_map_fusion,
            pixel_mask=pixel_mask,
            pts=pts,
        )

        if not hasattr(self, 'm3dm_dlf_predictions'):
            self.m3dm_dlf_predictions = []
        if fused_seg_map is not None:
            dlf_map_np = fused_seg_map.detach().cpu().squeeze().numpy()
            if not hasattr(self, 'm3dm_dlf_pixel_preds'):
                self.m3dm_dlf_pixel_preds = []
            if not hasattr(self, 'm3dm_dlf_pixel_labels'):
                self.m3dm_dlf_pixel_labels = []
            self.m3dm_dlf_pixel_preds.extend(dlf_map_np.flatten())
            self.m3dm_dlf_pixel_labels.extend(fusion_gt_np.flatten())
            self.m3dm_dlf_predictions.append(dlf_map_np)

    def _make_fusion_pixel_gt(self, pixel_mask, target_shape):
        """
        Build a 2D ground-truth mask for the learned fusion-memory score map.

        The learned fusion score map lives on the image grid after interpolation
        to 224 x 224. Therefore we use the union of RGB and Infra masks when
        their shapes match the fusion map. If only one compatible 2D mask exists,
        it is used. Point labels are only used if they already have the same 2D
        shape, otherwise they remain reserved for PC point-level metrics.
        """
        masks = []
        for idx in (0, 1, 2):
            try:
                m = pixel_mask[idx]
            except Exception:
                continue
            if torch.is_tensor(m):
                m = m.detach().cpu().squeeze()
                if tuple(m.shape) == tuple(target_shape):
                    masks.append((m > 0).float())
            else:
                m = np.asarray(m).squeeze()
                if tuple(m.shape) == tuple(target_shape):
                    masks.append(torch.from_numpy((m > 0).astype(np.float32)))

        if len(masks) == 0:
            # Defensive fallback: use an all-zero mask rather than crashing.
            return np.zeros(target_shape, dtype=np.float32)

        gt = torch.stack(masks, dim=0).amax(dim=0)
        return gt.numpy().astype(np.float32)

    def _compute_binary_pixel_metrics(self, labels, preds):
        """Return ROC-AUC, best-F1, AUPR, and AP for flattened pixel scores."""
        from sklearn.metrics import (
            auc,
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
        )

        y_true = np.asarray(labels).astype(np.int32).reshape(-1)
        y_score = np.asarray(preds).astype(np.float64).reshape(-1)

        valid = np.isfinite(y_score)
        y_true = y_true[valid]
        y_score = y_score[valid]

        if y_true.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        # ROC-AUC is undefined if only one class is present.
        if np.unique(y_true).size < 2:
            rocauc = 0.0
        else:
            rocauc = float(roc_auc_score(y_true, y_score))

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        f1_values = (2.0 * precision * recall) / (precision + recall + 1e-12)
        f1 = float(np.nanmax(f1_values)) if f1_values.size > 0 else 0.0
        aupr = float(auc(recall, precision)) if precision.size > 0 else 0.0
        ap = float(average_precision_score(y_true, y_score)) if y_true.size > 0 else 0.0

        return rocauc, f1, aupr, ap

    def calculate_metrics(self):
        """
        Calculate the original MulSen-AD metrics and additionally calculate
        learned fusion-memory pixel metrics.
        """
        super().calculate_metrics()

        if hasattr(self, 'fusion_pixel_preds') and len(self.fusion_pixel_preds) > 0:
            (
                self.fusion_pixel_rocauc,
                self.fusion_pixel_f1,
                self.fusion_pixel_aupr,
                self.fusion_pixel_ap,
            ) = self._compute_binary_pixel_metrics(
                self.fusion_pixel_labels,
                self.fusion_pixel_preds,
            )
        else:
            self.fusion_pixel_rocauc = 0.0
            self.fusion_pixel_f1 = 0.0
            self.fusion_pixel_aupr = 0.0
            self.fusion_pixel_ap = 0.0

        # Optional: metrics for the D_s decision-layer fused segmentation map,
        # if memory_bank='multiple' and seg_fuser was fitted.
        if hasattr(self, 'm3dm_dlf_pixel_preds') and len(self.m3dm_dlf_pixel_preds) > 0:
            (
                self.m3dm_dlf_pixel_rocauc,
                self.m3dm_dlf_pixel_f1,
                self.m3dm_dlf_pixel_aupr,
                self.m3dm_dlf_pixel_ap,
            ) = self._compute_binary_pixel_metrics(
                self.m3dm_dlf_pixel_labels,
                self.m3dm_dlf_pixel_preds,
            )
        else:
            self.m3dm_dlf_pixel_rocauc = 0.0
            self.m3dm_dlf_pixel_f1 = 0.0
            self.m3dm_dlf_pixel_aupr = 0.0
            self.m3dm_dlf_pixel_ap = 0.0


        # Pixel-level metrics for requested modality combinations.
        self.pixel_combo_metrics = {}
        for name, data in getattr(self, 'pixel_combo_raw', {}).items():
            preds = data.get('preds', [])
            labels_combo = data.get('labels', [])
            if len(preds) == 0 or len(labels_combo) == 0:
                continue
            rocauc, f1, aupr, ap = self._compute_binary_pixel_metrics(labels_combo, preds)
            self.pixel_combo_metrics[name] = {
                'rocauc': rocauc,
                'f1': f1,
                'aupr': aupr,
                'ap': ap,
            }

        # Sample-level AUROC for single memory banks and all requested combinations.
        from sklearn.metrics import roc_auc_score

        labels = np.asarray(self.image_labels).astype(np.int32).reshape(-1)
        self.object_score_rocauc = {}

        for name, scores in getattr(self, 'object_score_dict', {}).items():
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            if labels.size == 0 or scores.size != labels.size or np.unique(labels).size < 2:
                self.object_score_rocauc[name] = 0.0
            else:
                valid = np.isfinite(scores)
                if valid.sum() == 0 or np.unique(labels[valid]).size < 2:
                    self.object_score_rocauc[name] = 0.0
                else:
                    self.object_score_rocauc[name] = float(roc_auc_score(labels[valid], scores[valid]))

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz_2d'):
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000
        m_test = patch[s_idx].unsqueeze(0)

        if modal in ['xyz', 'xyz_2d']:
            bank = self.patch_xyz_lib
        elif modal == 'rgb':
            bank = self.patch_rgb_lib
        elif modal == 'infra':
            bank = self.patch_infra_lib
        elif modal == 'fusion':
            bank = self.patch_fusion_lib
        else:
            raise ValueError(f'Unknown modal: {modal}')

        m_star = bank[min_idx[s_idx]].unsqueeze(0)
        w_dist = torch.cdist(m_star, bank)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)
        m_star_knn = torch.linalg.norm(m_test - bank[nn_idx[0, 1:]], dim=1) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1], dtype=patch.dtype))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-8))
        s = w * s_star

        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear', align_corners=True)
        s_map = self.blur(s_map)
        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = self._reduce_multiscale_patch_for_bank(
            torch.cat(self.patch_xyz_lib, 0)
        ).detach().cpu()
        self.patch_rgb_lib = self._reduce_multiscale_patch_for_bank(
            torch.cat(self.patch_rgb_lib, 0)
        ).detach().cpu()
        self.patch_infra_lib = self._reduce_multiscale_patch_for_bank(
            torch.cat(self.patch_infra_lib, 0)
        ).detach().cpu()
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0).detach().cpu()

        print(
            '[FusionMemory] Memory bank shapes after scale reduction: '
            f'xyz={tuple(self.patch_xyz_lib.shape)}, '
            f'rgb={tuple(self.patch_rgb_lib.shape)}, '
            f'infra={tuple(self.patch_infra_lib.shape)}, '
            f'fusion={tuple(self.patch_fusion_lib.shape)}'
        )

        eps = 1e-8
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib) + eps
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib) + eps
        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + eps
        self.fusion_mean = torch.mean(self.patch_fusion_lib)
        self.fusion_std = torch.std(self.patch_fusion_lib) + eps

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean) / self.infra_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std

        if self.f_coreset < 1:
            n_xyz = max(1, int(self.f_coreset * self.patch_xyz_lib.shape[0]))
            n_rgb = max(1, int(self.f_coreset * self.patch_rgb_lib.shape[0]))
            n_infra = max(1, int(self.f_coreset * self.patch_infra_lib.shape[0]))
            n_fusion = max(1, int(self.f_coreset * self.patch_fusion_lib.shape[0]))

            coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib, n=n_xyz, eps=self.coreset_eps)
            self.patch_xyz_lib = self.patch_xyz_lib[coreset_idx]

            coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib, n=n_rgb, eps=self.coreset_eps)
            self.patch_rgb_lib = self.patch_rgb_lib[coreset_idx]

            coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib, n=n_infra, eps=self.coreset_eps)
            self.patch_infra_lib = self.patch_infra_lib[coreset_idx]

            coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib, n=n_fusion, eps=self.coreset_eps)
            self.patch_fusion_lib = self.patch_fusion_lib[coreset_idx]
