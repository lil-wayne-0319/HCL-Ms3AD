# Paste these helpers near the top of feature_extractors/mulsen_features.py
# after the imports. Then replace the original TripleRGBInfraPointFeatures class
# with the class below.

import torch
import torch.nn.functional as F
import numpy as np
import math


def normalize_xy_to_grid(centers, eps=1e-6):
    """
    Normalize x-y coordinates to [-1, 1].

    Args:
        centers: [B, G, 3] or [B, N, 3]
    Returns:
        xy_norm: [B, G/N, 2]
    """
    xy = centers[..., :2]
    xy_min = xy.amin(dim=1, keepdim=True)
    xy_max = xy.amax(dim=1, keepdim=True)
    xy_norm = (xy - xy_min) / (xy_max - xy_min + eps)
    xy_norm = xy_norm * 2.0 - 1.0
    return xy_norm


def splat_tokens_to_2d(tokens, centers, out_h=28, out_w=28, eps=1e-6):
    """
    Project unordered point tokens to a 2D feature map using x-y soft splatting.

    Args:
        tokens:  [B, G, C]
        centers: [B, G, 3]
    Returns:
        feat_2d: [B, C, out_h, out_w]
        weight:  [B, 1, out_h, out_w]
    """
    B, G, C = tokens.shape
    device = tokens.device
    dtype = tokens.dtype

    centers = centers.to(device=device, dtype=dtype)
    xy = normalize_xy_to_grid(centers)  # [B, G, 2], range [-1, 1]

    x = (xy[..., 0] + 1.0) * 0.5 * (out_w - 1)
    y = (xy[..., 1] + 1.0) * 0.5 * (out_h - 1)

    x0 = torch.floor(x).long().clamp(0, out_w - 1)
    y0 = torch.floor(y).long().clamp(0, out_h - 1)
    x1 = (x0 + 1).clamp(0, out_w - 1)
    y1 = (y0 + 1).clamp(0, out_h - 1)

    wx = x - x0.float()
    wy = y - y0.float()

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w10 = wx * (1.0 - wy)
    w11 = wx * wy

    feat = torch.zeros(B, C, out_h * out_w, device=device, dtype=dtype)
    weight = torch.zeros(B, 1, out_h * out_w, device=device, dtype=dtype)

    def scatter(ix, iy, w):
        idx = iy * out_w + ix  # [B, G]

        idx_feat = idx.unsqueeze(1).expand(B, C, G)
        val = tokens.permute(0, 2, 1) * w.unsqueeze(1)
        feat.scatter_add_(2, idx_feat, val)

        idx_weight = idx.unsqueeze(1)
        weight.scatter_add_(2, idx_weight, w.unsqueeze(1))

    scatter(x0, y0, w00)
    scatter(x0, y1, w01)
    scatter(x1, y0, w10)
    scatter(x1, y1, w11)

    feat = feat / (weight + eps)

    feat_2d = feat.view(B, C, out_h, out_w)
    weight_2d = weight.view(B, 1, out_h, out_w)
    return feat_2d, weight_2d


def sample_2d_score_to_points(score_2d, pts, eps=1e-6):
    """
    Back-project a 2D score map to original point-level scores by x-y sampling.

    Args:
        score_2d: [H, W], [1, H, W], or [1, 1, H, W]
        pts:      [B, N, 3]
    Returns:
        point_score: numpy array, [N]
    """
    if not torch.is_tensor(score_2d):
        score_2d = torch.tensor(score_2d)

    if score_2d.dim() == 2:
        score_2d = score_2d[None, None]
    elif score_2d.dim() == 3:
        score_2d = score_2d[:, None]
    elif score_2d.dim() != 4:
        raise ValueError(f"Unsupported score_2d shape: {score_2d.shape}")

    device = score_2d.device
    dtype = score_2d.dtype
    pts = pts.to(device=device, dtype=dtype)

    xy_norm = normalize_xy_to_grid(pts, eps=eps)  # [B, N, 2]
    grid = xy_norm.unsqueeze(2)                   # [B, N, 1, 2]

    sampled = F.grid_sample(
        score_2d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    point_score = sampled.squeeze(1).squeeze(-1)  # [B, N]
    return point_score[0].detach().cpu().numpy()


class TripleRGBInfraPointFeatures(Features):
    """
    PC-to-2D version of the original TripleRGBInfraPointFeatures.

    Main change:
        original PC tokens: [G, C] memory bank
        new PC tokens:      [28*28, C] memory bank after x-y soft splatting

    It still uses PatchCore/memory-bank scoring and the original MulSen-AD
    evaluation buffers.
    """

    def _pc_tokens_to_2d_patch(self, xyz_feature_maps, center, out_h=28, out_w=28):
        """
        Convert PointMAE group tokens to 2D patch tokens.

        Args:
            xyz_feature_maps: list containing [B, C, G]
            center:           [B, G, 3]
        Returns:
            xyz_patch:  [out_h*out_w, C]
            xyz_2d_map: [B, C, out_h, out_w]
        """
        xyz_tokens = torch.cat(xyz_feature_maps, 1).to(self.device)  # [B, C, G]
        xyz_tokens = xyz_tokens.permute(0, 2, 1).contiguous()        # [B, G, C]

        xyz_2d_map, _ = splat_tokens_to_2d(
            tokens=xyz_tokens,
            centers=center.to(self.device),
            out_h=out_h,
            out_w=out_w,
        )  # [B, C, H, W]

        xyz_patch = xyz_2d_map.reshape(xyz_2d_map.shape[1], -1).T.contiguous()  # [H*W, C]
        return xyz_patch, xyz_2d_map

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(
            sample[0], sample[1], sample[2]
        )

        # RGB: [B, C, 28, 28] -> [784, C]
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T.contiguous()

        # Infra: [B, C, 28, 28] -> [784, C]
        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T.contiguous()

        # PC: [B, C, G] -> [B, C, 28, 28] -> [784, C]
        xyz_patch, _ = self._pc_tokens_to_2d_patch(
            xyz_feature_maps=xyz_feature_maps,
            center=center,
            out_h=28,
            out_w=28,
        )

        self.patch_xyz_lib.append(xyz_patch.detach().cpu())
        self.patch_rgb_lib.append(rgb_patch.detach().cpu())
        self.patch_infra_lib.append(infra_patch.detach().cpu())

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

        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T.contiguous()

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T.contiguous()

        xyz_patch, _ = self._pc_tokens_to_2d_patch(
            xyz_feature_maps=xyz_feature_maps,
            center=center,
            out_h=28,
            out_w=28,
        )

        self.compute_s_s_map(
            xyz_patch=xyz_patch,
            rgb_patch=rgb_patch,
            infra_patch=infra_patch,
            label=label_s,
            pixel_mask=pixel_mask,
            center_idx=center_idx,
            pts=pts,
        )

    def add_sample_to_late_fusion_mem_bank(self, sample):
        rgb = sample[0].to(self.device)
        infra = sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(
            rgb, infra, pc
        )

        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T.contiguous()

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T.contiguous()

        xyz_patch, _ = self._pc_tokens_to_2d_patch(
            xyz_feature_maps=xyz_feature_maps,
            center=center,
            out_h=28,
            out_w=28,
        )

        xyz_patch = ((xyz_patch - self.xyz_mean.to(self.device)) / (self.xyz_std.to(self.device) + 1e-6)).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean.to(self.device)) / (self.rgb_std.to(self.device) + 1e-6)).cpu()
        infra_patch = ((infra_patch - self.infra_mean.to(self.device)) / (self.infra_std.to(self.device) + 1e-6)).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        xyz_feat_size = (28, 28)
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_xyz, _ = self.compute_single_s_s_map(
            xyz_patch, dist_xyz, xyz_feat_size, center_idx=center_idx, pts=pts, modal="xyz_2d"
        )
        s_rgb, _ = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal="rgb")
        s_infra, _ = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal="infra")

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        """
        Compute memory-bank anomaly scores.

        xyz_patch is now 2D PC tokens with shape [784, C], not original group tokens.
        Therefore, PC map is generated as a 2D map first, then back-projected to
        original point-level scores for PC_Pixel_ROCAUC.
        """
        xyz_patch = ((xyz_patch - self.xyz_mean.to(self.device)) / (self.xyz_std.to(self.device) + 1e-6)).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean.to(self.device)) / (self.rgb_std.to(self.device) + 1e-6)).cpu()
        infra_patch = ((infra_patch - self.infra_mean.to(self.device)) / (self.infra_std.to(self.device) + 1e-6)).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        xyz_feat_size = (28, 28)
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_xyz, s_map_xyz_2d = self.compute_single_s_s_map(
            xyz_patch, dist_xyz, xyz_feat_size, center_idx=center_idx, pts=pts, modal="xyz_2d"
        )
        s_rgb, s_map_rgb = self.compute_single_s_s_map(
            rgb_patch, dist_rgb, rgb_feat_size, modal="rgb"
        )
        s_infra, s_map_infra = self.compute_single_s_s_map(
            infra_patch, dist_infra, infra_feat_size, modal="infra"
        )

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])
        if self.args.memory_bank == "multiple":
            s = torch.tensor(self.detect_fuser.score_samples(s))

        # Object-level preds and labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # RGB pixel-level
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        # Infra pixel-level
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        # PC point-level: 2D score map -> original point scores
        s_map_xyz_points = sample_2d_score_to_points(
            score_2d=s_map_xyz_2d.to(self.device),
            pts=pts,
        )

        pc_gt = pixel_mask[2].detach().cpu().flatten().numpy()
        pc_pred = np.asarray(s_map_xyz_points).reshape(-1)

        # If there is a rare mismatch due to dataset preprocessing, crop to the common length.
        # Ideally this should not happen.
        if pc_pred.shape[0] != pc_gt.shape[0]:
            min_len = min(pc_pred.shape[0], pc_gt.shape[0])
            pc_pred = pc_pred[:min_len]
            pc_gt = pc_gt[:min_len]

        self.pc_pixel_preds.extend(pc_pred)
        self.pc_pixel_labels.extend(pc_gt)
        self.pc_pts.append(pts[0, :].detach().cpu())
        self.pc_predictions.append(pc_pred)
        self.pc_gts.append(pc_gt)

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal="xyz"):
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000

        # Reweighting
        m_test = patch[s_idx].unsqueeze(0)

        if modal in ["xyz", "xyz_2d"]:
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)
        elif modal == "rgb":
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_infra_lib)

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)

        if modal in ["xyz", "xyz_2d"]:
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) / 1000
        elif modal == "rgb":
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1) / 1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1], dtype=torch.float32))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-6))
        s = w * s_star

        if modal == "xyz":
            # Original MulSen-AD point-token mode. Kept for compatibility.
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()
            sample_data = pts[0, center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data, s_xyz_map, pts, k=1)
            return s, full_s_xyz_map

        # RGB / Infra / new XYZ-2D mode: patch map -> 224x224 map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(
            s_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=True,
        )
        s_map = self.blur(s_map)
        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        # Important: original code had copy-paste mistakes here.
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib) + 1e-6

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib) + 1e-6

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + 1e-6

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean) / self.infra_std

        if self.f_coreset < 1:
            n_xyz = max(1, int(self.f_coreset * self.patch_xyz_lib.shape[0]))
            self.coreset_idx = self.get_coreset_idx_randomp(
                self.patch_xyz_lib,
                n=n_xyz,
                eps=self.coreset_eps,
            )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]

            n_rgb = max(1, int(self.f_coreset * self.patch_rgb_lib.shape[0]))
            self.coreset_idx = self.get_coreset_idx_randomp(
                self.patch_rgb_lib,
                n=n_rgb,
                eps=self.coreset_eps,
            )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

            n_infra = max(1, int(self.f_coreset * self.patch_infra_lib.shape[0]))
            self.coreset_idx = self.get_coreset_idx_randomp(
                self.patch_infra_lib,
                n=n_infra,
                eps=self.coreset_eps,
            )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
