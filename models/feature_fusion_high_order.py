import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim, min(in_dim, 1024))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x.float()), p=2, dim=-1, eps=1e-12)


class TriModalHighOrderFusionBlock(nn.Module):
    """
    Input-level multi-scale tri-modal UFF-style fusion block for MulSen-AD.

    This block assumes multi-scale features are already extracted and cached
    before fusion pretraining:
      - image full / 1/2 / 1/4 inputs are each passed through the image backbone
      - point-cloud full / FPS-1/2 / FPS-1/4 inputs are each passed through the point backbone

    Supported inputs for each modality:
      1) Single scale, backward compatible:
         xyz_patch:   [N, xyz_dim]
         rgb_patch:   [N, rgb_dim]
         infra_patch: [N, infra_dim]

      2) Three input-level scales:
         xyz_patch:   [N, 3, xyz_dim]
         rgb_patch:   [N, 3, rgb_dim]
         infra_patch: [N, 3, infra_dim]

      3) list / tuple:
         [full, half, quarter], each [N, D]

    For each modality:
      projected full / half / quarter features are averaged to form one
      single-modal embedding. An intra-modal compactness term pulls the three
      scale features toward their own mean.

    Loss:
      L = L_high_order_contrastive
          + centroid_weight * L_centroid
          + compactness_weight * L_intra_modal_compactness

    Dirichlet-process arguments are kept only for compatibility with older
    runners, but DP is intentionally not used in this version.
    """

    def __init__(
        self,
        xyz_dim=1152,
        rgb_dim=768,
        infra_dim=768,
        embed_dim=512,
        temperature=0.07,
        dropout=0.0,
        compactness_weight=0.1,
        centroid_weight=0.25,
        # Backward-compatible unused DP args.
        use_dp=False,
        dp_alpha=1.0,
        dp_lambda=0.01,
        num_prototypes=64,
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.compactness_weight = compactness_weight
        self.centroid_weight = centroid_weight

        # Explicitly disabled. These attrs prevent old checkpoints/runners from failing.
        self.use_dp = False
        self.dp_alpha = dp_alpha
        self.dp_lambda = dp_lambda
        self.num_prototypes = num_prototypes

        self.xyz_proj = MLPProjector(xyz_dim, embed_dim, dropout=dropout)
        self.rgb_proj = MLPProjector(rgb_dim, embed_dim, dropout=dropout)
        self.infra_proj = MLPProjector(infra_dim, embed_dim, dropout=dropout)

        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _to_scale_tensor(self, x):
        """
        Convert input to [S, N, D].

        Accepted:
          [N, D]    -> [1, N, D]
          [N, S, D] -> [S, N, D]
          [S, N, D] -> [S, N, D]
          list/tuple of [N, D] -> [S, N, D]
        """
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError('Empty multi-scale input list.')
            return torch.stack([t.float() for t in x], dim=0)

        if not torch.is_tensor(x):
            raise TypeError(f'Expected tensor/list/tuple, got {type(x)}')

        if x.dim() == 2:
            return x.float().unsqueeze(0)

        if x.dim() == 3:
            # Prefer [N, S, D] when the middle dimension is small.
            if x.shape[1] in (2, 3, 4):
                return x.permute(1, 0, 2).contiguous().float()
            # Also support [S, N, D].
            if x.shape[0] in (2, 3, 4):
                return x.float()
            raise ValueError(
                f'Ambiguous 3D multi-scale shape {tuple(x.shape)}. '
                'Expected [N,S,D] or [S,N,D], with S usually 3.'
            )

        raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')

    def _encode_multiscale_modality(self, x, projector):
        """
        Encode one modality.

        return:
          z_mean:   [N, embed_dim]
          z_scales: [S, N, embed_dim]
        """
        scale_tokens = self._to_scale_tensor(x)  # [S, N, Din]
        z_scales = []
        for s in range(scale_tokens.shape[0]):
            z_scales.append(projector(scale_tokens[s]))
        z_scales = torch.stack(z_scales, dim=0)
        z_mean = F.normalize(z_scales.mean(dim=0), p=2, dim=-1, eps=1e-12)
        return z_mean, z_scales

    def encode_views_with_scales(self, xyz_patch, rgb_patch, infra_patch):
        z_xyz, z_xyz_scales = self._encode_multiscale_modality(xyz_patch, self.xyz_proj)
        z_rgb, z_rgb_scales = self._encode_multiscale_modality(rgb_patch, self.rgb_proj)
        z_infra, z_infra_scales = self._encode_multiscale_modality(infra_patch, self.infra_proj)

        z_fused = F.normalize(
            self.fusion_head(torch.cat([z_xyz, z_rgb, z_infra], dim=-1)),
            p=2,
            dim=-1,
            eps=1e-12,
        )

        scale_pack = {
            'xyz': z_xyz_scales,
            'rgb': z_rgb_scales,
            'infra': z_infra_scales,
        }
        return z_xyz, z_rgb, z_infra, z_fused, scale_pack

    def encode_views(self, xyz_patch, rgb_patch, infra_patch):
        z_xyz, z_rgb, z_infra, z_fused, _ = self.encode_views_with_scales(
            xyz_patch, rgb_patch, infra_patch
        )
        return z_xyz, z_rgb, z_infra, z_fused

    @torch.no_grad()
    def encode(self, xyz_patch, rgb_patch, infra_patch):
        self.eval()
        _, _, _, z_fused = self.encode_views(xyz_patch, rgb_patch, infra_patch)
        return z_fused

    def high_order_contrastive_loss(self, views):
        """
        views: [V, N, D], where V=4 views: xyz, rgb, infra, fused.
        Positives are all other views of the same patch index.
        """
        v, n, d = views.shape
        z = views.permute(1, 0, 2).reshape(n * v, d)
        patch_ids = torch.arange(n, device=z.device).repeat_interleave(v)

        logits = torch.matmul(z, z.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(n * v, device=z.device, dtype=torch.bool)
        pos_mask = (patch_ids[:, None] == patch_ids[None, :]) & (~self_mask)
        all_mask = ~self_mask

        exp_logits = torch.exp(logits) * all_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
        return loss.mean()

    def centroid_alignment_loss(self, z_xyz, z_rgb, z_infra, z_fused):
        centroid = F.normalize(
            (z_xyz + z_rgb + z_infra + z_fused) / 4.0,
            p=2,
            dim=-1,
            eps=1e-12,
        )
        labels = torch.arange(z_fused.shape[0], device=z_fused.device)

        loss_fused = F.cross_entropy(torch.matmul(z_fused, centroid.t()) / self.temperature, labels)
        loss_modal = 0.0
        for z in (z_xyz, z_rgb, z_infra):
            loss_modal = loss_modal + F.cross_entropy(torch.matmul(z, centroid.t()) / self.temperature, labels)
        return (loss_modal + loss_fused) / 4.0

    def intra_modal_compactness_loss(self, scale_pack):
        """
        Pull full / half / quarter features inside each modality together.
        If only one scale is available, this term is zero for that modality.
        """
        total_loss = 0.0
        count = 0

        for z_scales in scale_pack.values():
            # [S, N, D]
            if z_scales.shape[0] <= 1:
                continue
            center = F.normalize(z_scales.mean(dim=0), p=2, dim=-1, eps=1e-12)
            cos_sim = (z_scales * center.unsqueeze(0)).sum(dim=-1)
            total_loss = total_loss + (1.0 - cos_sim).mean()
            count += 1

        if count == 0:
            first = next(iter(scale_pack.values()))
            return first.sum() * 0.0
        return total_loss / count

    def forward(self, xyz_patch, rgb_patch, infra_patch):
        z_xyz, z_rgb, z_infra, z_fused, scale_pack = self.encode_views_with_scales(
            xyz_patch, rgb_patch, infra_patch
        )
        views = torch.stack([z_xyz, z_rgb, z_infra, z_fused], dim=0)

        loss_supcon = self.high_order_contrastive_loss(views)
        loss_centroid = self.centroid_alignment_loss(z_xyz, z_rgb, z_infra, z_fused)
        loss_compact = self.intra_modal_compactness_loss(scale_pack)

        loss = (
            loss_supcon
            + self.centroid_weight * loss_centroid
            + self.compactness_weight * loss_compact
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                'Non-finite fusion pretrain loss: '
                f'loss_supcon={float(loss_supcon.detach().cpu())}, '
                f'loss_centroid={float(loss_centroid.detach().cpu())}, '
                f'loss_compact={float(loss_compact.detach().cpu())}'
            )
        return loss
