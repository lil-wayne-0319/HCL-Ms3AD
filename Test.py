"""
Code based on the official M3DM code found at
https://github.com/nomewang/M3DM
and PatchCore found at
https://github.com/amazon-science/patchcore-inspection
"""
"""
# Copyright (c) 2023 nomewang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from runner import Tester
from dataset import mulsen_classes


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_3d_ads(args):
    """
    Three-stage runner for fusion memory:

    Stage 1: --fusion_extract_only
        Extract and cache frozen RGB / Infra / PC patch features for all selected classes.
        No fusion pretraining, no memory-bank construction, no evaluation.

    Stage 2: --fusion_pretrain_only
        Use all selected classes' training caches together to train ONE global
        tri-modal high-order contrastive fusion module.
        No per-class memory-bank construction, no evaluation.

    Stage 3: normal run
        For each class, load the global fusion module, build class-specific
        RGB / Infra / PC / Fusion memory banks, and evaluate with MulSen-AD metrics.
    """
    if args.random_state is not None:
        set_random_seed(args.random_state)

    if getattr(args, "classes", ""):
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = mulsen_classes()

    timestamp = args.run_timestamp if getattr(args, "run_timestamp", "") else datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, args.method_name, timestamp)

    extract_only = bool(getattr(args, "fusion_extract_only", False))
    pretrain_only = bool(getattr(args, "fusion_pretrain_only", False))
    skip_eval = bool(getattr(args, "fusion_skip_eval", False))

    if extract_only and pretrain_only:
        raise ValueError("--fusion_extract_only and --fusion_pretrain_only cannot both be set.")

    # ------------------------------------------------------------------
    # Stage 1: extract and cache features for all selected classes.
    # ------------------------------------------------------------------
    if extract_only:
        print("\n################################################################################")
        print("[FusionMemory] Stage 1: extract frozen features only")
        print("################################################################################\n")

        for cls in classes:
            print("\n################################################################################")
            print(f"[FusionMemory] Extracting feature cache for class: {cls}")
            print("################################################################################\n")
            model = Tester(args)
            model.extract_fusion_features_only(cls)
            torch.cuda.empty_cache()

        print("\n################################################################################")
        print("[FusionMemory] Finished extracting feature caches for all selected classes.")
        print("################################################################################\n")
        return

    # ------------------------------------------------------------------
    # Stage 2: global high-order contrastive fusion pretraining.
    # ------------------------------------------------------------------
    if pretrain_only:
        print("\n################################################################################")
        print("[FusionMemory] Stage 2: global tri-modal high-order contrastive pretraining")
        print("[FusionMemory] Training caches from all selected classes will be used together.")
        print("################################################################################\n")

        model = Tester(args)
        model.pretrain_global_fusion_only(classes)
        torch.cuda.empty_cache()

        print("\n################################################################################")
        print("[FusionMemory] Finished global fusion pretraining.")
        print("################################################################################\n")
        return

    # ------------------------------------------------------------------
    # Stage 3: per-class memory-bank construction and evaluation.
    # ------------------------------------------------------------------
    all_metrics_df = pd.DataFrame()

    for cls in classes:
        output_dir = os.path.join(args.output_dir, cls)

        print("\n################################################################################")
        print(f"[FusionMemory] Stage 3: class-specific memory banks and evaluation: {cls}")
        print(f"Method: {args.method_name}")
        print("################################################################################\n")

        model = Tester(args)
        model.fit(cls)
        torch.cuda.empty_cache()

        if skip_eval:
            print(f"[FusionMemory] --fusion_skip_eval enabled. Skipping evaluation for class {cls}.")
            torch.cuda.empty_cache()
            continue

        metrics_df = model.evaluate(cls, output_dir)
        if metrics_df is None:
            print(f"[Warning] model.evaluate() returned None for class {cls}; skipping metrics aggregation.")
            torch.cuda.empty_cache()
            continue

        metrics_df["Class"] = cls.title()
        all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)

        torch.cuda.empty_cache()
        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    if all_metrics_df.empty:
        print("\n################################################################################")
        print("No metrics generated.")
        print("################################################################################\n")
        return

    metric_columns = [col for col in all_metrics_df.columns if col not in ["Method", "Class"]]
    mean_row = all_metrics_df[metric_columns].mean(axis=0, skipna=True)

    mean_row_data = {col: mean_row.get(col, None) for col in metric_columns}
    mean_row_data["Method"] = "Mean"
    mean_row_data["Class"] = "Overall"

    all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([mean_row_data])], ignore_index=True)

    print("\n\n################################################################################")
    print("############################# Metrics with Mean ###############################")
    print("################################################################################\n")
    print(all_metrics_df.to_markdown(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "all_metrics_results.csv")
    all_metrics_df.to_csv(output_file, index=False)
    print(f"\nSaved metrics to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MulSen-AD tri-modal fusion-memory runner.")

    parser.add_argument("--method_name", default="PC+RGB+Infra+gating", type=str,
                        help="Anomaly detection method name.")
    parser.add_argument("--classes", default="", type=str,
                        help="Comma-separated class names. Empty means all classes.")
    parser.add_argument("--run_timestamp", default="", type=str,
                        help="Output subfolder name. Empty means current timestamp.")
    parser.add_argument("--memory_bank", default="multiple", type=str,
                        choices=["multiple", "single"],
                        help="memory bank mode: multiple or single.")
    parser.add_argument("--rgb_backbone_name", default="vit_base_patch8_224_dino", type=str,
                        choices=["vit_base_patch8_224_dino", "vit_base_patch8_224", "vit_base_patch8_224_in21k", "vit_small_patch8_224_dino"],
                        help="Timm checkpoint name of RGB backbone.")
    parser.add_argument("--xyz_backbone_name", default="Point_MAE", type=str,
                        choices=["Point_MAE", "Point_Bert"],
                        help="Checkpoint name of point backbone.")

    # Fusion memory / high-order contrastive pretraining options.
    parser.add_argument("--fusion_module_path", default="", type=str,
                        help="Optional pretrained global fusion checkpoint. If empty, use fusion_pretrain_dir/fusion_global_ckpt_name.")
    parser.add_argument("--fusion_pretrain_dir", default="./checkpoints/fusion_high_order", type=str,
                        help="Directory to save/load the global fusion pretrain checkpoint.")
    parser.add_argument("--fusion_global_ckpt_name", default="global_tri_modal_high_order_fusion.pth", type=str,
                        help="Filename of the global tri-modal high-order fusion checkpoint.")
    parser.add_argument("--fusion_global_pretrain", default=True, action=argparse.BooleanOptionalAction,
                        help="Use one global fusion checkpoint pretrained on all selected classes. Default: True.")
    parser.add_argument("--fusion_pretrain_epochs", default=5, type=int,
                        help="Epochs for tri-modal high-order fusion pretraining. Set 0 to disable.")
    parser.add_argument("--fusion_feature_cache_dir", default="./cache/fusion_high_order_features", type=str,
                        help="Directory to save/load cached frozen RGB/Infra/PC patch features.")
    parser.add_argument("--fusion_force_recache", default=False, action="store_true",
                        help="Force rebuilding cached frozen patch features even if cache exists.")
    parser.add_argument("--fusion_force_pretrain", default=False, action="store_true",
                        help="Force retraining fusion block even if checkpoint exists.")
    parser.add_argument("--fusion_pretrain_lr", default=1e-4, type=float,
                        help="Learning rate for fusion pretraining.")
    parser.add_argument("--fusion_pretrain_weight_decay", default=1e-5, type=float,
                        help="Weight decay for fusion pretraining.")
    parser.add_argument("--fusion_pretrain_max_patches", default=512, type=int,
                        help="Maximum aligned patches sampled from each cached training instance. <=0 uses all patches.")
    parser.add_argument("--fusion_embed_dim", default=512, type=int,
                        help="Output dimension of learned fusion memory-bank feature.")
    parser.add_argument("--fusion_temperature", default=0.07, type=float,
                        help="Temperature for high-order contrastive fusion pretraining.")
    parser.add_argument("--fusion_dropout", default=0.0, type=float,
                        help="Dropout inside fusion projection MLPs.")
    parser.add_argument("--use_multiscale", default=False, action=argparse.BooleanOptionalAction,
                        help="Use input-level multi-scale features: image 1/2 and 1/4 down-up before backbone, point-cloud FPS N/2 and N/4 before point backbone. Default: True. Use --no-use_multiscale to disable.")

    # Stage-control switches.
    parser.add_argument("--fusion_extract_only", default=False, action="store_true",
                        help="Only extract and cache frozen train/test features, then exit.")
    parser.add_argument("--fusion_pretrain_only", default=False, action="store_true",
                        help="Only train the global tri-modal high-order fusion module from cached train features, then exit.")
    parser.add_argument("--fusion_skip_eval", default=False, action="store_true",
                        help="Run fit/build stages but skip evaluation and metric computation.")

    # M3DM-style DLF weighting options.
    parser.add_argument(
        "--xyz_s_lambda",
        default=0.1,
        type=float,
        help="DLF object-score weight for PC/XYZ memory bank.",
    )
    parser.add_argument(
        "--xyz_smap_lambda",
        default=0.1,
        type=float,
        help="DLF segmentation-map weight for PC/XYZ memory bank.",
    )
    parser.add_argument(
        "--rgb_s_lambda",
        default=1.0,
        type=float,
        help="DLF object-score weight for RGB memory bank.",
    )
    parser.add_argument(
        "--rgb_smap_lambda",
        default=1.2,
        type=float,
        help="DLF segmentation-map weight for RGB memory bank.",
    )
    parser.add_argument(
        "--infra_s_lambda",
        default=1.0,
        type=float,
        help="DLF object-score weight for Infrared memory bank.",
    )
    parser.add_argument(
        "--infra_smap_lambda",
        default=1.1,
        type=float,
        help="DLF segmentation-map weight for Infrared memory bank.",
    )
    parser.add_argument(
        "--fusion_s_lambda",
        default=1.0,
        type=float,
        help="DLF object-score weight for learned fusion memory bank.",
    )
    parser.add_argument(
        "--fusion_smap_lambda",
        default=1.15,
        type=float,
        help="DLF segmentation-map weight for learned fusion memory bank.",
    )

    # General options.
    parser.add_argument("--save_feature", default=True, action="store_true",
                        help="Save features for training fusion block.")
    parser.add_argument("--save_preds", default=False, action="store_true",
                        help="Save prediction results.")
    parser.add_argument("--group_size", default=128, type=int,
                        help="Point group size of Point Transformer.")
    parser.add_argument("--num_group", default=1024, type=int,
                        help="Point groups number of Point Transformer.")
    parser.add_argument("--random_state", default=None, type=int,
                        help="Random seed.")
    parser.add_argument("--dataset_path", default="./dataset/MulSen_AD", type=str,
                        help="Dataset path.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Image size for model.")
    parser.add_argument("--coreset_eps", default=0.9, type=float,
                        help="eps for sparse projection.")
    parser.add_argument("--f_coreset", default=0.1, type=float,
                        help="Coreset sampling ratio.")
    parser.add_argument("--asy_memory_bank", default=None, type=int,
                        help="Build an asymmetric memory bank for point clouds.")
    parser.add_argument("--ocsvm_nu", default=0.5, type=float,
                        help="OCSVM nu.")
    parser.add_argument("--ocsvm_maxiter", default=1000, type=int,
                        help="OCSVM max iterations.")
    parser.add_argument("--rm_zero_for_project", default=False, action="store_true",
                        help="Remove zero points for projection.")
    parser.add_argument("--total_epochs", default=1, type=int,
                        help="Training epochs.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight decay.")
    parser.add_argument("--output_dir", default="./output_dir", type=str,
                        help="Path where to save outputs.")

    args = parser.parse_args()
    run_3d_ads(args)
