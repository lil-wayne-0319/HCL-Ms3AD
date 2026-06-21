import os
import torch
from tqdm import tqdm
import pandas as pd

from dataset import get_data_loader
from feature_extractors import mulsen_features


class Tester():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        if args.method_name == 'RGB':
            self.methods = {
                "RGB": mulsen_features.RGBFeatures(args),
            }
        elif args.method_name == 'Infra':
            self.methods = {
                "Infra": mulsen_features.InfraFeatures(args),
            }
        elif args.method_name == 'PC':
            self.methods = {
                "PC": mulsen_features.PCFeatures(args),
            }
        elif args.method_name == 'PC+RGB+gating':
            self.methods = {
                "PC+RGB+gating": mulsen_features.PCRGBGatingFeatures(args),
            }
        elif args.method_name == 'PC+Infra+gating':
            self.methods = {
                "PC+Infra+gating": mulsen_features.PCInfraGatingFeatures(args),
            }
        elif args.method_name == 'RGB+Infra+gating':
            self.methods = {
                "RGB+Infra+gating": mulsen_features.RGBInfraGatingFeatures(args),
            }
        elif args.method_name == 'PC+RGB+Infra+gating':
            self.methods = {
                "PC+RGB+Infra+gating": mulsen_features.TripleRGBInfraPointFeatures(args),
            }
        elif args.method_name in ['PC+RGB+Infra+fusion_memory', 'FusionMemory4M', 'M3DMFusionMemory']:
            self.methods = {
                args.method_name: mulsen_features.TripleRGBInfraPointFusionMemoryFeatures(args),
            }
        else:
            raise ValueError(f'Unknown method_name: {args.method_name}')

    def _get_train_loader(self, class_name):
        return get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)

    def _get_test_loader(self, class_name):
        return get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)

    def extract_fusion_features_only(self, class_name):
        """Cache train and test frozen features for one class, then return."""
        train_loader = self._get_train_loader(class_name)
        test_loader = self._get_test_loader(class_name)

        for method_name, method in self.methods.items():
            if not hasattr(method, 'build_feature_cache'):
                raise RuntimeError(
                    f'Method {method_name} does not support feature cache extraction. '
                    'Use --method_name PC+RGB+Infra+fusion_memory.'
                )
            method.build_feature_cache(train_loader, class_name, split='train')
            method.build_feature_cache(test_loader, class_name, split='test')

    def pretrain_global_fusion_only(self, classes):
        """
        Use all selected classes' TRAIN caches to pretrain one global fusion block.
        Caches must already exist, or they will be built if missing.
        """
        if not classes:
            raise ValueError('classes is empty; cannot run global fusion pretraining.')

        for method_name, method in self.methods.items():
            if not hasattr(method, 'pretrain_fusion_block_from_caches'):
                raise RuntimeError(
                    f'Method {method_name} does not support global fusion pretraining. '
                    'Please update feature_extractors/mulsen_features.py.'
                )

            train_caches = []
            for class_name in classes:
                train_loader = self._get_train_loader(class_name)
                cache = method.build_feature_cache(train_loader, class_name, split='train')
                train_caches.append(cache)

            method.pretrain_fusion_block_from_caches(train_caches, class_names=classes)

    def fit(self, class_name):
        train_loader = self._get_train_loader(class_name)

        # Fusion-memory path: extract frozen features once, cache them, then reuse
        # the cached tensor dataset for global-fusion loading, memory-bank building,
        # and late-fusion scoring.
        cache_by_method = {}
        for method_name, method in self.methods.items():
            if hasattr(method, 'build_feature_cache'):
                cache_by_method[method_name] = method.build_feature_cache(train_loader, class_name, split='train')

                # Normal fit should NOT train per-class fusion if a global checkpoint exists.
                # It just loads the global checkpoint. If the checkpoint does not exist and
                # fusion_pretrain_epochs > 0, the method may train from this class cache as a fallback.
                if hasattr(method, 'ensure_fusion_block_ready'):
                    method.ensure_fusion_block_ready(cache_by_method[method_name], class_name=class_name)
                elif hasattr(method, 'pretrain_fusion_block_from_cache'):
                    method.pretrain_fusion_block_from_cache(cache_by_method[method_name], class_name=class_name)

        if bool(getattr(self.args, 'fusion_extract_only', False)):
            return

        print(f'Building train memory banks for class {class_name}')
        if cache_by_method:
            for method_name, method in self.methods.items():
                if method_name in cache_by_method and hasattr(method, 'add_cached_item_to_mem_bank'):
                    for item in tqdm(method.iter_feature_cache(cache_by_method[method_name]),
                                     total=len(cache_by_method[method_name]['files']),
                                     desc=f'Building cached memory bank {method_name} {class_name}'):
                        method.add_cached_item_to_mem_bank(item)
                else:
                    for sample, _ in train_loader:
                        if self.args.save_feature:
                            method.add_sample_to_mem_bank(sample, class_name=class_name)
                        else:
                            method.add_sample_to_mem_bank(sample)
        else:
            for sample, _ in train_loader:
                for method in self.methods.values():
                    if self.args.save_feature:
                        method.add_sample_to_mem_bank(sample, class_name=class_name)
                    else:
                        method.add_sample_to_mem_bank(sample)

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()

        if self.args.memory_bank == 'multiple':
            print(f'MultiScoring for class {class_name}..')
            for method_name, method in self.methods.items():
                if method_name in cache_by_method and hasattr(method, 'add_cached_item_to_late_fusion_mem_bank'):
                    for item in tqdm(method.iter_feature_cache(cache_by_method[method_name]),
                                     total=len(cache_by_method[method_name]['files']),
                                     desc=f'Cached late-fusion scores {method_name} {class_name}'):
                        method.add_cached_item_to_late_fusion_mem_bank(item)
                else:
                    for sample, _ in train_loader:
                        method.add_sample_to_late_fusion_mem_bank(sample)

            for method_name, method in self.methods.items():
                # The original MulSen-AD only trained gating models here. FusionMemory4M
                # reuses the OCSVM late-fusion logic implemented in its feature extractor.
                if 'gating' in method_name or 'fusion_memory' in method_name or method_name in ['FusionMemory4M', 'M3DMFusionMemory']:
                    print(f'\n\nTraining late-fusion unit for {method_name} on class {class_name}...')
                    method.run_late_fusion()

    def evaluate(self, class_name, output_dir):
        metrics_data = []

        test_loader = self._get_test_loader(class_name)

        rgb_paths = []
        infra_paths = []
        pc_paths = []

        with torch.no_grad():
            for method_name, method in self.methods.items():
                if hasattr(method, 'build_feature_cache') and hasattr(method, 'predict_cached_item'):
                    test_cache = method.build_feature_cache(test_loader, class_name, split='test')
                    for item in tqdm(method.iter_feature_cache(test_cache),
                                     total=len(test_cache['files']),
                                     desc=f'Evaluating cached features {method_name} {class_name}'):
                        method.predict_cached_item(item)
                        paths = item.get('paths', None)
                        if paths is not None:
                            rgb_paths.append(paths[0])
                            infra_paths.append(paths[1])
                            pc_paths.append(paths[2])
                        torch.cuda.empty_cache()
                else:
                    for sample, label, pixel_mask, paths in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                        method.predict(sample, label, pixel_mask)
                        rgb_paths.append(paths[0])
                        infra_paths.append(paths[1])
                        pc_paths.append(paths[2])
                        torch.cuda.empty_cache()

        def _round_or_none(value):
            if value is None:
                return None
            try:
                return round(float(value), 3)
            except Exception:
                return None

        def _format_metric_dict(metric_dict, metric_order=('rocauc', 'f1', 'aupr', 'ap')):
            texts = []
            for name, values in metric_dict.items():
                parts = []
                for key in metric_order:
                    if key in values and values[key] is not None:
                        label = key.upper() if key != 'rocauc' else 'ROCAUC'
                        parts.append(f'{label}: {float(values[key]):.3f}')
                texts.append(f'{name} ({", ".join(parts)})')
            return ' | '.join(texts)

        for method_name, method in self.methods.items():
            method.calculate_metrics()

            metrics = {
                "Method": method_name,
                "Image_ROCAUC": _round_or_none(getattr(method, 'image_rocauc', None)),

                "RGB_Pixel_ROCAUC": _round_or_none(getattr(method, 'rgb_pixel_rocauc', None)),
                "RGB_Pixel_F1": _round_or_none(getattr(method, 'rgb_pixel_f1', None)),
                "RGB_Pixel_AUPR": _round_or_none(getattr(method, 'rgb_pixel_aupr', None)),
                "RGB_Pixel_AP": _round_or_none(getattr(method, 'rgb_pixel_ap', None)),

                "Infra_Pixel_ROCAUC": _round_or_none(getattr(method, 'infra_pixel_rocauc', None)),
                "Infra_Pixel_F1": _round_or_none(getattr(method, 'infra_pixel_f1', None)),
                "Infra_Pixel_AUPR": _round_or_none(getattr(method, 'infra_pixel_aupr', None)),
                "Infra_Pixel_AP": _round_or_none(getattr(method, 'infra_pixel_ap', None)),

                "PC_Pixel_ROCAUC": _round_or_none(getattr(method, 'pc_pixel_rocauc', None)),
                "PC_Pixel_F1": _round_or_none(getattr(method, 'pc_pixel_f1', None)),
                "PC_Pixel_AUPR": _round_or_none(getattr(method, 'pc_pixel_aupr', None)),
            }

            # Learned fusion-memory pixel metrics.
            if hasattr(method, 'fusion_pixel_rocauc'):
                metrics["Fusion_Pixel_ROCAUC"] = _round_or_none(method.fusion_pixel_rocauc)
                metrics["Fusion_Pixel_F1"] = _round_or_none(getattr(method, 'fusion_pixel_f1', None))
                metrics["Fusion_Pixel_AUPR"] = _round_or_none(getattr(method, 'fusion_pixel_aupr', None))
                metrics["Fusion_Pixel_AP"] = _round_or_none(getattr(method, 'fusion_pixel_ap', None))

            # M3DM D_s decision-layer fused segmentation metrics.
            if hasattr(method, 'm3dm_dlf_pixel_rocauc'):
                metrics["M3DM_DLF_Pixel_ROCAUC"] = _round_or_none(method.m3dm_dlf_pixel_rocauc)
                metrics["M3DM_DLF_Pixel_F1"] = _round_or_none(getattr(method, 'm3dm_dlf_pixel_f1', None))
                metrics["M3DM_DLF_Pixel_AUPR"] = _round_or_none(getattr(method, 'm3dm_dlf_pixel_aupr', None))
                metrics["M3DM_DLF_Pixel_AP"] = _round_or_none(getattr(method, 'm3dm_dlf_pixel_ap', None))

            # Sample-level AUROC for single memory banks and all requested combinations.
            if hasattr(method, 'object_score_rocauc'):
                for score_name, auc_value in method.object_score_rocauc.items():
                    prefix = score_name.replace('+', '_').replace(' ', '_')
                    metrics[f"Object_{prefix}_ROCAUC"] = _round_or_none(auc_value)

            # Pixel-level metrics for all requested modality combinations.
            # This must be BEFORE metrics_data.append(), otherwise it prints but will not enter CSV.
            if hasattr(method, 'pixel_combo_metrics'):
                for combo_name, combo_metrics in method.pixel_combo_metrics.items():
                    prefix = combo_name.replace('+', '_').replace(' ', '_')
                    metrics[f"{prefix}_Pixel_ROCAUC"] = _round_or_none(combo_metrics.get('rocauc', None))
                    metrics[f"{prefix}_Pixel_F1"] = _round_or_none(combo_metrics.get('f1', None))
                    metrics[f"{prefix}_Pixel_AUPR"] = _round_or_none(combo_metrics.get('aupr', None))
                    metrics[f"{prefix}_Pixel_AP"] = _round_or_none(combo_metrics.get('ap', None))

            metrics_data.append(metrics)

            print(f'Method:{method_name}, Class: {class_name}, Image ROCAUC: {method.image_rocauc:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, RGB Pixel ROCAUC: {method.rgb_pixel_rocauc:.3f}, F1: {method.rgb_pixel_f1:.3f}, AUPR: {method.rgb_pixel_aupr:.3f}, AP: {method.rgb_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, Infra Pixel ROCAUC: {method.infra_pixel_rocauc:.3f}, F1: {method.infra_pixel_f1:.3f}, AUPR: {method.infra_pixel_aupr:.3f}, AP: {method.infra_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, PC Pixel ROCAUC: {method.pc_pixel_rocauc:.3f}, F1: {method.pc_pixel_f1:.3f}, AUPR: {method.pc_pixel_aupr:.3f}')

            if hasattr(method, 'fusion_pixel_rocauc'):
                print(f'Method:{method_name}, Class: {class_name}, Fusion Pixel ROCAUC: {method.fusion_pixel_rocauc:.3f}, F1: {method.fusion_pixel_f1:.3f}, AUPR: {method.fusion_pixel_aupr:.3f}, AP: {method.fusion_pixel_ap:.3f}')

            if hasattr(method, 'm3dm_dlf_pixel_rocauc'):
                print(f'Method:{method_name}, Class: {class_name}, M3DM DLF Pixel ROCAUC: {method.m3dm_dlf_pixel_rocauc:.3f}, F1: {method.m3dm_dlf_pixel_f1:.3f}, AUPR: {method.m3dm_dlf_pixel_aupr:.3f}, AP: {method.m3dm_dlf_pixel_ap:.3f}')

            # Keep the previous object-combo print.
            if hasattr(method, 'object_score_rocauc'):
                combo_text = ', '.join([
                    f'{name}: {auc:.3f}' for name, auc in method.object_score_rocauc.items()
                ])
                print(f'Method:{method_name}, Class: {class_name}, Object score AUROC combos: {combo_text}')

            # New pixel-combo print.
            if hasattr(method, 'pixel_combo_metrics'):
                print(
                    f'Method:{method_name}, Class: {class_name}, '
                    f'Pixel score combo metrics: {_format_metric_dict(method.pixel_combo_metrics)}'
                )

        return pd.DataFrame(metrics_data)
