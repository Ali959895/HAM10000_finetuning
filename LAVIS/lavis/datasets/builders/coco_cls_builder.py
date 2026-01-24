from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry
from lavis.datasets.datasets.coco_cls_dataset import CocoClassificationDataset


@registry.register_builder("coco_classification")
class CocoClassificationBuilder(BaseDatasetBuilder):
    train_dataset_cls = CocoClassificationDataset
    eval_dataset_cls = CocoClassificationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/coco_classification.yaml"
    }

    def build_datasets(self):
        # Skip BaseDatasetBuilder downloading since data is local
        train_vis_root = self.config.build_info.images.storage
        train_ann_file = self.config.build_info.annotations[0].storage

        eval_vis_root = self.config.build_info_eval.images.storage
        eval_ann_file = self.config.build_info_eval.annotations[0].storage

        datasets = {}
        datasets["train"] = self.train_dataset_cls(
            vis_root=train_vis_root,
            ann_file=train_ann_file,
            transform=self.vis_processors["train"],
        )
        datasets["val"] = self.eval_dataset_cls(
            vis_root=eval_vis_root,
            ann_file=eval_ann_file,
            transform=self.vis_processors["eval"],
        )
        return datasets

