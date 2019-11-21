# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },

        ##############################################
        # These ones are deprecated, should be removed
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        ##############################################
        "cityscapes_poly_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "poly",
        },
        "cityscapes_poly_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
        },
        "cityscapes_poly_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
            "mini": 10,
        },
        "cityscapes_mask_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "mask",
        },
        "cityscapes_mask_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
        },
        "cityscapes_mask_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
            "mini": 10,
        },
        "mosquitoes_cocostyle_noaug_rectified_DJI_0033": {
            "img_dir": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0033",
            "ann_file": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0033/coco_format_rectfied_DJI_0033.json"
        },
        "mosquitoes_cocostyle_noaug_rectified_DJI_0038": {
            "img_dir": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0038",
            "ann_file": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0038/coco_format_rectfied_DJI_0038.json"
        },
        "mosquitoes_cocostyle_noaug_rectified_DJI_0043": {
            "img_dir": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0043",
            "ann_file": "mosquitoes/data_aug/no_aug/Test/rectfied_DJI_0043/coco_format_rectfied_DJI_0043.json"
        },
        "mosquitoes_cocostyle_noaug_test": {
            "img_dir": "mosquitoes/data_aug/no_aug/Test",
            "ann_file": "mosquitoes/data_aug/no_aug/Test/coco_format_Test.json"
        },
        "mosquitoes_cocostyle_noaug_train": {
            "img_dir": "mosquitoes/data_aug/no_aug/Train",
            "ann_file": "mosquitoes/data_aug/no_aug/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_balanced_train": {
                "img_dir": "mosquitoes/data_aug/mix/Train",
                "ann_file": "mosquitoes/data_aug/mix/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_blur_test": {
            "img_dir": "mosquitoes/data_aug/blur/Test",
            "ann_file": "mosquitoes/data_aug/blur/Test/coco_format_Test.json"
        },
        "mosquitoes_cocostyle_blur_train": {
            "img_dir": "mosquitoes/data_aug/blur/Train",
            "ann_file": "mosquitoes/data_aug/blur/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_blend_test": {
            "img_dir": "mosquitoes/data_aug/blend/Test",
            "ann_file": "mosquitoes/data_aug/blend/Test/coco_format_Test.json"
        },
        "mosquitoes_cocostyle_blend_train": {
            "img_dir": "mosquitoes/data_aug/blend/Train",
            "ann_file": "mosquitoes/data_aug/blend/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_paste_test": {
            "img_dir": "mosquitoes/data_aug/paste/Test",
            "ann_file": "mosquitoes/data_aug/paste/Test/coco_format_Test.json"
        },
        "mosquitoes_cocostyle_paste_train": {
            "img_dir": "mosquitoes/data_aug/paste/Train",
            "ann_file": "mosquitoes/data_aug/paste/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_CEFET_train": {
            "img_dir": "mosquitoes/CEFET/VideoDataSet/5m/Train",
            "ann_file": "mosquitoes/CEFET/VideoDataSet/5m/Train/coco_format_Train.json"
        },
        "mosquitoes_cocostyle_CEFET_test": {
            "img_dir": "mosquitoes/CEFET/VideoDataSet/5m/Test",
            "ann_file": "mosquitoes/CEFET/VideoDataSet/5m/Test/coco_format_Test.json"
        },
        "mosquitoes_CEFET_train": ("mosquitoes/CEFET/VideoDataSet/5m/Train",
                                   'mosquitoes/CEFET/zframer-marcacoes'),
        "mosquitoes_CEFET_test": ("mosquitoes/CEFET/VideoDataSet/5m/Test",
                                  'mosquitoes/CEFET/zframer-marcacoes')
    }

    @staticmethod
    def get(name):
        if "mosquitoes_cocostyle" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="MosquitoesCOCODataset",
                args=args,
            )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
