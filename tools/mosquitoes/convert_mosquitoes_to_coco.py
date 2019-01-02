import argparse
import fnmatch
import json
import os
import sys

from PIL import Image

from boxes import box_area, clip_box_to_image, xyxy_to_xywh

try:
    sys.path.append('../..')
    from maskrcnn_benchmark.data.datasets.mosquitoes_utils.annotation import \
        AnnotationImage
except:
    raise

train_videos = [
    'DJI00777',
    'DJI00800',
    'DJI00807',
]

test_videos = [
    'DJI00804',
    'DJI00806',
    'DJI00810',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    # parser.add_argument('--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir',
        help="output dir for png files",
        default='../../../../data/CEFET/VideoDataSet/5m',
        type=str)

    parser.add_argument(
        '--anndir',
        help="annotaion dir for txt files",
        default='../../../../data/CEFET/zframer-marcacoes',
        type=str)

    parser.add_argument(
        '--datadir',
        help="data dir for annotations to be converted",
        default='../../../../data/CEFET/VideoDataSet/5m',
        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def convert_mosquitoes_instance_only(data_dir, ann_dir, out_dir):

    sets = [
        'Train',
        'Test',
    ]

    categories = [
        '__background__',
        'tire',
    ]

    json_name = 'coco_format_%s.json'

    # ann_dir = 'zframer-marcacoes'

    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {
        '__background__': 0,
        'tire': 1,
    }

    for data_set in sets:
        print(f'Starting {data_set}')

        ann_dict = {}
        images = []
        annotations = []
        # ann_dir = os.path.join(data_dir, ann_dir)
        print(data_dir)
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(data_dir, data_set)):

            if len(filenames) == 0:
                continue

            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    image = {}

                    img_name_short = os.path.join(os.path.split(dirpath)[-1], filename)
                    img_name = os.path.join(dirpath, filename)

                    width = get_img_size(img_name)[0]
                    height = get_img_size(img_name)[1]

                    image['file_name'] = img_name_short
                    # image['seg_file_name'] = img_name
                    image['width'] = width
                    image['height'] = height
                    image['id'] = img_id
                    img_id += 1

                    for box, labels in zip(*get_groundtruth(img_name, ann_dir)):
                        # trim bbox for img limits

                        box = clip_box_to_image(box, height, width)

                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']

                        ann['category_id'] = 1
                        ann['iscrowd'] = 0
                        ann['area'] = box_area(box)
                        ann['bbox'] = xyxy_to_xywh(box)
                        # just to save segmentation
                        # ann['segmentation'] = ann['bbox']

                        annotations.append(ann)
                    images.append(image)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))

        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))
    return


def get_img_size(datapath):
    img = Image.open(datapath)
    width, height = img.size
    return width, height


def get_groundtruth(img_path, annotation_folder):
    annot_path = None
    if annotation_folder is not None:
        # look for annotation file
        annot_path = _find_annot_file(img_path, annotation_folder)

    if annot_path is not None:
        frame_number = _get_frame_number(img_path)
        annotation = AnnotationImage(frame_number, annot_path)
        boxes, labels = annotation.get_bboxes_labels()

        return boxes, labels

    return [], []


def _find_annot_file(frame_path, annot_folder):
    """Find annotation file based on video name.

    Arguments:
        frame_path {str} -- video path
        annot_folder {str} -- folder where to look in order to find the annotation file

    Returns:
        str -- [The annotation file path]
    """

    vid_filename = frame_path.split('/')[-2]
    # vid_filename, vid_ext = os.path.splitext(vid_filename)
    found = False

    for (dirpath, dirnames, filenames) in os.walk(annot_folder):

        if len(filenames) == 0:
            continue

        for file_name in filenames:
            if fnmatch.fnmatch(file_name, vid_filename + '.txt'):
                annot_path = os.path.join(dirpath, file_name)
                found = True
                break
        if found:
            return annot_path

    return


def _get_frame_number(frame_path):
    frame_filename = os.path.split(frame_path)[-1]
    frame_filename, vid_ext = os.path.splitext(frame_filename)

    frame_number = frame_filename.split('_')[-1]

    return int(frame_number)


if __name__ == '__main__':
    args = parse_args()
    # if args.dataset == "cityscapes_instance_only":
    convert_mosquitoes_instance_only(args.datadir, args.anndir, args.outdir)
    # else:
    #     print("Dataset not supported: %s" % args.dataset)
