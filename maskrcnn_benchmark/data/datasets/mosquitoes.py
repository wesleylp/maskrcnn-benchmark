import fnmatch
import os
import sys
# sys.path.append('')

import torch
from PIL import Image
from ..datasets.mosquitoes_utils.files_utils import Directory
from ..datasets.mosquitoes_utils.annotation import AnnotationImage

# from annotation import AnnotationImage
from maskrcnn_benchmark.structures.bounding_box import BoxList


class MosquitoDataset(object):
    CLASSES = ('__background__', 'tire')

    def __init__(self,
                 root_dir,
                 annotation_folder=None,
                 remove_images_without_annotations=True,
                 transforms=None):
        ext = ('.png')

        self.root_dir = root_dir
        self.annotation_folder = annotation_folder
        self.transforms = transforms
        self.all_frames = Directory.get_files(self.root_dir, ext, recursive=True)

        cls = MosquitoDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        # retain only frames with annotations
        if remove_images_without_annotations:
            frames_with_annotation = [
                frame for frame in self.all_frames if len(self.get_groundtruth(frame)[0]) > 0
            ]
            self.frames_list = frames_with_annotation

        else:
            self.frames_list = self.all_frames

    def __getitem__(self, idx):
        img_path = self.frames_list[idx]
        self.img = Image.open(open(img_path, 'rb'))

        boxes, labels = self.get_groundtruth(img_path)
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, self.img.size, mode="xyxy")

        # create a BoxList from the boxes
        # add the labels to the boxlist
        classes = [self.class_to_ind['tire'] for label in labels]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms:
            self.img, target = self.transforms(self.img, target)

        return self.img, target, idx

    def __len__(self):
        return len(self.frames_list)

    def get_groundtruth(self, img_path):
        annot_path = None
        if self.annotation_folder is not None:
            # look for annotation file
            annot_path = _find_annot_file(img_path, self.annotation_folder)

        if annot_path is not None:
            frame_number = _get_frame_number(img_path)
            annotation = AnnotationImage(frame_number, annot_path)
            boxes, labels = annotation.get_bboxes_labels()

            return boxes, labels

        return [], []

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        # img_height = self.img.height
        # img_width = self.img.width
        # TODO: hardcoded!
        img_height = 1080
        img_width = 1920

        return {"height": img_height, "width": img_width}


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
