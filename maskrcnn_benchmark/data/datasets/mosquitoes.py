import fnmatch
import os

import torch
from PIL import Image

# from annotation import AnnotationImage
from maskrcnn_benchmark.structures.bounding_box import BoxList

# from utils.files_utils import Directory


class MosquitoDataset(object):
    def __init__(self, root_dir, annotation_folder=None, transforms=None):
        ext = ('.png')
        self.root_dir = root_dir
        self.frames_list = Directory.get_files(self.root_dir, ext, recursive=True)
        self.annotation_folder = annotation_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        img_path = self.frames_list[idx]
        self.img = Image.open(open(img_path, 'rb'))

        annot_path = None
        if self.annotation_folder is not None:
            # look for annotation file
            annot_path = _find_annot_file(self.frames_list[idx], self.annotation_folder)
            print('******{}********'.format(annot_path))

        if annot_path is not None:
            frame_number = _get_frame_number(self.frames_list[idx])
            annotation = AnnotationImage(frame_number, annot_path)
            boxes, classes = annotation.get_bboxes_labels()

        # TODO: Make a better decision about this
        else:
            boxes, classes = [], [0]
            print('Annotation not found: ', self.frames_list[idx])

        # TODO: check and fix it, if necessary
        labels = torch.ones(len(classes))

        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # create a BoxList from the boxes
        target = BoxList(boxes, self.img.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        if self.transforms:
            self.img, target = self.transforms(self.img, target)

        return self.img, target, idx

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


class Annotation:
    def __init__(self, annotation_path=None, total_frames=None, encoding='ISO-8859-1'):
        self.total_frames = total_frames
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False

    def _parse_file(self):
        if (self.annotation_path is None) or (os.path.exists(self.annotation_path) is False):
            return False

        # create dictonary with number of frames
        self.annotation_dict = {'frame_{:04d}'.format(d): {} for d in range(self.total_frames)}

        # reading annotation file
        with open(self.annotation_path, encoding=self.encoding) as annotation_file:

            # reading line
            for line in annotation_file:

                if len(line) == 0:
                    continue

                if 'NAME' in line:
                    object_name = line.strip().split(':', 1)[-1]
                    continue

                if 'RECT' in line:
                    frame = line.strip().split(None, 5)
                    frame = [s.replace(',', '') for s in frame]
                    frame_idx = int(frame[1])
                    bb = [int(frame[p]) for p in list(range(2, 6))]

                    self.annotation_dict['frame_{:04d}'.format(frame_idx)][object_name] = bb
                    continue
        annotation_file.close()

        self.parsed = True
        self.error = False

        return True

    def is_valid(self):
        if self.parsed is False:
            return self._parse_file()
        else:
            return self.error

    def get_annoted_frame(self, frame_idx):
        return self.annotation_dict['frame_{:04d}'.format(frame_idx)]


class AnnotationImage(object):
    def __init__(self, frame_number, annotation_path=None, encoding='ISO-8859-1'):
        # self.total_frames = total_frames
        self.frame_number = frame_number
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False

    def _parse_file(self):
        if (self.annotation_path is None) or (os.path.exists(self.annotation_path) is False):
            return False

        # create dictonary with number of frames
        # self.annotation_dict = {'frame_{:04d}'.format(d): {} for d in range(self.total_frames)}

        # reading annotation file
        with open(self.annotation_path, encoding=self.encoding) as annotation_file:

            # reading line
            for line in annotation_file:

                if len(line) == 0:
                    continue

                if 'NAME' in line:
                    object_name = line.strip().split(':', 1)[-1]
                    continue

                if 'RECT' in line:
                    frame = line.strip().split(None, 5)
                    frame = [s.replace(',', '') for s in frame]
                    frame_idx = int(frame[1])

                    # print(type(frame_idx))
                    # print(self.frame_number)
                    if frame_idx == self.frame_number:
                        bb = [int(frame[p]) for p in list(range(2, 6))]
                        self.annotation_dict['{}'.format(object_name)] = bb
                    continue

        annotation_file.close()

        self.parsed = True
        self.error = False

        return True

    def is_valid(self):
        if self.parsed is False:
            return self._parse_file()
        else:
            return self.error

    def parse_annotation(self):
        return self._parse_file()

    def get_annotations(self):
        if self.parsed is False:
            self.parse_annotation()
        return self.annotation_dict

    def get_bboxes_labels(self):
        annotation_dict = self.get_annotations()
        bboxes = [bbox for bbox in annotation_dict.values()]
        labels = [obj for obj in annotation_dict.keys()]
        return bboxes, labels


class Directory(object):
    @staticmethod
    def get_files(root_dir, ext, recursive=False):
        """Get files in the root_dir

        Arguments:
            root_dir {str} -- String directory to search for files.
            ext {tuple} -- Tuple with extensions desired.

        Keyword Arguments:
            recursive {bool} -- Flag indicating whether to search files recursively or not. (default: {False})

        Returns:
            list -- list containing files pathes.
        """

        files = []

        if recursive:

            for (dirpath, dirnames, filenames) in os.walk(root_dir):

                if len(filenames) == 0:
                    continue

                # getting only the  files with desired extensions
                [
                    files.append(os.path.join(dirpath, s)) for s in filenames
                    if s.lower().endswith(ext)
                ]
        else:
            [files.append(s) for s in os.listdir(root_dir) if s.lower().endswith(ext)]

        return files


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
