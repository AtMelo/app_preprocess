# -*- coding: utf8 -*-
import random
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import albumentations as A
import cv2
import shutil


def plot(imgs, orig_img, with_orig=True, row_title=None, **imshow_kwargs):
    """
    Show Original (default) and transformed images
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False,
                            figsize=(100, 100))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


def setup_logger(name=None, level=logging.DEBUG):
    if name:
        logging.getLogger(name)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s %(filename)s:%(lineno)d] '
               '%(message)s',
        datefmt='%d.%m.%y %H:%M',
        encoding='utf-8'
    )


class Preprocess:
    def __init__(
            self,
            inp_path: str,
            preproc_path: str,
            dataset_path: str,
            class_name: str):

        self._check_params(inp_path,
                           preproc_path,
                           dataset_path,
                           class_name)

        self.inp_path = Path(inp_path)
        self.preproc_path = Path(preproc_path)
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name

        self.path_to_dir_cls = self.preproc_path / self.class_name
        self.img_dir = self.path_to_dir_cls / 'image'
        self.lbl_dir = self.path_to_dir_cls / 'labels'

        # Path to class dataset (save path for the augmentation)
        self.path_cls_DS = self.dataset_path / self.class_name
        self.path_img_DS = self.path_cls_DS / 'image'
        self.path_lbl_DS = self.path_cls_DS / 'labels'

    @staticmethod
    def _check_params(*args):
        assert None not in args, f'Specify initial parameters in config, ' \
                                 f'got {args}'

    def prepare_data(self):
        """
        Convert to RGB, rename image, move to other dir, duplicate empty txt
        file. Rename images in order as: {class_name}_i.jpeg
        """
        if list(self.lbl_dir.glob('*.txt')):
            raise RuntimeError('This directory contains txt files, '
                               'check the contents')
        if not self.path_to_dir_cls.exists():
            logging.debug(f'\t\tCreated dir: {self.path_to_dir_cls}')
        self.path_to_dir_cls.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        logging.debug(f'\t\tInput: {self.inp_path}\tOutput: {self.img_dir}')

        inp_files = os.listdir(self.inp_path)
        i = len(os.listdir(self.img_dir))  # if not empty
        logging.getLogger('PIL').setLevel(logging.WARNING)
        for file_i in tqdm(inp_files, ncols=79):
            i += 1
            with Image.open(self.inp_path / file_i) as img_i:
                img_i = img_i.convert('RGB')
                name_i = f'{self.class_name}_{i}'
                img_i.save(
                    self.img_dir / f'{name_i}.jpeg',
                    format='JPEG')
            with open(self.lbl_dir / f'{name_i}.txt', 'a'):
                pass

        logging.debug(f'\t\tRenamed {len(inp_files)} images')

    def expand_split_dataset(self,
                             num_transform: int,
                             size_train: float,
                             size_val: float,
                             size_test: float,
                             view=False):

        self._check_params(num_transform,
                           size_train,
                           size_val,
                           size_test)
        if not self.path_to_dir_cls.exists():
            raise NameError('This class name does not exist')
        # Augmentation Process
        images = os.listdir(self.img_dir)
        logging.debug('\t\tAugmentation process')
        for img_i in tqdm(images, ncols=79):
            path_img = self.img_dir / img_i
            self._augmentation(path_img, num_transform, view)

        # Split dataset train/val/test
        logging.debug('\t\tSplitting process')
        self._split_ds(size_train, size_val, size_test)
        return

    def _augmentation(self,
                      path_img: Union[str, Path],
                      num_trf: int,
                      view: bool):
        """
        Augmentation process. Expand dataset in num_trf times.
        """
        # Save path for the augmentation
        self.path_img_DS.mkdir(parents=True, exist_ok=True)
        self.path_lbl_DS.mkdir(parents=True, exist_ok=True)
        # Copy origin image to dataset dir
        shutil.copyfile(path_img, self.path_img_DS / path_img.name)
        # Find corresponding marking file
        path2lbl = self.lbl_dir / f'{path_img.stem}.txt'
        # Copy origin label.txt to dataset dir
        shutil.copyfile(path2lbl, self.path_lbl_DS / f'{path_img.stem}.txt')
        with open(path2lbl, 'r') as f:
            bb = []
            for line in f:
                line = list(map(float, line.strip().split(' ')))
                bb.append(line)

        all_trf_images = []
        image = cv2.imread(str(path_img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO format: ind_class, x_c, y_c, w, h
        b_boxes = np.array(bb)
        # Get ind_class column from bound boxes, transpose and take 1-Dimension
        if b_boxes.any():
            class_labels = np.transpose(b_boxes[:, :1].astype(int))[0]
            boxes = b_boxes[:, 1:]
        else:
            class_labels = []
            boxes = []

        transform = self._create_transforms(image)
        for i in range(num_trf):
            name_file = f'{path_img.stem}({i+1})'
            transformed = transform(image=image,
                                    bboxes=boxes,
                                    class_labels=class_labels)
            transformed_image = Image.fromarray(transformed['image'])
            transformed_image.save(
                self.path_img_DS / f'{name_file}.jpeg',
                format='JPEG'
            )
            transformed_bboxes = np.round(np.array(transformed['bboxes']), 6)
            transformed_lbls = transformed['class_labels']

            # Save augmented marking
            if not transformed_bboxes.any():
                # For img without object
                with open(self.path_lbl_DS / f'{name_file}.txt', 'a') as f:
                    pass
            for bbx in zip(transformed_lbls, transformed_bboxes):
                i_class = bbx[0]
                box = ' '.join(str(b) for b in bbx[1])
                msg = f'{i_class} {box}\n'
                with open(self.path_lbl_DS / f'{name_file}.txt', 'a') as f:
                    f.write(msg)
            if view:  # add transformed image
                all_trf_images.append(transformed['image'])

        # View result
        if view:
            plot(imgs=all_trf_images,
                 orig_img=image)

    @staticmethod
    def _create_transforms(img_i: np.ndarray,
                           percent_crop: Tuple[float] = (0.8, 0.8)):
        h, w, _ = img_i.shape
        crop_size = (
            int(percent_crop[0] * h),
            int(percent_crop[1] * w)
        )
        geometric_trfs = [
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.GaussNoise(var_limit=(1000, 1000), p=0.25),
            A.GaussianBlur(blur_limit=(17, 17), p=0.15),
            A.MotionBlur(blur_limit=31, allow_shifted=False, p=0.5),
            A.RandomRain(drop_color=(50, 50, 50), p=0.15),
            A.ToGray(p=0.2),
            A.PixelDropout(p=0.3),
            A.RandomCrop(height=crop_size[0], width=crop_size[1], p=0.75)
        ]

        transform = A.Compose(
            geometric_trfs,
            bbox_params=A.BboxParams(format='yolo',
                                     min_visibility=0.4,
                                     label_fields=['class_labels'],
                                     check_each_transform=True))
        return transform

    def _split_ds(self,
                  size_train: float,
                  size_val: float,
                  size_test: float):
        # Destination paths
        # Train dir
        p_train_im = self.path_cls_DS / 'train/images'
        p_train_lbl = self.path_cls_DS / 'train/labels'
        p_train_im.mkdir(parents=True, exist_ok=True)
        p_train_lbl.mkdir(parents=True, exist_ok=True)
        # Val dir
        p_val_im = self.path_cls_DS / 'val/images'
        p_val_lbl = self.path_cls_DS / 'val/labels'
        p_val_im.mkdir(parents=True, exist_ok=True)
        p_val_lbl.mkdir(parents=True, exist_ok=True)
        # Test dir
        p_test_im = self.path_cls_DS / 'test/images'
        p_test_lbl = self.path_cls_DS / 'test/labels'
        p_test_im.mkdir(parents=True, exist_ok=True)
        p_test_lbl.mkdir(parents=True, exist_ok=True)

        # Splitting process
        files = os.listdir(self.path_img_DS)
        random.shuffle(files)
        files = np.array(files)
        train, val, test = np.split(
            files,
            [int(size_train * len(files)),
             int((size_train + size_val) * len(files))]
        )
        train, val, test = list(train), list(val), list(test)

        # Source paths to files
        train_img = [self.path_img_DS / img_i for img_i in train]
        val_img = [self.path_img_DS / img_i for img_i in val]
        test_img = [self.path_img_DS / img_i for img_i in test]

        # Move files to dirs
        for path_file_i in train_img:
            # For image
            shutil.move(src=path_file_i,
                        dst=p_train_im)
            # For label
            name = Path(path_file_i).stem
            shutil.move(src=self.path_lbl_DS / f'{name}.txt',
                        dst=p_train_lbl)
        for path_file_i in val_img:
            # For image
            shutil.move(src=path_file_i,
                        dst=p_val_im)
            # For label
            name = Path(path_file_i).stem
            shutil.move(src=self.path_lbl_DS / f'{name}.txt',
                        dst=p_val_lbl)
        for path_file_i in test_img:
            # For image
            shutil.move(src=path_file_i,
                        dst=p_test_im)
            # For label
            name = Path(path_file_i).stem
            shutil.move(src=self.path_lbl_DS / f'{name}.txt',
                        dst=p_test_lbl)

        # Delete previous save dir
        shutil.rmtree(self.path_img_DS)
        shutil.rmtree(self.path_lbl_DS)

    def update_classes_txt(self,
                           names_cls: List[str]):
        """
        When adding another class, you need to update the text files in the
        entire dataset.
        """
        self._check_params(names_cls)
        for root, dirs, files in os.walk(self.dataset_path):
            name_root_dir = root.split('/')[-1]
            if name_root_dir == 'labels':
                with open(os.path.join(root, 'classes.txt'), 'w') as f:
                    for name_cls_i in names_cls:
                        f.write(name_cls_i + '\n')

                logging.debug(f'\t\tUpdated classes.txt for: {root}')







