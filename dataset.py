'''
    The script reads all necessary data on the disk and keep them on the Main Memory as numpy arrays.

    Pre-processing:
        * Subtract mean and divide by std
        * No shuffle, no flip, no resize ...
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import glob

class dataset():
    def __init__(self, params):
        '''
        :param params(dict) keys:
            mode: 'train', 'test'
            root_dir: coarse, fine and real folders under the root dir.
            batch: 64
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._batch = params.get('batch', 0)
        self._root_dir = params.get('root_dir', None)
        self._categories = ['ape','benchvise','cam','cat','duck']
        self._trainset = None
        self._testset = None
        self._dbset = None

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode!")
        else:
            if self._mode != 'train' and self._mode != 'test':
                sys.exit("Invalid mode. Must be either 'train' or 'test'.")
        if self._batch == 0:
            sys.exit("Batch size {} is not allowed!".format(self._batch))
        if self._root_dir is None:
            sys.exit("Must specify a dataset root.")

        # read data and store as numpy arrays
        self._trainset = self._get_trainset()
        self._testset = self._get_testset()
        self._dbset = self._get_dbset()

        print("Read data complete.")

    def _get_trainset(self):
        '''
        :returns a dict with 5 keys: [0,1,2,3,4] corresponding to 5 obj categories
        '''
        trainset = []
        for i in range(5):
            trainset[i] = []
            category = self._categories[i]
            # fine set
            imgs_path = os.path.join(self._root_dir, 'fine', category, '*.png')
            pose_txt = os.path.join(self._root_dir, 'fine', category, 'poses.txt')
            poses = self._get_poses(pose_txt) # list of numpy arrays
            files_img = glob.glob(imgs_path)
            files_img.sort()
            num_imgs = len(files_img)
            for j in range(num_imgs):
                img = self._read_img(files_img[j]) # numpy array [64,64,3], np.float32
                pose = poses[j] # numpy array [4,], np.float32
                trainset[i].append((img, pose))
            # real set




    def _read_img(self, img_path):
        '''
        :param img_path: e.g. /path_to_img/1.png
        :return: numpy array [64,64,3], np.float32, standardized
        '''
        img = Image.open(img_path)
        img_arr = np.array(img, dtype=np.float32)

        # TODO: standardize

        return img_arr

    def _get_poses(self, pose_path):
        '''
        :param pose_path: e.g. /path_to_pose/poses.txt
        :return: numpy list: [pose0, pose1, pose2, ...], where pose is a numpy array of len 4, np.float32
        '''
        poses = []
        with open(pose_path) as t:
            raw = t.read().splitlines()
            num_raw = len(raw)
            for i in range(num_raw):
                if i % 2 != 0:
                    pose_raw = raw[i].split(' ')
                    pose_v = []
                    for j in range(4):
                        pose_v.append(float(pose_raw[j]))
                    poses.append(np.array(pose_v, np.float32))
            if len(poses) != num_raw / 2:
                sys.exit("Error reading poses. Got {} raw lines, {} numeric poses".format(num_raw, len(poses)))

        return poses








