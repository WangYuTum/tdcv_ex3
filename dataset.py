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
from random import sample
from random import randint

class model_data():
    def __init__(self, params):
        '''
        :param params(dict) keys:
            mode: 'train', 'test'
            root_dir: coarse, fine and real folders under the root dir.
            batch: 64
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._root_dir = params.get('root_dir', None)
        self._batch = params.get('batch', 0)
        self._categories = ['ape','benchvise','cam','cat','duck']
        self._trainset = None
        self._testset = None
        self._dbset = None
        self._data_mean = np.array([63.9665267098, 54.8146651153, 48.0492310805]).reshape((1, 1, 3))
        self._data_std = np.array([69.0266489691, 59.7451054095, 55.8269048247]).reshape((1, 1, 3))

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode!")
        else:
            if self._mode != 'train' and self._mode != 'test':
                sys.exit("Invalid mode. Must be either 'train' or 'test'.")
        if self._batch == 0:
            sys.exit("Must specify a batch >= 1.")
        if self._root_dir is None:
            sys.exit("Must specify a dataset root.")

        # read data and store as numpy arrays
        self._trainset = self._get_trainset()
        self._testset = self._get_testset()
        self._dbset = self._get_dbset()

        print("Read data complete.")

    def _get_trainset(self):
        '''
        :returns a dict with 5 keys: [0,1,2,3,4] corresponding to 5 obj categories.
                 each key maps to a list of numpy tuples: (img, pose), np.float32
                 img has shape [64, 64, 3], pose has shape [4,]
        '''
        trainset = {}
        for i in range(5):
            trainset[i] = []
            category = self._categories[i]
            # fine set
            files_img = self._get_img_filelist('fine', category)
            poses = self._get_poses('fine', category)  # list of numpy arrays
            num_imgs = len(files_img)
            for j in range(num_imgs):
                img = self._read_img(files_img[j], 'fine') # numpy array [64,64,3], np.float32
                pose = poses[j] # numpy array [4,], np.float32
                trainset[i].append((img, pose, files_img[j]))
                print('img: {}, pose: {}'.format(files_img[j], pose))
            # real set
            list_ids = self._get_trainlist() # a list of training ids
            files_img = self._get_img_filelist('real', category)
            poses = self._get_poses('real', category)  # list of numpy arrays
            for train_id in list_ids:
                img = self._read_img(files_img[train_id], 'real')  # numpy array [64,64,3], np.float32
                pose = poses[train_id]  # numpy array [4,], np.float32
                trainset[i].append((img, pose, files_img[train_id]))

        return trainset

    def _get_testset(self):
        '''
        :return: a dict which has the same format returned from self._get_trainset()
        '''
        testset = {}
        for i in range(5):
            testset[i] = []
            category = self._categories[i]
            # real set
            files_img = self._get_img_filelist('real', category)
            poses = self._get_poses('real', category)  # list of numpy arrays
            list_ids = self._get_testlist(len(files_img))  # a list of testing ids
            for test_id in list_ids:
                if test_id > 1177:  # tricky one because of dirty data
                    break
                img = self._read_img(files_img[test_id], 'real')
                pose = poses[test_id]
                testset[i].append((img, pose))

        return testset

    def _get_dbset(self):
        '''
        :return: a dict which has the same format returned from self._get_trainset()
        '''
        dbset = {}
        for i in range(5):
            dbset[i] = []
            category = self._categories[i]
            # coarse set
            files_img = self._get_img_filelist('coarse', category)
            poses = self._get_poses('coarse', category)  # list of numpy arrays
            num_imgs= len(files_img)
            for j in range(num_imgs):
                img = self._read_img(files_img[j], 'coarse')
                pose = poses[j]
                dbset[i].append((img, pose, files_img[j]))

        return dbset

    def _get_img_filelist(self, subset, category):
        '''
        :param subset: 'coarse', 'fine', 'real'
        :param category: 'ape', 'benchvise', 'cam', 'cat', 'duck'
        :return:  a list of img files, a
        '''
        imgs_path = os.path.join(self._root_dir, subset, category, '*.png')
        files_img = glob.glob(imgs_path)
        # remove prefix
        num_imgs = len(files_img)
        for i in range(num_imgs):
            old_name = files_img[i].split('/')[-1]
            new_name = old_name.replace(subset, '')
            files_img[i] = files_img[i].replace(old_name, new_name)
        files_img.sort()

        return files_img

    def _read_img(self, img_path, prefix):
        '''
        :param img_path: e.g. /path_to_img/1.png
        :return: numpy array [64,64,3], np.float32, standardized
        '''
        num_s = img_path.split('/')[-1].split('.')[0]
        new_name = prefix + num_s
        img_path = img_path.replace(num_s, new_name)
        img = Image.open(img_path)
        img_arr = np.array(img, dtype=np.float32)

        # img stardardize
        img_arr = self._std_img(img_arr)

        return  img_arr

    def _std_img(self, img):
        '''
        :param img: numpy array [64,64,3], np.float32
        :return: standardized img
        '''

        img -= self._data_mean
        img /= self._data_std

        return img

    def _get_poses(self, subset, category):
        '''
        :param subset: 'coarse', 'fine', 'real'
        :param category: 'ape', 'benchvise', 'cam', 'cat', 'duck'
        :return: numpy list: [pose0, pose1, pose2, ...], where pose is a numpy array of len 4, np.float32
        '''
        pose_path = os.path.join(self._root_dir, subset, category, 'poses.txt')

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
                sys.exit("Error reading poses. Got {0} raw lines, {1} numeric poses, {2}, {3}".format(num_raw, len(poses), subset, category))

        return poses

    def _get_trainlist(self):

        list_txt = os.path.join(self._root_dir, 'real', 'training_split.txt')
        list_ids = []
        with open(list_txt) as t:
            raw = t.read().splitlines()
            raw_list = raw[0].split(', ')
            len_list = len(raw_list)
            for i in range(len_list):
                list_ids.append(int(raw_list[i]))

        return  list_ids

    def _get_testlist(self, total_len):
        '''
        :param total_len: total number of traintest files
        :return: a list of test ids, ordered
        '''
        total_list = range(total_len)
        trainlist = self._get_trainlist()
        total_set = set(total_list)
        train_set = set(trainlist)
        test_set = total_set.difference(train_set)
        test_list = list(test_set)
        test_list.sort()

        return test_list

    # user API
    def get_trainset(self):

        return self._trainset

    def get_testset(self):

        return self._testset

    def get_dbset(self):

        return self._dbset

    def next_batch(self):
        '''
        :return: a batch [3*batch, 64, 64, 3].
                 [0:N, 64, 64, 3] are anchor images
                 [N:2*N, 64, 64, 3] are positives(pullers)
                 [2*N:3*N, 64, 64, 3] are negatives(pushers)
        '''

        anchor_idx = [] # each element is a tuple: (category_id, img_id)
        for batch_id in range(self._batch):
            category_id = randint(0, 4)
            num_imgs = len(self._trainset[category_id])
            img_id = randint(0, num_imgs-1)
            anchor_idx.append((category_id, img_id))
        # get anchor data, from train set
        batch_data = self._trainset[anchor_idx[0][0]][anchor_idx[0][1]][0] # np.float32, [64,64,3]
        batch_data = np.expand_dims(batch_data, 0) # [1,64,64,3]
        for batch_id in range(self._batch-1):
            id = batch_id + 1
            img_data = self._trainset[anchor_idx[id][0]][anchor_idx[id][1]][0]
            img_data = np.expand_dims(img_data, 0)
            batch_data = np.concatenate((batch_data, img_data), 0) # [batch, 64, 64, 3] only for anchors
        # get positive data, from db set
        for batch_id in range(self._batch):
            category_id = anchor_idx[batch_id][0]
            img_id = anchor_idx[batch_id][1]
            image = self._get_puller(category_id, img_id)
            image = np.expand_dims(image, 0)
            batch_data = np.concatenate((batch_data, image), 0) # [2*batch, 64, 64, 3] anchors and positives
        # get negative data, from db set
        for batch_id in range(self._batch):
            # choose same obj or not
            obj_bool = randint(0,1)
            if obj_bool == 0:
                # choose the same obj, diff pose
                category_id = anchor_idx[batch_id][0]
                img_id = anchor_idx[batch_id][1]
                image = self._get_pusher(category_id, 'same', img_id)
                image = np.expand_dims(image, 0)
                batch_data = np.concatenate((batch_data, image), 0) # [3*batch, 64, 64, 3] anchors and positives
            else:
                # choose diff obj
                category_id = anchor_idx[batch_id][0]
                image = self._get_pusher(category_id, 'diff', 0)
                image = np.expand_dims(image, 0)
                batch_data = np.concatenate((batch_data, image), 0) # [3*batch, 64, 64, 3] anchors and positives

        # check shape
        batch_shape = batch_data.shape
        if batch_shape[0] != self._batch*3:
            sys.exit("Batch data shape 0 invalid: {}, should be {}".format(batch_shape[0], self._batch*3))

        return batch_data

    def _get_pusher(self, category_id, mode, img_id):
        '''
        :param category_id: the category id of anchor
        :param mode: either 'same' or 'diff'
        :param img_id: only useful when mode is 'same'
        :return: the pusher of the anchor
        '''
        if mode == 'diff':
            # choose random image from dbset with a diff obj id
            rand_cat = randint(0,4)
            while rand_cat == category_id:
                rand_cat = randint(0, 4)
            num_imgs = len(self._dbset[rand_cat])
            rand_img = randint(0, num_imgs-1)
            image = self._dbset[rand_cat][rand_img][0] # np.float32, [64,64,3]

            return image
        else:
            # choose random image from dbset of the same obj id, but diff pose
            pose_anchor = self._trainset[category_id][img_id][1] # np.float32, shape (4,)
            num_imgs = len(self._dbset[category_id])
            metric_list = [] # [(metric, img_id), (metric, img_id), ...]
            for i in range(num_imgs):
                pose_other = self._dbset[category_id][i][1]
                metric_list.append((self._compute_metric(pose_anchor, pose_other), i))
            # the smaller the metric, the more similar
            metric_list = sorted(metric_list, reverse=False)
            pusher_rand = randint(1, num_imgs-1) # a diff pose as long as not the most similar one
            pusher_id = metric_list[pusher_rand][1]
            image = self._dbset[category_id][pusher_id][0] # np.float32, [64,64,3]

            return image

    def _get_puller(self, category_id, img_id):
        '''
        :param category_id: the category id of anchor
        :param img_id: the img id of anchor
        :return: the puller of the corresponding anchor image
        '''
        pose_anchor = self._trainset[category_id][img_id][1] # np.float32, shape (4,)
        num_imgs = len(self._dbset[category_id])
        metric_list = [] # [(metric, img_id), (metric, img_id), ...]
        for i in range(num_imgs):
            pose_other = self._dbset[category_id][i][1]
            metric_list.append((self._compute_metric(pose_anchor, pose_other),i))
        # the smaller the metric, the more similar
        metric_list = sorted(metric_list, reverse=False)
        puller_id = metric_list[0][1]
        image = self._dbset[category_id][puller_id][0] # np.float32, [64,64,3]

        return image

    def _compute_metric(self, pose1, pose2):
        '''
        :param pose1: parameterized by quaternions
        :param pose2: parameterized by quaternions
        :return: np.float32
        '''

        metric_v = 2 * np.arccos(np.absolute(np.dot(pose1, pose2)))

        return metric_v











