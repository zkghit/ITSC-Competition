import caffe
import pdb
import numpy as np
from PIL import Image
import random
import scipy.io


class FaceRecogDataLayer(caffe.Layer):

    # def __init__(self):
    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.faceImage_dir = params['faceImage_dir']
        self.dataLabel_dir = params['label_dir']
        self.landMark_dir = params['landMark_dir']
        self.pose_dir = params['pose_dir']
        self.batch = int(params['batch'])
        self.mean = np.array(params['mean'])
        self.totalImage = int(params['totalImage'])
        self.image_dimension = int(params['image_dimension'])
        self.image_size_1 = int(params['image_size_1'])
        self.image_size_2 = int(params['image_size_2'])
        self.landMark_dim = int(params['landMark_dim'])
        self.pose_dim = int(params['pose_dim'])
        self.track = 0
        self.list_name = params['list_name']
        # three tops: data, label and landmark
        if len(top) != 4:
            raise Exception("Need to define three tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load label mapping
        split_f = '{}/{}.txt'.format(self.dataLabel_dir,
                                     self.list_name)
        split_Mapping = '{}/{}.txt'.format(self.dataLabel_dir, 'SubID_labelMappingFull')
        self.indices_Mapping = open(split_Mapping, 'r').read().splitlines()
        self.id_label_Mapping = {}
        for item in self.indices_Mapping:
            self.id_label_Mapping[item.split()[0]] = item.split()[1]

        self.indices = open(split_f, 'r').read().splitlines()
        random.shuffle(self.indices)
        # print(self.indices)
        # print(len(self.indices))
        # print(self.totalImage)

        self.data = np.zeros(shape=(self.batch, self.image_dimension, self.image_size_1, self.image_size_2))
        # print(self.data.shape)
        # self.label = np.zeros(shape=(self.batch, 2))       ## pair
        self.label = np.zeros(shape=(self.batch))  ## single image
        self.landMark = np.zeros(shape=(self.batch, self.landMark_dim))
        self.pose_info = np.zeros(shape=(self.batch, self.pose_dim))
        # print(self.label.shape)
        # pdb.set_trace()
        for num in range(0, self.batch):
            # print(self.batch)
            # print(num)
            # pdb.set_trace()
            self.idx = random.randint(1, self.totalImage)
            print('random number: ', self.idx)
            self.data[num] = self.load_image('{}/{}'.format(self.faceImage_dir, self.indices[self.idx].split()[0]))
            print('load image: ' + '{}/{}'.format(self.faceImage_dir, self.indices[self.idx].split()[0]))
            print(self.data[num].shape)
            self.label[num] = int(int(self.id_label_Mapping[self.indices[self.idx].split()[1]]) - 1 - 531)
            print('image label: ', self.label[num])
            self.landMark[num] = self.load_landmark(
                '{}/{}'.format(self.landMark_dir, self.indices[self.idx].split()[2]))
            print('load landmark: ' + '{}/{}'.format(self.landMark_dir, self.indices[self.idx].split()[2]))
            self.pose_info[num] = self.load_pose('{}/{}'.format(self.pose_dir, self.indices[self.idx].split()[3]))
            print('load pose: ' + '{}/{}'.format(self.pose_dir, self.indices[self.idx].split()[3]))

        # print(self.data.shape)
        # change data format
        self.data = np.array(self.data, dtype=np.float32)
        self.label = np.array(self.label, dtype=np.int)
        self.landMark = np.array(self.landMark, dtype=np.float32)
        self.pose_info = np.array(self.pose_info, dtype=np.float32)
        # print(self.data.shape)

    def forward(self, bottom, top):
        # assign output
        # self.reshape(self.bottom, self.top)
        # print("forward time ", self.track)
        if self.track > self.totalImage - self.track - 1:
            self.track = 0
            print("one epoch finished!")
        for num in range(self.track, self.track + self.batch):
            # print(self.batch)
            # print(num)
            # pdb.set_trace()
            # self.idx = random.randint(1, self.totalImage)
            # print('random number: ', self.track)
            self.data[num-self.track] = self.load_image('{}/{}'.format(self.faceImage_dir, self.indices[num].split()[0]))
            # print('load image: '+'{}/{}'.format(self.faceImage_dir, self.indices[num].split()[0]))
            self.label[num-self.track] = int(int(self.id_label_Mapping[self.indices[num].split()[1]]) - 1 -531)
            # print('image label: ', self.label[num-self.track])
            self.landMark[num-self.track] = self.load_landmark(
                '{}/{}'.format(self.landMark_dir, self.indices[num].split()[2]))
            # print('load landmark: '+'{}/{}'.format(self.landMark_dir, self.indices[self.track].split()[2]))
            self.pose_info[num-self.track] = self.load_pose('{}/{}'.format(self.pose_dir, self.indices[num].split()[3]))
            # print('load pose: '+'{}/{}'.format(self.pose_dir, self.indices[self.track].split()[3]))

        # print(self.data.shape)
        # change data format
        self.data = np.array(self.data, dtype=np.float32)
        self.label = np.array(self.label, dtype=np.int)
        self.landMark = np.array(self.landMark, dtype=np.float32)
        self.pose_info = np.array(self.pose_info, dtype=np.float32)
        self.track = self.track + self.batch

        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.landMark
        top[3].data[...] = self.pose_info
        # print("forward to produce data, label, landmark and pose info")

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)
        top[2].reshape(*self.landMark.shape)
        top[3].reshape(*self.pose_info.shape)
        pass

    def load_image(self, img_name):

        # self.faceImage_dir = params['faceImage_dir']
        im = Image.open(img_name)
        # im = im.resize((227, 227), Image.ANTIALIAS)
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape) != 3:
            in_rgb = np.zeros(shape=(3, 224, 224), dtype=float)
            in_rgb[0, :, :] = in_
            in_rgb[1, :, :] = in_
            in_rgb[2, :, :] = in_
            return in_rgb

        in_ = in_[:, :, ::-1]
        # in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_landmark(self, file_name):

        landmark_dict = scipy.io.loadmat(file_name)
        landmark_x = np.array(landmark_dict['x'], dtype=np.float32)
        landmark_y = np.array(landmark_dict['y'], dtype=np.float32)
        landmark_xy = np.concatenate((landmark_x[0], landmark_y[0]))
        landmark_xys = landmark_xy.squeeze()
        return landmark_xys

    def load_pose(self, file_name):

        pose_dict = scipy.io.loadmat(file_name)
        pose_info = np.array(pose_dict['Pose_Para'], dtype=np.float32)
        pose_info = pose_info[0].squeeze()
        return pose_info
