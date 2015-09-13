# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os.path as osp
import sys
this_dir = osp.dirname(__file__)
this_dir = osp.join(this_dir,'..')
sys.path.insert(0,this_dir)

#import datasets
import datasets.ilsvrc_imagenet
import os
import datasets.imdb_imagenet
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class ilsvrc_imagenet(datasets.imdb_imagenet):
    def __init__(self, image_set, year, devkit_path=None):
#changed        
#        datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set)
        datasets.imdb_imagenet.__init__(self, 'ILSVRC' + year + '_' + 'DET' + \
                                '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
#changed
#        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._data_path = '/data/ILSVRC2013_DET/'
        
#changed
#        self._classes = ('__background__', # always index 0
#                         'aeroplane', 'bicycle', 'bird', 'boat',
#                         'bottle', 'bus', 'car', 'cat', 'chair',
#                         'cow', 'diningtable', 'dog', 'horse',
#                         'motorbike', 'person', 'pottedplant',
#                         'sheep', 'sofa', 'train', 'tvmonitor')

        self._classes = self._load_imagenet_label2words()
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
#changed
#        self._image_ext = '.jpg'
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

#changed
    def _load_imagenet_label2words(self):
        filename = '/data/ILSVRC2013_DET/label2words.mat'
        assert os.path.exists(filename), \
               'label2words not found at: {}'.format(filename)
        classes = sio.loadmat(filename)['words'].ravel()
        classes = tuple([str(classes[i][0]) for i in range(len(classes))])
        result = ('__background__',) + classes
        return result
        
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
#changed
        '''
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        '''
        image_path = os.path.join(self._data_path,'ILSVRC2013_DET_val', index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
#changed
#        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
#                                      self._image_set + '.txt')
#        assert os.path.exists(image_set_file), \
#                'Path does not exist: {}'.format(image_set_file)
#        with open(image_set_file) as f:
#            image_index = [x.strip() for x in f.readlines()]

#        change to load the image names because their names include 2012 and 2013

        image_set_dir = path = '/data/ILSVRC2013_DET/ILSVRC2013_DET_val'
        assert os.path.exists(image_set_dir), \
                'Path does not exist:{}'.format(image_set_dir)
        image_index = os.listdir(image_set_dir)
        
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
#changed
        #return os.path.join(datasets.ROOT_DIR, 'data', VOCdevkit' + self._year)
        return '/data/ILSVRC2013_DET/'

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
#changed
#       gt_roidb = [self._load_pascal_annotation(index)
#                    for index in self.image_index]:
        gt_roidb = self._load_ilsvrc2013det_val_annotation()   
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

#changed
 #       if int(self._year) == 2007 or self._image_set != 'test':
        if int(self._year) == 2013 or self._image_set != 'val':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
#changed
#        filename = os.path.abspath(os.path.join(self.cache_path, '..',
#                                                'selective_search_data',
#                                                self.name + '.mat'))
        
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                'ilsvrc13_val.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

#changed
#   def _load_pascal_annotation(self, index):
    def _load_ilsvrc2013det_val_annotation(self):
        """
        liuwenran 
        Load image and bounding boxes info from XML file in the ilsvrc
        format.,this function is only for ilsvrc2013det_val,it needs to finetune 
        to satisfy other split
        """
        gt_roidb = []
        ilsvrc2013det_val_info = sio.loadmat('ilsvrc2013det_val_info.mat')
        img_basenames = sio.loadmat('img_basenames.mat')
        img_basenames = img_basenames['img_basenames']
        allimg_nums = len(img_basenames)

        gt_obj_bboxes = ilsvrc2013det_val_info['gt_obj_bboxes']
        gt_obj_img_ids = ilsvrc2013det_val_info['gt_obj_img_ids']
        gt_obj_labels = ilsvrc2013det_val_info['gt_obj_labels']
        gt_obj_img_ids = gt_obj_img_ids[0];
        gt_obj_labels = gt_obj_labels[0]
        allobj_nums = len(gt_obj_img_ids)
        xmin = gt_obj_bboxes[0]
        ymin = gt_obj_bboxes[1]
        xmax = gt_obj_bboxes[2]
        ymax = gt_obj_bboxes[3]
        imgFlag = 1
        indexFlag = 0
        for i in range(allobj_nums):
            if i%200 == 0:
                print '_load_ilsvrc2013det_val_annotation' + str(i) + 'in' + str(allobj_nums)
            
            if i == allobj_nums - 1:
                pic_objnum = i + 1 - indexFlag
                boxes = np.zeros((pic_objnum, 4), dtype=np.uint16)
                gt_classes = np.zeros((pic_objnum), dtype=np.int32)
                overlaps = np.zeros((pic_objnum, self.num_classes), dtype=np.float32)
                k = 0
                for j in range(indexFlag,i+1):
                    x1 = float(xmin[j]) - 1
                    y1 = float(ymin[j]) - 1
                    x2 = float(xmax[j]) - 1
                    y2 = float(ymax[j]) - 1
                    cls = gt_obj_labels[j]
                    boxes[k,:] = [x1, y1, x2, y2]
                    gt_classes[k] = cls
                    overlaps[k, cls] = 1.0
                    k = k + 1
                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes': boxes,
                                 'gt_classes': gt_classes,
                                 'gt_overlaps': overlaps,
                                 'flipped': False})
#cause there is no object in last image(image20121),so need to add a blank roidb
                boxes = np.zeros((0, 4), dtype=np.uint16)
                gt_classes = np.zeros((0), dtype=np.int32)
                overlaps = np.zeros((0, self.num_classes), dtype=np.float32)
                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes': boxes,
                                 'gt_classes': gt_classes,
                                 'gt_overlaps': overlaps,
                                 'flipped': False})
                break
            
            if gt_obj_img_ids[i] == imgFlag:
                continue

            pic_objnum = i - indexFlag
            boxes = np.zeros((pic_objnum, 4), dtype=np.uint16)
            gt_classes = np.zeros((pic_objnum), dtype=np.int32)
            overlaps = np.zeros((pic_objnum, self.num_classes), dtype=np.float32)
            k = 0
            for j in range(indexFlag,i):
                x1 = float(xmin[j]) - 1
                y1 = float(ymin[j]) - 1
                x2 = float(xmax[j]) - 1
                y2 = float(ymax[j]) - 1
                cls = gt_obj_labels[j]
                boxes[k,:] = [x1, y1, x2, y2]
                gt_classes[k] = cls
                overlaps[k, cls] = 1.0
                k = k + 1
            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_roidb.append({'boxes': boxes,
                             'gt_classes': gt_classes,
                             'gt_overlaps': overlaps,
                             'flipped': False})	
#if there is no object in a image, we need to add blank roidb for this img
            diff = gt_obj_img_ids[i] - imgFlag
            for k in range(diff - 1):
                boxes = np.zeros((0, 4), dtype=np.uint16)
                gt_classes = np.zeros((0), dtype=np.int32)
                overlaps = np.zeros((0, self.num_classes), dtype=np.float32)
                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes': boxes,
                                 'gt_classes': gt_classes,
                                 'gt_overlaps': overlaps,
                                 'flipped': False})
            imgFlag = gt_obj_img_ids[i]
            indexFlag = i

        print 'all gt_roidb nums is   ' + 'str(len(gt_roidb))'
        return gt_roidb
#        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
#        gt_classes = np.zeros((num_objs), dtype=np.int32)
#        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
#
        # Load object bounding boxes into a data frame.
#        for ix, obj in enumerate(objs):
#            # Make pixel indexes 0-based
#            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
#            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
#            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
#            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
#            cls = self._class_to_ind[
#                    str(get_data_from_tag(obj, "name")).lower().strip()]
#            boxes[ix, :] = [x1, y1, x2, y2]
#            gt_classes[ix] = cls
#            overlaps[ix, cls] = 1.0
#
#        overlaps = scipy.sparse.csr_matrix(overlaps)
#
#        return {'boxes' : boxes,
#                'gt_classes': gt_classes,
#                'gt_overlaps' : overlaps,
#                'flipped' : False}

#changed
#    def _write_voc_results_file(self, all_boxes):
    def _write_ilsvrc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
#changed
        #path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
        #                    'Main', comp_id + '_')
        path = os.path.join(self.cache_path(), 'results', 'ILSVRC_' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
#changed
#            print 'Writing {} VOC results file'.format(cls)
            print 'Writing {} ILSVRC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
#changed
#        comp_id = self._write_voc_results_file(all_boxes)
        comp_id = self._write_ilsvrc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.ilsvrc_imagenet('trainval', '2013')
    res = d.roidb
#    from IPython import embed; embed()
