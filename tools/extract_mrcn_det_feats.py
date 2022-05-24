from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import json
import time
import numpy as np
import h5py
import cv2

import torch

import _init_paths
# dataloader
from loaders.dets_loader import DetsLoader

def main(args):
  dataset_splitBy = args.dataset
  if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
    os.makedirs(osp.join('cache/feats/', dataset_splitBy))

  # detection feature Directory
  if 'rsvg' in dataset_splitBy:
    det_feats_input_dir = 'data/rsvg/hbb_obb_features_det'
  else:
    print('No detection bbox feature directory prepared for ', args.dataset)
    sys.exit(0)

  # load dataset
  data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
  data_h5 = osp.join('cache/prepro', dataset_splitBy, 'data.h5')
  dets_json = osp.join('cache/detections', dataset_splitBy, '%s_%s_%s_dets.json' % (args.net_name, args.imdb_name, args.tag))
  loader = DetsLoader(data_json, data_h5, dets_json)
  images = loader.images
  dets = loader.dets
  num_dets = len(dets)
  assert sum([len(image['det_ids']) for image in images]) == num_dets

  # load mrcn model
  #mrcn = inference_no_imdb.Inference(args)

  # feats_h5
  file_name = '%s_%s_%s_det_feats.h5' % (args.net_name, args.imdb_name, args.tag)
  feats_h5 = osp.join('cache/feats', dataset_splitBy, file_name)
  if os.path.exists(feats_h5):
    print('Error: file exists! ' + feats_h5)
    assert False

  f = h5py.File(feats_h5, 'w')
  fc7_set   = f.create_dataset('fc7',   (num_dets, 256), dtype=np.float32)
  pool5_set = f.create_dataset('pool5', (num_dets, 256), dtype=np.float32)

  # extract
  for i, image in enumerate(images):
    image_id = image['image_id']
    det_ids = image['det_ids']
    for det_id in det_ids:
      one_det_filepath = osp.join( det_feats_input_dir, \
          str(det_id)+"_hbb_det_" + ('%s_%s_%s.hdf5' % (args.net_name, args.imdb_name, args.tag)) )
      det_roi_feats = h5py.File(one_det_filepath, 'r')['roi_feats'][0]
      det = loader.Dets[det_id]
      det_pool5, det_fc7 = det_roi_feats.mean(2).mean(1), det_roi_feats.mean(2).mean(1)
      assert det_pool5.shape[0] == 256
      #print(type(det_pool5))
      det_h5_id = det['h5_id']
      fc7_set[det_h5_id] = det_fc7
      pool5_set[det_h5_id] = det_pool5
    if i % 20 == 0:
      print('%s/%s done.' % (i+1, len(images)))

  f.close()
  print('%s written.' % feats_h5)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--imdb_name', default='dota_v1_0', help='image databased trained on.')
  parser.add_argument('--net_name', default='res50')
  parser.add_argument('--tag', default='RoITransformer')

  parser.add_argument('--dataset', type=str, default='rsvg', help='dataset name: rsvg')
  args = parser.parse_args()
  main(args)


