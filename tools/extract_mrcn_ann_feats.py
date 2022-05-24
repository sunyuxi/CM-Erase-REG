from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import numpy as np
import h5py

import _init_paths
# dataloader
from loaders.loader import Loader

def main(args):
  dataset_splitBy = args.dataset
  if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
    os.makedirs(osp.join('cache/feats/', dataset_splitBy))

  # detection feature Directory
  if 'rsvg' in dataset_splitBy:
    gt_feats_input_dir = 'data/rsvg/hbb_obb_features_gt'
  else:
    print('No ground truth bbox feature directory prepared for ', args.dataset)
    sys.exit(0)

  # load dataset
  data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
  data_h5 = osp.join('cache/prepro', dataset_splitBy, 'data.h5')
  loader = Loader(data_json, data_h5)
  images = loader.images
  anns = loader.anns
  num_anns = len(anns)
  assert sum([len(image['ann_ids']) for image in images]) == num_anns

  # feats_h5
  # feats_h5 = osp.join('cache/feats', dataset_splitBy, args.file_name)
  file_name = '%s_%s_%s_ann_feats.h5' % (args.net_name, args.imdb_name, args.tag)
  feats_h5 = osp.join('cache/feats', dataset_splitBy, file_name)

  f = h5py.File(feats_h5, 'w')
  pool5_set = f.create_dataset('pool5', (num_anns, 256), dtype=np.float32)
  fc7_set = f.create_dataset('fc7', (num_anns, 256), dtype=np.float32)

  # extract
  for i, image in enumerate(images):
    image_id = image['image_id']
    ann_ids = image['ann_ids']
    for ann_id in ann_ids: # ann_id is ground truth bbox id
      one_gt_filepath = osp.join( gt_feats_input_dir, \
          str(ann_id)+"_hbb_gt_" + ('%s_%s_%s.hdf5' % (args.net_name, args.imdb_name, args.tag)) )
      gt_roi_feats = h5py.File(one_gt_filepath, 'r')['roi_feats'][0]
      
      ann_pool5, ann_fc7 = gt_roi_feats.mean(2).mean(1), gt_roi_feats.mean(2).mean(1)

      ann = loader.Anns[ann_id]
      ann_h5_id = ann['h5_id']
      pool5_set[ann_h5_id] = ann_pool5
      fc7_set[ann_h5_id] = ann_fc7
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


