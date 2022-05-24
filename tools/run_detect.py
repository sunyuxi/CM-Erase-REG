"""
Run detection on all images and save detected bounding box (with category_name and score).

cache/detections/refcoco_unc/{net}_{imdb}_{tag}_dets.json has
0. dets: list of {det_id, box, image_id, category_id, category_name, score}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import argparse
import json
import torch

def main(args):

  # Image Directory
  params = vars(args)
  dataset_splitBy = params['dataset']

  # make save dir
  save_dir = osp.join('cache/detections', dataset_splitBy)
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)
  print(save_dir)

  # import refer
  from refer import REFER
  data_root, dataset = params['data_root'], params['dataset']
  refer = REFER(data_root, dataset)
  cat_name_to_cat_ix = {category_name: category_id for category_id, category_name in refer.Cats.items()}

  det_json_path = params['det_json_path']
  with open(det_json_path, 'r') as f:
    det_json_data = json.load(f)

  # detect and prepare dets.json
  dets = []
  cnt = 0
  for image_id_ori, image in refer.Imgs.items():
    file_name = image['file_name']
    print(file_name)
    assert file_name in det_json_data
    one_detections = det_json_data[file_name]
    det_rbbox_ids, det_bboxes, det_scores = one_detections['det_rbbox_ids'], one_detections['det_bboxes'], one_detections['det_scores']
    det_categories, image_id = one_detections['det_categories'], one_detections['image_id']
    assert image_id_ori == image_id
    assert len(det_rbbox_ids) == len(det_bboxes)
    assert len(det_bboxes) == len(det_scores)
    assert len(det_scores) == len(det_categories)
    for idx, det_id in enumerate(det_rbbox_ids):
      category_name=det_categories[idx]
      sc = det_scores[idx]
      x, y, w, h = det_bboxes[idx]
      assert category_name in cat_name_to_cat_ix
      det = {'det_id': det_id,
              'h5_id' : det_id,  # we make h5_id == det_id
              'box': [x, y, w+1, h+1],
              'image_id': image_id,
              'category_id': cat_name_to_cat_ix[category_name],
              'category_name': category_name,
              'score': sc} 
      dets += [det]
    cnt += 1

    print('%s/%s done.' % (cnt, len(refer.Imgs)))
  assert cnt == len(refer.Imgs)

  # save dets.json = [{det_id, box, image_id, score}]
  # to cache/detections/
  save_path = osp.join(save_dir, '%s_%s_%s_dets.json' % (args.net_name, args.imdb_name, args.tag))
  with open(save_path, 'w') as f:
    json.dump(dets, f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--det_json_path', default='data/rsvg/det_instances_rsvg.json', type=str, help='detected bboxes per image')
  parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
  parser.add_argument('--dataset', default='rsvg', type=str, help='rsvg')
  parser.add_argument('--imdb_name', default='dota_v1_0', help='image databased trained on.')
  parser.add_argument('--net_name', default='res50')
  parser.add_argument('--tag', default='RoITransformer')

  args = parser.parse_args()

  main(args)


