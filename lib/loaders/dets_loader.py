"""
data_json has 
0. refs        : list of {ref_id, ann_id, box, image_id, split, category_id, sent_ids}
1. images      : list of {image_id, ref_ids, ann_ids, file_name, width, height, h5_id}
2. anns        : list of {ann_id, category_id, image_id, box, h5_id}
3. sentences   : list of {sent_id, tokens, h5_id}
4: word_to_ix  : word->ix
5: cat_to_ix   : cat->ix
6: label_length: L
Note, box in [xywh] format

dets_json has
0. dets        : list of {det_id, box, image_id, category_id, category_name, score}

label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py
import json
import random
import functools
from loaders.loader import Loader

import torch
from torch.autograd import Variable


# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


# DetsLoader instance
class DetsLoader(Loader):

  def __init__(self, data_json, data_h5, dets_json):
    # parent loader instance
    Loader.__init__(self, data_json, data_h5)

    # prepare attributes
    self.att_to_ix = self.info['att_to_ix']
    self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
    self.num_atts = len(self.att_to_ix)
    self.att_to_cnt = self.info['att_to_cnt']

    # prepare dets
    self.dets = json.load(open(dets_json))
    self.Dets = {det['det_id']: det for det in self.dets}

    # add dets to image
    for image in self.images:
      image['det_ids'] = []
    for det in self.dets:
      image = self.Images[det['image_id']]
      image['det_ids'] += [det['det_id']]

    # img_iterators for each split
    self.split_ix = {}
    self.iterators = {}
    for image_id, image in self.Images.items():
      split = self.Refs[image['ref_ids'][0]]['split']
      if split not in self.split_ix:
        self.split_ix[split] = []
        self.iterators[split] = 0
      self.split_ix[split] += [image_id]
    for k, v in self.split_ix.items():
      print('assigned %d images to split %s' % (len(v), k))

  def prepare_mrcn(self, head_feats_dir, suffix, args, img_feats_dir=None, imgfeats_suffix=None):
    """
    Arguments:
      head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
      args: imdb_name, net_name, iters, tag
    """
    self.head_feats_dir = head_feats_dir
    self.dict_detid2feats = {}
    dets_key_list = self.Dets.keys()
    for det_id in dets_key_list:
      filepath=osp.join(head_feats_dir, str(det_id)+"_"+suffix)
      with h5py.File(filepath, 'r') as f:
        self.dict_detid2feats[det_id] = f['roi_feats'][0]
    assert args.net_name == 'res50'
    self.pool5_dim = 256
    self.fc7_dim = 256

    self.dict_imgid2feats = {}
    if img_feats_dir != None:
      #load img features
      for imgid in self.Images:
        filepath=osp.join(img_feats_dir, str(imgid)+"_"+imgfeats_suffix)
        with h5py.File(filepath, 'r') as f:
          self.dict_imgid2feats[imgid] = f['roi_feats'][0]

  def load_img_to_spatial_fc7(self, input_imgid):
    imgid_list = [input_imgid]
    assert len(self.dict_imgid2feats)>0
    data_list=list()
    for imgid in imgid_list:
      data_list.append(self.dict_imgid2feats[imgid])
    pool5_fc7 = np.stack(data_list, axis=0)

    return torch.from_numpy(pool5_fc7).float().cuda(), torch.from_numpy(pool5_fc7).float().cuda()

  # load different kinds of feats
  def loadFeats(self, Feats):
    # Feats = {feats_name: feats_path}
    self.feats = {}
    self.feat_dim = None
    for feats_name, feats_path in Feats.items():
      if osp.isfile(feats_path):
        self.feats[feats_name] = h5py.File(feats_path, 'r')
        self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
        assert self.feat_dim == self.fc7_dim
        print('FeatLoader loading [%s] from %s [feat_dim %s]' % \
              (feats_name, feats_path, self.feat_dim))

  # reset iterator
  def resetIterator(self, split):
    self.iterators[split] = 0

  # expand list by seq_per_ref, i.e., [a,b], 3 -> [aaabbb]
  def expand_list(self, L, n):
    out = []
    for l in L:
      out += [l] * n
    return out

  def fetch_neighbour_ids(self, ref_det_id):
    """
    For a given ref_det_id, we return
    - st_det_ids: same-type neighbouring det_ids (not including itself)
    - dt_det_ids: different-type neighbouring det_ids
    Ordered by distance to the input det_id
    """
    ref_det = self.Dets[ref_det_id]
    x, y, w, h = ref_det['box']
    rx, ry = x+w/2, y+h/2

    def compare(det_id0, det_id1):
      x, y, w, h = self.Dets[det_id0]['box']
      ax0, ay0 = x+w/2, y+h/2
      x, y, w, h = self.Dets[det_id1]['box']
      ax1, ay1 = x+w/2, y+h/2
      # closer --> former
      if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
        return -1
      else:
        return 1
    image = self.Images[ref_det['image_id']]

    det_ids = list(image['det_ids'])  # copy in case the raw list is changed
    det_ids = sorted(det_ids, key=functools.cmp_to_key(compare))

    st_det_ids, dt_det_ids = [], []
    for det_id in det_ids:
      if det_id != ref_det_id:
        if self.Dets[det_id]['category_id'] == ref_det['category_id']:
          st_det_ids += [det_id]
        else:
          dt_det_ids += [det_id]

    return st_det_ids, dt_det_ids    

  def fetch_grid_feats(self, bbox_ids):
    """returns 
    - pool5 (n, 256, 7, 7)
    - fc7   (n, 256, 7, 7) 
    """
    data_list=list()
    for bbox_id in bbox_ids:
      data_list.append(self.dict_detid2feats[bbox_id])
    pool5_fc7 = np.stack(data_list, axis=0)

    return torch.from_numpy(pool5_fc7).float().cuda(), torch.from_numpy(pool5_fc7).float().cuda()

  def compute_lfeats(self, det_ids):
    # return ndarray float32 (#det_ids, 5)
    lfeats = np.empty((len(det_ids), 5), dtype=np.float32)
    for ix, det_id in enumerate(det_ids):
      det = self.Dets[det_id]
      image = self.Images[det['image_id']]
      x, y, w, h = det['box']
      ih, iw = image['height'], image['width']
      lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
    return lfeats

  def compute_dif_lfeats(self, det_ids, topK=5):
    # return ndarray float32 (#det_ids, 5*topK)
    dif_lfeats = np.zeros((len(det_ids), 5*topK), dtype=np.float32)
    for i, ref_det_id in enumerate(det_ids):
      # reference box 
      rbox = self.Dets[ref_det_id]['box']
      rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
      # candidate boxes
      st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id)
      for j, cand_det_id in enumerate(st_det_ids[:topK]):
        cbox = self.Dets[cand_det_id]['box']
        cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
        dif_lfeats[i, j*5:(j+1)*5] = \
          np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])

    return dif_lfeats

  def fetch_cxt_feats(self, det_ids, opt):
    """
    Return 
    - cxt_feats  : ndarray (#det_ids, topK, feat_dim)
    - cxt_lfeats : ndarray (#det_ids, topK, 5) 
    - cxt_det_ids: [[det_id]] of size (#det_ids, topK), padded with -1
    Note we use neighbouring objects for computing context objects, zeros padded.
    """
    topK = opt['num_cxt']
    cxt_feats = np.zeros((len(det_ids), topK, self.feat_dim), dtype=np.float32)  
    cxt_lfeats = np.zeros((len(det_ids), topK, 5), dtype=np.float32)
    cxt_det_ids = -np.ones((len(det_ids), topK), dtype=np.int32) # (#det_ids, topK)
    for i, ref_det_id in enumerate(det_ids):
      # reference box
      rbox = self.Dets[ref_det_id]['box']
      rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
      # candidate boxes
      st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id)
      if opt['with_st'] > 0:
        cand_det_ids = dt_det_ids + st_det_ids
      else:
        cand_det_ids = dt_det_ids
      cand_det_ids = cand_det_ids[:topK]
      for j, cand_det_id in enumerate(cand_det_ids):
        cand_det = self.Dets[cand_det_id]
        cbox = cand_det['box']
        cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
        cxt_lfeats[i, j, :] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        cxt_feats[i, j, :] = self.feats['det']['fc7'][cand_det['h5_id'], :]
        cxt_det_ids[i, j] = cand_det_id

    cxt_det_ids = cxt_det_ids.tolist()
    return cxt_feats, cxt_lfeats, cxt_det_ids


  def getTestBatch(self, split, opt):
    # Fetch feats according to the image_split_ix 
    # current image
    wrapped = False
    split_ix = self.split_ix[split]
    max_index = len(split_ix) - 1
    ri = self.iterators[split]
    ri_next = ri+1
    if ri_next > max_index:
      ri_next = 0
      wrapped = True
    self.iterators[split] = ri_next
    image_id = split_ix[ri]
    image = self.Images[image_id]

    # fetch feats
    det_ids = image['det_ids']
    pool5, fc7 = self.fetch_grid_feats(det_ids) # (#det_ids, k, 7, 7)
    lfeats     = self.compute_lfeats(det_ids)
    dif_lfeats = self.compute_dif_lfeats(det_ids)
    cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, opt)

    # img_fc7
    head_fc7, _ = self.load_img_to_spatial_fc7(image_id)
    batch_head_fc7 = []
    for cnt in range(len(det_ids)):
      batch_head_fc7 += [head_fc7]
    img_fc7 = torch.cat(batch_head_fc7, 0); img_fc7.detach_()

    # fetch sents, labels and gd_boxes
    sent_ids = []
    gd_boxes = []
    for ref_id in image['ref_ids']:
      ref = self.Refs[ref_id]
      for sent_id in ref['sent_ids']:
        sent_ids += [sent_id]
        gd_boxes += [ref['box']]
    labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])

    # move to Variable
    lfeats = Variable(torch.from_numpy(lfeats).cuda(), volatile=True)
    labels = Variable(torch.from_numpy(labels).long().cuda(), volatile=True)
    dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda(), volatile=True)
    cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda(), volatile=True)
    cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda(), volatile=True)

    # return data
    data = {}
    data['image_id'] = image_id
    data['det_ids'] = det_ids
    data['cxt_det_ids'] = cxt_det_ids
    data['sent_ids'] = sent_ids
    data['gd_boxes'] = gd_boxes
    data['Feats']  = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                      'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'img_fc7': img_fc7}
    data['labels'] = labels
    data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
    return data

  def getImageBatch(self, image_id, sent_ids=None, opt={}):
    # fetch head and im_info
    image = self.Images[image_id]
    
    # fetch feats
    det_ids = image['det_ids']
    pool5, fc7 = self.fetch_grid_feats(det_ids) # (#det_ids, k, 7, 7)
    lfeats     = self.compute_lfeats(det_ids)
    dif_lfeats = self.compute_dif_lfeats(det_ids)
    cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, opt)

    # fetch sents
    gd_boxes = []
    if sent_ids is None:
      sent_ids = []
      for ref_id in image['ref_ids']:
        ref = self.Refs[ref_id]
        for sent_id in ref['sent_ids']:
          sent_ids += [sent_id]
          gd_boxes += [ref['box']]
    else:
      # given sent_id, we find the gd_ix
      for sent_id in sent_ids:
        ref = self.sentToRef[sent_id]
        gd_boxes += [ref['box']]
    labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])

    # move to Variable
    lfeats = Variable(torch.from_numpy(lfeats).cuda())
    labels = Variable(torch.from_numpy(labels).long().cuda())
    dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
    cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
    cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())

    # return data
    data = {}
    data['image_id'] = image_id
    data['det_ids'] = det_ids
    data['cxt_det_ids'] = cxt_det_ids
    data['sent_ids'] = sent_ids
    data['gd_boxes'] = gd_boxes
    data['Feats']  = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                      'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats}
    data['labels'] = labels
    return data


