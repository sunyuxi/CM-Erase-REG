import os.path as osp
import sys 

# mrcn path
this_dir = osp.dirname(__file__)

# model path
sys.path.insert(0, osp.join(this_dir, '..', 'lib'))