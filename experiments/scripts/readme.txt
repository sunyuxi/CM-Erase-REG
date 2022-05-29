#具体流程记不太清楚
#CM-Erase-Reg比MAttNet数据集，多了一个glove向量，即
data/rsvg/glove/glove.840B.300d.txt
#通过./utils/extract_glove.py文件，生成data/rsvg/glove/rsvg_glove.840B.300d.npy文件，手动复制到对应文件夹

#首先训练MattNet模型
./experiments/scripts/train_mattnet.sh

训练CM-Att-Erase模型
./experiments/scripts/train_erase.sh

#评估模型
./experiments/scripts/eval_dets.sh