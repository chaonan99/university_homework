import sys
import glob
import re

labels_file = "/home/chenhaonan/data/Image/cup_dataset/train_img_forstu/label.txt"
with tf.gfile.FastGFile(labels_file, 'r') as f:
    f.readline()
    texts = [l.strip().split()[1].encode('utf-8') for l in f.readlines()]
for l in tf.gfile.FastGFile(labels_file, 'r').readlines():
    sys.stdout.write(l.strip().split()[1])
[l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

data_dir = '/home/chenhaonan/data/Image/cup_dataset/train_img_forstu'
filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, "*.jpg")), key=numericalSort)


