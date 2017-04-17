"""
preprocess txt files into svm-format, save tags/labels into different file for decompose
"""
import sys

if len(sys.argv) != 2:
    sys.exit("Usage: python {} filename.".format(sys.argv[0]))

fn = sys.argv[1]
svmfile = fn + ".svm"
tagfile = fn + ".tag"
firstline = True
with open(fn, 'r') as f, open(svmfile, 'w') as sf, open(tagfile, 'w') as tf:
    for line in f.readlines():
        splitted = line.split()
        tag = splitted[0]
        label = 1 if tag[0] == 'A' else -1
        splitted[0] = str(label)
        if not firstline:
            tf.write('\n')
            sf.write('\n')
        tf.write(tag)
        sf.write(' '.join(splitted))
        firstline = False


print('Write svm-format data to: ', svmfile)
print('Write tag to: ', tagfile)