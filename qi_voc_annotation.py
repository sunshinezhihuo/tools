import xml.etree.ElementTree as ET
from os import getcwd

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(fo, image_id, list_file):
    xmlpath = rootpath + fo +'/Annotations/'
    in_file = open(xmlpath+'%s.xml'% image_id)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))

        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


'''
foldername = ('MOT17-02', 'MOT17-04', 'MOT17-05',
              'MOT17-09', 'MOT17-10', 'MOT17-11',
              'MOT17-13')

length = (600, 1050, 837, 525, 654, 900, 750)

sunny(5,13)
night(4,10)
indoor(9,11)
cloudy(2)
'''
#%%
# 5 train 2 val
#%%


rootpath = '/home/qi/benchmark/QiMOT17det/'
trainvaltype = 'train5val2/'
# Here we use 5 train sequences and 2 val sequences

train = True
if train:
    trainval = 'train'

    foldername = ('MOT17-02', 'MOT17-04',
                   'MOT17-10', 'MOT17-11',
                  'MOT17-13')

    length = (600, 1050, 837, 525, 654, 900, 750)
else:
    trainval = 'val'

    foldername = ('MOT17-05',
                  'MOT17-09')

    length = (600, 1050, 837, 525, 654, 900, 750)

traintxt = rootpath + trainvaltype + '%s.txt'%(trainval)
list_file = open(traintxt, 'w')

for folder, l in zip(foldername, length):

    image_ids = open(rootpath +'%s/ImageSets/Main/%s.txt' % (folder, trainval)).read().strip().split()

    # traintxt = rootpath + trainvaltype + '%s.txt'%(trainval)
    # list_file = open(traintxt, 'w')
    for image_id in image_ids:
        outputpath = rootpath + folder + '/JPEGImages/%s.jpg'
        list_file.write( outputpath % image_id)

        convert_annotation(folder, image_id, list_file)
        list_file.write('\n')

list_file.close()

