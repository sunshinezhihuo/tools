#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS
from operator import itemgetter
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3  # 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,frame,det_bb):
        start = time.time()

        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #draw = ImageDraw.Draw(image)
            #label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom)) 
            
            x = left
            y = top            
            w = right - left
            h = bottom - top
            
            if(predicted_class == 'person'):
                newdet = [frame, -1, x, y, w, h, score]
                det_bb.append(newdet)

            #if top - label_size[1] >= 0:
            #    text_origin = np.array([left, top - label_size[1]])
            #else:
            #    text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            #for i in range(thickness):
            #    draw.rectangle(
            #        [left + i, top + i, right - i, bottom - i],
            #        outline=self.colors[c])
            #draw.rectangle(
            #    [tuple(text_origin), tuple(text_origin + label_size)],
            #   fill=self.colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            #del draw

        end = time.time()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

'''
def detect_img(yolo):
    while True:

        # img = input('Input image filename:')
        img = '65.jpg'
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
'''
def detect_img(yolo):   
    # img = input('Input image filename:')
    img = '65.jpg'
    curframe = img.split(".")[0]
    frame = int(curframe)   

    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()


if __name__ == '__main__':    
    yolo = YOLO()
    train = True
    if train:
        foldername = ('MOT17-02', 'MOT17-04', 'MOT17-05',
                     'MOT17-09', 'MOT17-10', 'MOT17-11',
                     'MOT17-13')
        length = (600, 1050, 837, 525, 654, 900, 750)

        # foldername = ( 'MOT17-13', 'MOT17-13')
        # length = (750, 750)

    else:
        foldername = ('MOT17-01', 'MOT17-03', 'MOT17-06',
                      'MOT17-07', 'MOT17-08', 'MOT17-12',
                      'MOT17-14')
        length = (450, 1500, 1194, 500, 625, 900, 750)
        
    for folder, l in zip(foldername, length):
        print('Producing sequence:%s' % folder)

        im_names = []

        for num in range(1,l+1):
            if num < 10 and num > 0:
                pic = '00000' + str(num) + '.jpg'
            elif num < 100:
                pic = '0000' + str(num) + '.jpg'
            elif num < 1000:
                pic = '000' + str(num) + '.jpg'
            else:
                pic = '00' + str(num) + '.jpg'
            im_names.append(pic)

        det_bb = []
        
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for data/demo/{}'.format(im_name))
            curframe = im_name.split(".")[0]
            frame = int(curframe)
            # print(frame)
            # in case of Nonetype
            if train:
                path = '/home/qi/benchmark/MOT17Det/train/%s/img1' % folder
            else:
                path = '/home/qi/benchmark/MOT17Det/test/%s/img1' % folder
           
            img = os.path.join(path, im_name)    
    
            #yolo = YOLO()
            
            # img = '65.jpg'
            # curframe = img.split(".")[0]
            # frame = int(curframe)   
        
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                
            else:
                r_image = yolo.detect_image(image,frame,det_bb)
                # r_image.show()
        
        det_bb.sort(key=itemgetter(1, 0))
    
        with open('/home/qi/keras-yolo3/MOT17Det/yolov3/train/%s/det.txt' % folder, 'w') as rst_f:
        # with open('det.txt', 'w') as rst_f:
            for bb in det_bb:
                rst_f.write(','.join([str(value) for value in bb]) + '\n')
                    
        #yolo.close_session()
    yolo.close_session()
    