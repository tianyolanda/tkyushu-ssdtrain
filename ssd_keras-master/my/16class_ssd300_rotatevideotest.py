import cv2
import time
import random
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Set the image size.
img_height = 300
img_width = 300
key = ''
# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=16,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = '/home/ogai1234/lala/ssd_keras-master/tia_trained_weight/16class_ssd300_pascal_07+12_epoch-118_loss-2.9086_val_loss-2.2870.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


cap = cv2.VideoCapture('../video/t4.mp4')
success, image = cap.read()
framenum = 0
while key != 113:
    t1 = time.time()
    input_images = []
    orig_images = []
    #.read() have two parameters!!
    success, image = cap.read()

    # imgpil = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    center = (w/2, h/2)
    # 将图像旋转180度
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image_rotate = cv2.warpAffine(image, M, (w, h))

    orig_images.append(image_rotate)
    # cv2.imshow('video',image)

    resize = cv2.resize(image_rotate, (300, 300))

    # resize = np.rot90(resize)

    input_images.append(resize)


    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    np.set_printoptions(precision=2, suppress=True, linewidth=180)
    # print(y_pred[0][0], "++++++++++++++++++++++++++++")
    confidence_threshold = 0.5


    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    print('obj num: ',len(y_pred_thresh[0]))

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes

    # classes = ["background", "dog", "umbrellaman", "cone", "car", "bicycle", "person"]
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bus', 'car', 'cat', 'cow', 'dog', 'horse',
               'motorbike', 'person', 'sheep', 'train',
               'umbrellaman', 'cone']

    colors = dict()
    for box in y_pred_thresh[0]:
        # color = colors[int(box[0])]
        cls_id = int(box[0])
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(box[2] * orig_images[0].shape[1] / img_width)
        ymin = int(box[3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[4] * orig_images[0].shape[1] / img_width)
        ymax = int(box[5] * orig_images[0].shape[0] / img_height)
        # print(box[2])
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        print(orig_images[0].shape[1])
        print(img_width)

        print(xmin, ymin)
        print(xmax, ymax)
        text_top = (xmin, ymin - 20)
        text_bot = (xmin + 280, ymin + 5)
        text_pos = (xmin + 10, ymin)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), color=colors[cls_id], thickness=4)
        # cv2.rectangle(orig_images[0], text_top, text_bot, color=colors[cls_id], -1)
        cv2.putText(orig_images[0], label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("success!", orig_images[0])
    # print("OK-----------------------")
    key = cv2.waitKey(1)
    # count += 1
    t2 = time.time()
    print('fps:', 1 / (t2 - t1))

cap.release()
cv2.destroyAllWindows()




