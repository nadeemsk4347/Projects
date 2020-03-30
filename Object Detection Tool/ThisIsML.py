#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
 
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
 
sys.path.append("..")
from utils import ops as utils_ops
 
from utils import label_map_util
 
from utils import visualization_utils as vis_util


# In[12]:

def GiveMePathsReturnsListOfImg(image_path,toDetectOnlyThese=[] , index=0,detection_score=0.5):
    if detection_score==0:
        detection_score=.5
    
    models = ['ssdlite_mobilenet_v2_coco_2018_05_09', 'ssd_mobilenet_v1_coco_2018_01_28', 'faster_rcnn_inception_v2_coco_2018_01_28']
    MODEL_NAME = models[index]
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    #DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
     
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
     
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
     
    NUM_CLASSES = 90
   
    
    # opener = urllib.request.URLopener()
    # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    

    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

#    images = Run_Object_Detectio_On_Images(TEST_IMAGE_PATHS,category_index,detection_graph)
    
#def Run_Object_Detectio_On_Images(images_path,category_index,detection_graph):
    #IMAGE_SIZE = (12, 8)
    #for image_path in TEST_IMAGE_PATH:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    
    
    print(output_dict['detection_classes'])
    print('detection_classes')
    print(category_index)
    print('category_index')
    
    
    
    vis_util.visualize_boxes_and_labels_on_image_array_custom(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index, toDetectOnlyThese,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      min_score_thresh=detection_score,
      line_thickness=5,
      
      )
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    print(type(image_np))
    print(MODEL_NAME)
    return image_np
    
# In[15]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                #END if tensor_name in
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                #END IF DETECTION MASKS
            
            #END FOR KEY LOOP
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# In[18]:


image = GiveMePathsReturnsListOfImg('/home/nadeem/Desktop/Python/Tensorflow/models/research/object_detection/test_images/image/image1.jpg')
IMAGE_SIZE = (12, 8)
    #plt.figure(figsize=IMAGE_SIZE)
plt.imshow(image)
height, width, channel = image.shape
bytesPerLine = 3 * width
    #qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)


