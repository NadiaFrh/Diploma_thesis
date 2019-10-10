#%%
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw


ROOT_DIR = 'Mask_RCNN-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


training_samples = 30000
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
    
    
class CaptConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "captioning"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 90  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    
    BACKBONE = 'resnet101'
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = CaptConfig()
config.display()





class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        index =1
        i=1
        # Add the class names using the base method from utils.Dataset
        source_name = "coco"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return

            while index != class_id :
                self.add_class('removed_class', -i, '-')
                i+=1
                index +=1 

            self.add_class(source_name, class_id, class_name)
            index +=1

            
            
            
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            is_counts = False
            seg = annotation['segmentation']
            image_id = annotation['image_id']          
            if image_id not in annotations:
                annotations[image_id] = []

            if  'counts' in seg :
                is_counts = True
            else:  
                annotations[image_id].append(annotation)      



        # Get all images and add them to the dataset
        seen_images = {}
        count = 0
        for image in coco_json['images']:
                '''
                if count == training_samples:
                    break
                count+=1
                '''
            #if count > 16 :
                image_id = image['id']
                if image_id in seen_images:
                    print("Warning: Skipping duplicate image id: {}".format(image))
                else:
                    seen_images[image_id] = image
                    try:
                        image_file_name = image['file_name']
                        image_width = image['width']
                        image_height = image['height']
                        image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                        image_annotations = annotations[image_id]
                        # Add the image using the base method from utils.Dataset
                        self.add_image(
                            source=source_name,
                            image_id=image_id,
                            path=image_path,
                            width=image_width,
                            height=image_height,
                            annotations=image_annotations
                        )
                    except KeyError as key:
                        print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                    #if count%16 == 0:
                    #    break
                

                

                

    def load_mask(self, image_id):
            """ Load instance masks for the given image.
            MaskRCNN expects masks in the form of a bitmap [height, width, instances].
            Args:
                image_id: The id of the image to load masks for
            Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
            """
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']
            instance_masks = []
            class_ids = []
            
            for annotation in annotations:
                class_id = annotation['category_id']
                mask = Image.new('1', (image_info['width'], image_info['height']))
                mask_draw = ImageDraw.ImageDraw(mask, '1')
                for segmentation in annotation['segmentation']:
                    mask_draw.polygon(segmentation, fill=1)
                    bool_array = np.array(mask) > 0
                    instance_masks.append(bool_array)
                    class_ids.append(class_id)
    
            mask = np.dstack(instance_masks)
            class_ids = np.array(class_ids, dtype=np.int32)
            
            return mask, class_ids


    
 
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_data('annotations/instances_train2014.json','train2014/')
dataset_train.prepare()
image_info_train = dataset_train.image_info
class_info_train = dataset_train.class_info

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_data('annotations/instances_val2014.json','val2014/')
dataset_val.prepare()
image_info_val = dataset_val.image_info



# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
    
#%%
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
'''
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads')

end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')  
'''
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to

# train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=17, 
            layers="all")
model.save_weights('rcnn_weigths')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)


#%%
class InferenceConfig(CaptConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    DETECTION_MIN_CONFIDENCE = 0.0
    

inference_config = InferenceConfig()



# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

#model_path = model.find_last()
model_path ='Mask_RCNN-master/logs/captioning20190803T2305/mask_rcnn_captioning_0017.h5'
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#%%

import skimage
real_test_dir = 'val2014/'
image_paths = []
features = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths[:20]:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    features.append(r['box_features'])
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(8,8))
    
    
    
#%%
import pickle
from PIL import Image
import numpy as np
from collections import defaultdict
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import AveragePooling2D

def inception_model():
    model_img = InceptionV3(include_top=False,
                            input_shape=(299, 299, 3),
                            weights='imagenet',
                            pooling= 'avg')
    #avg_out = AveragePooling2D(pool_size=(8, 8), strides=(8,8))(model_img.output)
    model_new = Model(model_img.input, model_img.output)

    return model_new



#dict_total_images = defaultdict(list)
#dict_rcnn_train = pickle.load( open( "rcnn_val.pickle", "rb" ) )   
incept_model = inception_model()
#%%    
def preprocess_image(image):
    image = image.resize((299,299))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    
    return img_array
i = 0
regions = []

for image_path in total_val_image_name:

    if i == 1:
        break
    img = load_img(image_path)
    regions.append(preprocess_image(img))
    inception_out = incept_model.predict(np.array(regions))
    i+=1
#%%
import pickle
print("Loading data")
total_train_image_name = pickle.load( open( 'train_images.pickle', "rb" ) )
print("Loading data Completed")
#%%
print("Loading data")
total_val_image_name = pickle.load( open( 'val_images.pickle', "rb" ) )
print("Loading data Completed")
#%%
for image_path in image_paths:
    
#%%
import skimage
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
i=0
dict_total_images = []
image_paths = []
path = 'eikones/eikones/'
        
def preprocess_image(image):
    image = image.resize((299,299))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    
    return img_array



for filename in os.listdir(path):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(path, filename))



for image_path in image_paths:
    '''
    if i == 11000:
        break
    '''
    print(i)
    
    i+=1
    #print('{}/{}'.format(j,i))
    regions = []
    img = skimage.io.imread(image_path,plugin='matplotlib')
    img_arr = np.array(img)
    try:
        results = model.detect([img_arr], verbose=1)
    except Exception as e: 
        print(e)
        continue
    r = results[0]
    '''
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(8,8))
    '''
    rois = r['rois']
    box_features = r['box_features']
    img = load_img(image_path)
    width, height = img.size

    regions.append(preprocess_image(img))

    plt.figure()
    plt.imshow(img)
    j=0

    for r in rois[:4]:   
        img_black = np.zeros([height, width, 3], dtype=np.uint8)
        img_black.fill(0)
        img_black = Image.fromarray(img_black)
        cropped = img.crop((r[1],r[0],r[3],r[2]))
        img_black.paste(cropped, (r[1],r[0],r[3],r[2]))
        #print(r)
        '''
        plt.figure()
        plt.imshow(img_black) 
        plt.show()
        '''
        regions.append(preprocess_image(img_black))

    inception_out = incept_model.predict(np.array(regions))
    roi_num = len(inception_out)
       #img_incept = incept_rois_train[image][0]


    if roi_num<5:
        z = np.zeros((5-roi_num, 2048), dtype=int)
        tmp = np.concatenate((inception_out,z),axis=0)

    else:
        tmp = inception_out[:5]
        #bf, p = zip(tmp, image_path)
    try: 
        np.save(image_path, tmp)
        dict_total_images.append(image_path)
    except Exception as e: print(e)
    #dict_total_images[image_path].append(inception_out)

pickle_out = open("my_images.pickle","wb")
pickle.dump(dict_total_images, pickle_out)
pickle_out.close()

    
#%%
import skimage
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import pickle

i=0
dict_total_images = []
image_paths = []

        
def preprocess_image(image):
    image = image.resize((299,299))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    
    return img_array



for filename in os.listdir('train2014/'):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join('train2014/', filename))



for image_path in total_val_image_name:
    '''
    if i == 5:
        break
    '''
    print(i)
    
    i+=1
    regions = []
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    try:
        results = model.detect([img_arr], verbose=1)
    except Exception as e: 
        print(e)
        continue
    r = results[0]
    '''
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(8,8))
    '''
    rois = r['rois']
    box_features = r['box_features']
    img = load_img(image_path)
    width, height = img.size

    regions.append(preprocess_image(img))
    roi_num = len(box_features)

    if roi_num<4:
        z = np.zeros((4-roi_num, 1024), dtype=int)
        tmp = np.concatenate((box_features,z),axis=0)

    else:
        tmp = box_features[:4]
        #bf, p = zip(tmp, image_path)
    try: 
        np.save('box_features/'+image_path, tmp)
        dict_total_images.append(image_path)
    except Exception as e: print(e)
    #dict_total_images[image_path].append(inception_out)
'''
pickle_out = open("train_images.pickle","wb")
pickle.dump(dict_total_images, pickle_out)
pickle_out.close()
'''   
    
    
    
    