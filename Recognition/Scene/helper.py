#!/usr/bin/env python

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

# colour map for ai edge
aiedge_label_color = [(0,     0, 255),  # 1.  Car
                      (193, 214,   0),  # 2.  Bus
                      (180,   0, 129),  # 3.  Truck
                      (255, 121, 166),  # 4.  SVehicle
                      (255,   0,   0),  # 5.  Pedestrian
                      ( 65, 166,   1),  # 6.  Motobike
                      (208, 149,   1),  # 7.  Bicycle
                      (255, 255,   0),  # 8.  Signal
                      (255, 134,   0),  # 9.  Signs
                      (  0, 152, 225),  # 10. Sky
                      (  0, 203, 151),  # 11. Building
                      ( 85, 255,  50),  # 12. Natural
                      ( 92, 136, 125),  # 13. Wall
                      ( 69,  47, 142),  # 14. Lane
                      (136,  45,  66),  # 15. Ground
                      (  0, 255, 255),  # 16. Sidewalk
                      (215,   0, 255),  # 17. RoadShoulder
                      (180, 131, 135),  # 18. Obstacle
                      ( 81,  99,   0),  # 19. others
                      ( 86,  62,  67)]  # 20. own

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

        
def gen_batch_function(dataset, data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_kitti_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)):
            path for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    def get_aiedge_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'images', '*.jpg'))
        label_paths = {
            re.sub(r'.png', '.jpg', os.path.basename(path)):
            path for path in glob(os.path.join(data_folder, 'annotations', '*.png'))}
        
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                images.append(image)
                
                gt_image_file = label_paths[os.path.basename(image_file)]
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt = []
                for i in range(len(aiedge_label_color)):
                    gt.append(np.all(gt_image==aiedge_label_color[i], axis=2))

                gt_image = np.dstack(gt)
                #gt_image = np.concatenate(gt, axis=2)
                #gt_image = gt
                gt_images.append(gt_image)
                
            yield np.array(images), np.array(gt_images)
            
    if dataset == 'Kitti':
        return get_kitti_batches_fn
    else:
        return get_aiedge_batches_fn
    

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, test_image_dir, file_type, image_shape):
    """
    Generate test output using the test images
    :param sess: IF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, test_image_dir, file_type)):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
             [tf.nn.softmax(logits)],
             {keep_prob: 1.0, image_pl: [image]})

        _, classes = im_softmax[0].shape

        # for dataset of Kitti
        if classes < 3:
        
            im_softmax = im_softmax[0][:,1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
        
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

        # for dataset of AI edge
        else:
            for i in range(classes):
                im = im_softmax[0][:,i].reshape(image_shape[0], image_shape[1])
                segmentation = (im > 0.5).reshape(image_shape[0], image_shape[1], 1)
                r, g, b = aiedge_label_color[i]
                mask = np.dot(segmentation, np.array([[r, g, b, 127]]))
                mask = scipy.misc.toimage(mask, mode="RGBA")

                street_im = scipy.misc.toimage(image)
                street_im.paste(mask, box=None, mask=mask)
                
        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(model_dir, runs_dir, data_dir, test_dir, test_image_dir, file_type,
                           sess, image_shape, logits, keep_prob, input_image, saver):
    # Make folder for current run
    run_time   = str(time.time())
    output_dir = os.path.join(runs_dir, run_time)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Make folder for the model
    model_out_dir = os.path.join(model_dir, run_time)
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)
    os.makedirs(model_out_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, os.path.join(data_dir, test_dir), test_image_dir, file_type, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    saver.save(sess, os.path.join(model_out_dir, 'model'))
