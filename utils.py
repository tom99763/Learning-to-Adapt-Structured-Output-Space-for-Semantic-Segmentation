import tensorflow as tf
import os 
from sklearn.model_selection import train_test_split as ttp
AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_pair(img_pth, mask_pth):
  img = tf.image.decode_jpeg(tf.io.read_file(img_pth), channels=3)
  img /= 255.
  mask = tf.image.decode_jpeg(tf.io.read_file(mask_pth), channels=1)
  mask = tf.where(mask==0., 0., 1.)
  return img, mask
  

def create_tf_dataset(source_imgs_pth, source_masks_pth, target_imgs_pth, target_masks_pth):
  ds_soucre = tf.data.Dataset.from_tensor_slices((source_imgs_pth, source_masks_pth)).shuffle(256).\
                map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth))
  
  ds_target = tf.data.Dataset.from_tensor_slices((target_imgs_pth, target_masks_pth)).repeat().\
               map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth))
  
  ds = tf.data.Dataset.zip((ds_source, ds_target)).batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
  return ds


  
