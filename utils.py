import tensorflow as tf
import os 
from sklearn.model_selection import train_test_split as ttp
AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_pair(img_pth, mask_pth, img_size):
    img = tf.image.decode_jpeg(tf.io.read_file(img_pth), channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, 'float32')
    img /= 255.
    mask = tf.image.decode_jpeg(tf.io.read_file(mask_pth), channels=1)
    mask = tf.image.resize(mask, [img_size, img_size])
    mask = tf.cast(mask, 'float32')
    mask = tf.where(mask==0., 0., 1.)
    return img, mask
  

def create_tf_dataset(source_imgs_pth, source_masks_pth, target_imgs_pth, target_masks_pth, img_size):

    ds_source = tf.data.Dataset.from_tensor_slices((source_imgs_pth, source_masks_pth)).shuffle(256).\
                map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth, img_size))
    
    ds_target = tf.data.Dataset.from_tensor_slices((target_imgs_pth, target_masks_pth)).repeat().\
               map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth, img_size))

    ds = tf.data.Dataset.zip((ds_source, ds_target)).shuffle(256).\
                 batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def create_dataset(opt):
    source_imgs_pth = list(map(lambda name: opt.source_pth + '/images/' + name, os.listdir(opt.source_pth + '/images')))
    source_masks_pth = list(map(lambda name: opt.source_pth + '/masks/' + name, os.listdir(opt.source_pth + '/masks')))
    target_imgs_pth = list(map(lambda name: opt.target_pth + '/images/' + name, os.listdir(opt.target_pth + '/images')))
    target_masks_pth = list(map(lambda name: opt.target_pth + '/masks/' + name, os.listdir(opt.target_pth + '/masks')))
    
    source_imgs_pth_train, source_imgs_pth_val, source_masks_pth_train, source_masks_pth_val = ttp(
        source_imgs_pth,
        source_masks_pth,
        test_size = opt.val_size,
        random_state = opt.seed
    )
    
    target_imgs_pth_train, target_imgs_pth_val, target_masks_pth_train, target_masks_pth_val = ttp(
        target_imgs_pth,
        target_masks_pth,
        test_size = opt.val_size,
        random_state = opt.seed
    )
    
    ds_train = create_tf_dataset(
        source_imgs_pth_train,
        source_masks_pth_train,
        target_imgs_pth_train,
        target_masks_pth_train,
        opt.img_size
    )
    
    ds_val = create_tf_dataset(
        source_imgs_pth_val,
        source_masks_pth_val,
        target_imgs_pth_val,
        target_masks_pth_val,
        opt.img_size
    )
    return ds_train, ds_val

  
