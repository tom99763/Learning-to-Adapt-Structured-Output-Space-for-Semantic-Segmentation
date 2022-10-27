import tensorflow as tf
from tensorflow.keras import callbacks
import os
from sklearn.model_selection import train_test_split as ttp
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_pair(img_pth, mask_pth, img_size):
    img = tf.image.decode_jpeg(tf.io.read_file(img_pth), channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, 'float32')
    img /= 255.
    mask = tf.image.decode_jpeg(tf.io.read_file(mask_pth), channels=1)
    mask = tf.image.resize(mask, [img_size, img_size])
    mask = tf.cast(mask, 'float32')
    mask = tf.where(mask == 0., 0., 1.)[..., 0]
    return img, mask


def create_tf_dataset(source_imgs_pth, source_masks_pth, target_imgs_pth, target_masks_pth, opt):
    ds_source = tf.data.Dataset.from_tensor_slices((source_imgs_pth, source_masks_pth)).shuffle(256). \
        map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth, opt.img_size))

    ds_target = tf.data.Dataset.from_tensor_slices((target_imgs_pth, target_masks_pth)).repeat(). \
        map(lambda img_pth, mask_pth: read_pair(img_pth, mask_pth, opt.img_size))

    ds = tf.data.Dataset.zip((ds_source, ds_target)).shuffle(256). \
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
        test_size=opt.val_size,
        random_state=opt.seed
    )

    target_imgs_pth_train, target_imgs_pth_val, target_masks_pth_train, target_masks_pth_val = ttp(
        target_imgs_pth,
        target_masks_pth,
        test_size=opt.val_size,
        random_state=opt.seed
    )

    ds_train = create_tf_dataset(
        source_imgs_pth_train,
        source_masks_pth_train,
        target_imgs_pth_train,
        target_masks_pth_train,
        opt
    )

    ds_val = create_tf_dataset(
        source_imgs_pth_val,
        source_masks_pth_val,
        target_imgs_pth_val,
        target_masks_pth_val,
        opt
    )
    return ds_train, ds_val


def create_callbacks(opt, sample):
    if not os.path.exists(f'{opt.ckpt_dir}/AdaptFCN_{opt.lambda_adv}'):
        os.makedirs(f'{opt.ckpt_dir}/AdaptFCN_{opt.lambda_adv}')

    if not os.path.exists(f'{opt.result_dir}/AdaptFCN_{opt.lambda_adv}'):
        os.makedirs(f'{opt.result_dir}/AdaptFCN_{opt.lambda_adv}')

    checkpoint = callbacks.ModelCheckpoint(
        filepath=f'{opt.ckpt_dir}/AdaptFCN_{opt.lambda_adv}/AdaptFCN',
        save_weights_only=True)

    history = callbacks.CSVLogger(
        f"{opt.result_dir}/AdaptFCN_{opt.lambda_adv}/AdaptFCN.csv",
        separator=",",
        append=False)

    visualization = VisualizeCallback(opt, sample)

    callbacks_ = [checkpoint, history, visualization]

    return callbacks_


class VisualizeCallback(callbacks.Callback):
    def __init__(self, opt, sample):
        super().__init__()
        self.opt = opt
        self.sample = sample

    def on_epoch_end(self, epoch, logs=None):
        source ,target = self.sample
        xs, ms = source
        xt, mt = target
        ms_pred = tf.cast(tf.argmax(self.model.fcn(xs[:self.opt.num_samples]), axis=-1)[..., None], 'float32')
        mt_pred = tf.cast(tf.argmax(self.model.fcn(xt[:self.opt.num_samples]), axis=-1)[..., None], 'float32')

        fig, ax = plt.subplots(nrows=self.opt.num_samples, ncols=4, figsize=(8, 8))
        titles = ['source', 'source pred', 'target', 'target pred']
        for i in range(self.opt.num_samples):
            for j, x in enumerate([xs, ms_pred, xt, mt_pred]):
                if j % 2:
                    ax[i, j].imshow(x[i], cmap='gray')
                else:
                    ax[i, j].imshow(x[i])
                ax[i, j].axis('off')
                ax[i, j].set_title(titles[j])

        if not os.path.exists(f'{self.opt.result_dir}/AdaptFCN_{self.opt.lambda_adv}/results'):
            os.makedirs(f'{self.opt.result_dir}/AdaptFCN_{self.opt.lambda_adv}/results')

        plt.savefig(f'{self.opt.result_dir}/AdaptFCN_{self.opt.lambda_adv}/results/{epoch}.png')
