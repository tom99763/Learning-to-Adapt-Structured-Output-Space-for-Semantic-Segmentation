from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras import layers, metrics
import sys
sys.path.append('./models')
from losses import *


class FCN(tf.keras.Model):
    def __init__(self, opt):
        super().__init__()
        self.fcn = self.build_fcn(opt)

    def call(self, x, training=False):
        return self.fcn(x, training=training)

    def build_fcn(self, opt):
        # vgg16
        inputs = layers.Input(shape=(opt.img_size, opt.img_size, 3), name='input')
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
        f3 = vgg16.get_layer('block3_pool').output
        f4 = vgg16.get_layer('block4_pool').output
        f5 = vgg16.get_layer('block5_pool').output

        # Replacing VGG dense layers by convolutions
        f5_conv1 = layers.Conv2D(filters=4096, kernel_size=7, padding='same',
                                 activation='relu')(f5)
        f5_drop1 = layers.Dropout(0.5)(f5_conv1)
        f5_conv2 = layers.Conv2D(filters=4096, kernel_size=1, padding='same',
                                 activation='relu')(f5_drop1)
        f5_drop2 = layers.Dropout(0.5)(f5_conv2)
        f5_conv3 = layers.Conv2D(filters=opt.num_classes, kernel_size=1, padding='same',
                                 activation=None)(f5_drop2)

        # merge feautres & prediction
        f5_conv3_x2 = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=4, strides=2,
                                             use_bias=False, padding='same', activation='relu')(f5_conv3)  # f5 feature
        f4_conv1 = layers.Conv2D(filters=opt.num_classes, kernel_size=1, padding='same', activation=None)(
            f4)  # f4 feature
        merge1 = tf.add(f4_conv1, f5_conv3_x2)  # fuse the result of f4 & f5 feature
        merge1_x2 = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=4, strides=2,
                                           use_bias=False, padding='same', activation='relu')(merge1)  # f4 & f5 fused
        f3_conv1 = layers.Conv2D(filters=opt.num_classes, kernel_size=1, padding='same', activation=None)(f3)  # f3
        merge2 = tf.add(f3_conv1, merge1_x2)  # fuse f3 & f4 * f5
        outputs = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=16, strides=8,
                                         padding='same', activation=None)(merge2)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class Discriminator(tf.keras.Model):
    def __init__(self, opt):
        super().__init__()
        self.disc = tf.keras.Sequential([
            layers.Conv2D(filters=opt.base, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2)
        ])
        for i in range(1, opt.num_downsamples):
            self.disc.add(layers.Conv2D(filters=opt.base * 2 ** i, kernel_size=4, strides=2, padding='same'))
            self.disc.add(layers.LeakyReLU(0.2))

        self.disc.add(layers.Conv2D(filters=1, kernel_size=4, strides=2, padding='same'))

    def call(self, x):
        return self.disc(x)


class AdaptFCN(tf.keras.Model):
    def __init__(self, opt):
        super().__init__()
        self.fcn = FCN(opt)
        self.disc = Discriminator(opt)
        self.opt = opt
        self.metrics_=[metrics.MeanIoU(opt.num_classes), metrics.MeanIoU(opt.num_classes)]

    def call(self, x, training=False):
        m_pred = self.fcn(x, training=training)
        return m_pred

    @tf.function
    def train_step(self, inputs):
        source, target = inputs
        xs, ms = source
        xt, mt = target
        with tf.GradientTape(persistent=True) as tape:
            # predict mask
            ms_pred = self.call(xs, True)
            mt_pred = self.call(xt, True)

            # predict domain
            critic_source = self.disc(tf.nn.softmax(ms_pred, axis=-1))
            critic_target = self.disc(tf.nn.softmax(mt_pred, axis=-1))

            # loss functions
            l_cls = crossentropy(ms_pred, ms)
            l_d, l_g = adversarial_loss(critic_source, critic_target, self.opt.gan_loss)
            g_loss = self.opt.lambda_cls * l_cls + self.opt.lambda_adv * l_g
            d_loss = self.opt.lambda_adv * l_d

        g_grad = tape.gradient(g_loss, self.fcn.trainable_weights)
        d_grad = tape.gradient(d_loss, self.disc.trainable_weights)

        self.optimizer[0].apply_gradients(zip(g_grad, self.fcn.trainable_weights))
        self.optimizer[1].apply_gradients(zip(d_grad, self.disc.trainable_weights))

        # compute metrics
        history = {'l_cls': l_cls, 'l_g': l_g, 'l_d': l_d}
        ms_pred = self.call(xs)
        mt_pred = self.call(xt)
        self.metrics_[0].update_state(ms, tf.nn.softmax(ms_pred, axis=-1))
        self.metrics_[1].update_state(mt, tf.nn.softmax(mt_pred, axis=-1))
        history['mIoU_source'] = self.metrics_[0].result()
        history['mIoU_target'] = self.metrics_[1].result()
        self.reset_metrics()
        return history

    @tf.function
    def test_step(self, inputs):
        source, target = inputs
        xs, ms = source
        xt, mt = target

        # compute metrics
        history = {}
        ms_pred = self.call(xs)
        mt_pred = self.call(xt)
        self.metrics_[0].update_state(ms, tf.nn.softmax(ms_pred, axis=-1))
        self.metrics_[1].update_state(mt, tf.nn.softmax(mt_pred, axis=-1))
        history['mIoU_source'] = self.metrics_[0].result()
        history['mIoU_target'] = self.metrics_[1].result()
        self.reset_metrics()
        return history
