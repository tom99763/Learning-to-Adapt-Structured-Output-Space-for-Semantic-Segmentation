from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras import layers, metrics
from losses import *

class FCN(tf.keras.Model):
  def __init__(self, opt):
    super().__init__()
    self.fcn = self.build_fcn(opt)
    
  def call(self, x, training=False):
    return self.fcn(x, training=training)
    
  def build_fcn(self, opt):
    
    #vgg16
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
    
    #merge feautres & prediction
    f5_conv3_x2 = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')(f5)  #f5 feature
    f4_conv1 = layers.Conv2D(filters=opt.num_classes, kernel_size=1, padding='same', activation=None)(f4) #f4 feature
    merge1 = tf.add(f4_conv1, f5_conv3_x2) #fuse the result of f4 & f5 feautre
    merge1_x2 = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')(merge1) #f4 & f5 fused 
    f3_conv1 = layers.Conv2D(filters=opt.num_classes, kernel_size=1, padding='same', activation=None)(f3) #f3
    merge2 = tf.add(f3_conv1, merge1_x2) #fuse f3 & f4 * f5
    outputs = layers.Conv2DTranspose(filters=opt.num_classes, kernel_size=16, strides=8,
                              padding='same', activation=None)(merge2)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, f5])
  
  
class Discriminator(tf.keras.Model):
  def __init__(self, opt):
    super().__init__()
    self.disc = tf.keras.Sequential([
      layers.Conv2D(filters = opt.base, kernel_size =4, strides=2, padding='same'), 
      layers.LeakyReLU(0.2)
    ])
    for i in range(1, opt.num_downsamples):  
      self.disc.add(layers.Conv2D(filters = opt.base * 2 **i, kernel_size =4, strides=2, padding='same')) 
      self.disc.add(layers.LeakyReLU(0.2))
      
    self.disc.add(layers.Conv2D(filters = 1, kernel_size =4, strides=2, padding='same'))
    
  def call(self, x):
    return self.disc(x)
  

class AdaptFCN(tf.keras.Model):
  def __init__(self, opt):
    super().__init__()
    self.fcn = FCN(opt)
    self.disc = Discriminator(opt)
    self.opt = opt
    
  def call(self, x, training=False):
    m_pred, f =self.fcn(x, training=training)
    return m_pred, f
  
  @tf.function
  def train_step(self, source, target):
    xs, ms = source
    xt, mt = target
    with tf.GradientTape(persistent=True) as tape:
      #predict mask
      ms_pred, fs = self.call(xs, True)
      mt_pred, ft = self.call(xt, True)
      
      #predict domain
      critic_source = self.disc(fs)
      critic_target = self.disc(ft)
      
      #loss functions
      l_cls = crossentropy(ms_pred, ms)
      l_d, l_g = adversarial_loss(critic_source, critic_target, self.opt.gan_loss)
      g_loss = self.opt.lambda_cls * l_cls + self.opt.lambda_adv * l_g
      d_loss = self.opt.lambda_adv * l_d
      
    g_grad=tape.gradient(g_loss, self.fcn.trainable_weights)
    d_grad=tape.gradient(d_loss, self.disc.trainable_weights)
    
    self.optimizer[0].apply_gradients(zip(g_grad, self.fcn.trainable_weights))
    self.optimizer[1].apply_gradients(zip(d_grad, self.disc.trainable_weights))
    
    #compute metrics
    
    return {'l_cls':l_cls, 'l_g':l_g, 'l_d':l_d}
  
  @tf.function
  def test_step(self, source, target):
    xs, ms = source
    xt, mt = target
    
    #predict mask
    ms_pred, fs = self.call(xs)
    mt_pred, ft = self.call(xt)

    #loss functions
    l_cls = crossentropy(ms_pred, ms)
    
    return {'l_cls':l_cls}
