import tensorflow as tf
from tensorflow.keras import losses

bce = losses.BinaryCrossentropy(from_logits=True)

def adversarial_loss(critic_real, critic_fake, loss_tyle='logistic'):
  if loss_type == 'logistic':
    d_loss = bce(tf.ones_like(critic_real), critic_real) +\
             bce(tf.zeros_like(critic_fake), critic_fake)
    
    g_loss = bce(tf.ones_like(critic_fake), critic_fake)
    
  elif loss_type == 'ls':
    d_loss = tf.reduce_mean((1-critic_real) **2 + critic_fake ** 2)
    g_loss = tf.reduce_mean((1-critic_fake) ** 2)
    
  return d_loss, g_loss
