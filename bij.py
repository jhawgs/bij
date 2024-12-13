import os
from glob import glob

def clear():
    os.system("rm -r ./checkpoints")
    os.system("rm -r ./logs")
    os.system("rm -r ./_r")
    os.system("rm -r ./_g")
    os.system("rm -r ./_s")

    os.system("mkdir ./_r")
    os.system("mkdir ./_g")
    os.system("mkdir ./_s")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

POINT_SIZE = 5
ALPHA = .6
LINE_WIDTH = 0
INTERVAL = 10
SCATTER_XLIM = (-1.5, 1.5)
SCATTER_YLIM = (-1.5, 1.5)
RG_LIMS = (-1.5, 1.5)

class RGCallback(tf.keras.callbacks.Callback):
    def __init__(self, images, interval=INTERVAL, lims=(-3, 3)):
        super().__init__()
        self.images = images
        self.interval = interval
        self.lims = lims

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
          x1 = np.expand_dims(self.images[np.random.choice(np.arange(self.images.shape[0]))], 0)
          x2 = np.expand_dims(self.images[np.random.choice(np.arange(self.images.shape[0]))], 0)
          v1 = self.model.encode(x1)
          y1 = self.model.decode(v1)
          v2 = self.model.encode(x2)
          y2 = self.model.decode(v2)
          f, axarr = plt.subplots(2,2)
          axarr[0,0].imshow(x1[0])
          axarr[0,1].imshow(y1[0])
          axarr[1,0].imshow(x2[0])
          axarr[1,1].imshow(y2[0])
          plt.savefig("./_r/{}.png".format(epoch))
          plt.clf()
          plt.close()

          size = 5
          f, axarr = plt.subplots(size,size)
          ivl = np.linspace(self.lims[0], self.lims[1], size)
          for r in range(size):
              for c in range(size):
                  axarr[r,c].imshow(model.decode(np.array([[ivl[r], ivl[c]]]))[0])
          plt.savefig("./_g/{}.png".format(epoch))
          plt.clf()
          plt.close()

class SCallback(tf.keras.callbacks.Callback):
    def __init__(self, images, labels, interval=INTERVAL, xlim=(-4, 4), ylim=(-4, 4)):
        super().__init__()
        self.images = images
        self.labels = labels
        self.interval = interval
        self.xlim = xlim
        self.ylim = ylim

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
          v = model.encode(self.images)
          plt.scatter(v[:, 0], v[:, 1], c=self.labels, s=POINT_SIZE, alpha=ALPHA, linewidths=LINE_WIDTH)
          plt.gca().set_xlim(*self.xlim)
          plt.gca().set_ylim(*self.ylim)
          plt.savefig("./_s/{}.png".format(epoch))
          plt.clf()
          plt.close()

class GlobalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads=2, key_dim1=28, key_dim2=28, dropout=.25):
    super().__init__()
    self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim1, dropout=dropout)
    self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim2, dropout=dropout)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    self.layernorm1 = tf.keras.layers.LayerNormalization()#tf.keras.layers.LayerNormalization()
    self.layernorm2 = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    self.positional_encoding = tf.keras.layers.Embedding(key_dim1, key_dim2)
    self.enc_temp = tf.range(key_dim1)
    self.permute = tf.keras.layers.Permute((2, 1, 3))

  def call(self, x):
    x = self.add([x, self.positional_encoding(self.enc_temp)[tf.newaxis, :, :, tf.newaxis]])
    attn_output = self.mha1(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm1(x)
    x = self.batchnorm1(x)
    x = self.permute(x)
    attn_output = self.mha2(
        query=x,
        value=x,
        key=x
    )
    x = self.add([x, attn_output])
    x = self.layernorm2(x)
    x = self.batchnorm2(x)
    x = self.permute(x)
    return x

class PGlobalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads=2, key_dim1=28, key_dim2=28, key_dim3=1, dropout=.25):
    super().__init__()
    self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim1, dropout=dropout)
    self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim2, dropout=dropout)
    self.batchnorm = tf.keras.layers.BatchNormalization()
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    self.positional_encoding1 = tf.keras.layers.Embedding(key_dim1, key_dim3)
    self.positional_encoding2 = tf.keras.layers.Embedding(key_dim2, key_dim3)
    self.enc_temp1 = tf.range(key_dim1)
    self.enc_temp2 = tf.range(key_dim2)
    self.permute = tf.keras.layers.Permute((2, 1, 3))

  def call(self, x):
    x = self.add([x, self.positional_encoding1(self.enc_temp1)[tf.newaxis, :, tf.newaxis, :], self.positional_encoding2(self.enc_temp2)[tf.newaxis, tf.newaxis, :, :]])
    attn_output1 = self.mha1(
        query=x,
        value=x,
        key=x)
    xt = self.permute(x)
    attn_output2 = self.mha2(
        query=x,
        value=x,
        key=x
    )
    attn_output2 = self.permute(attn_output2)
    x = self.add([x, attn_output1, attn_output2])
    x = self.layernorm(x)
    x = self.batchnorm(x)
    return x

#GlobalSelfAttention(num_heads=2, key_dim=512)

class VSAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name="eo_input"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            PGlobalSelfAttention(key_dim1=13, key_dim2=13, key_dim3=32),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            PGlobalSelfAttention(key_dim1=6, key_dim2=6, key_dim3=64),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,), name="de_input"),
            tf.keras.layers.Dense(units=7*7*32, activation="leaky_relu"),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2,
                padding='same', activation='leaky_relu'
            ),
            tf.keras.layers.BatchNormalization(),
            PGlobalSelfAttention(key_dim1=14, key_dim2=14, key_dim3=64),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2,
                padding='same', activation='leaky_relu'
            ),
            tf.keras.layers.BatchNormalization(),
            PGlobalSelfAttention(key_dim1=28, key_dim2=28, key_dim3=32),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1,
                padding='same', activation="tanh"
            )
        ])
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name="dc_input"),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3,
                strides=(2, 2), activation='leaky_relu'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.2),
            PGlobalSelfAttention(key_dim1=13, key_dim2=13, key_dim3=32),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3,
                strides=(2, 2), activation='leaky_relu'
            ),
            PGlobalSelfAttention(key_dim1=6, key_dim2=6, key_dim3=64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
        self.bijector = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 2), name="bij_input"),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3,
                strides=(2, 2), activation='leaky_relu'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.2),
            PGlobalSelfAttention(key_dim1=13, key_dim2=13, key_dim3=32),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3,
                strides=(2, 2), activation='leaky_relu'
            ),
            PGlobalSelfAttention(key_dim1=6, key_dim2=6, key_dim3=64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
        self.flat = tf.keras.layers.Flatten()
        self.ed_loss_tracker = tf.keras.metrics.Mean(name="ed_loss")
        self.de_loss_tracker = tf.keras.metrics.Mean(name="de_loss")
        self.dc_loss_tracker = tf.keras.metrics.Mean(name="dc_loss")
        self.coherence_tracker = tf.keras.metrics.Mean(name="coh")
        self.closure_tracker = tf.keras.metrics.Mean(name="clo")

    def compile(self, eo, deo, dco, bio):
        super().compile()
        self.encoder_optimizer = eo
        self.decoder_optimizer = deo
        self.discriminator_optimizer = dco
        self.bijector_optimizer = bio
        self.lmse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self.lbce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def biject(self, x1, x2):
       return self.bijector(tf.concat([x1, x2], axis=-1))

    def clo_call(self, x):
        v = tf.random.normal((tf.shape(x)[0], self.latent_dim))
        y = self.decode(v)
        p_x = self.discriminate(x)
        p_y = self.discriminate(y)
        return p_x, p_y

    def coh_call(self, x):
        n = tf.shape(x)[0]
        v = tf.random.normal((n, self.latent_dim))
        y = self.decode(v)
        v_prime = self.encode(y)

        v_x = tf.stop_gradient(self.encode(x))
        y_x = self.decode(v_x)
        v_x_prime = self.encode(y_x)
        v = tf.concat([v, v_x], axis=0)
        v_prime = tf.concat([v_prime, v_x_prime], axis=0)

        b_idx = tf.maximum(0, tf.cast((tf.random.uniform((n,)) * 2 * tf.cast(n, tf.float32)) - tf.cast(n, tf.float32), tf.int32))
        b = tf.cast(b_idx == 0, tf.float32)
        b = b * .8 + .1
        b_prime = self.biject(x, tf.gather(y_x, (b_idx + tf.range(n)) % n, axis=0))

        return v, v_prime, b, b_prime

    def train_call(self, x):
      self.clo_call(x)
      self.coh_call(x)
    def test_call(self, x):
      return self.train_call(x)

    def train_step(self, data):
        x = data
        n = tf.cast(tf.shape(x)[0], tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            p_x, p_y = self.clo_call(x)
            closure = self.lbce(tf.zeros_like(p_y) + .1, p_y)/n
            discriminator_loss = self.lbce(tf.concat((
                tf.ones_like(p_y) - .1,
                tf.zeros_like(p_x) + .1
            ), axis=0), tf.concat((
                p_y,
                p_x
            ), axis=0))/n/2.
            decoder_loss = closure

        decoder_vars = self.decoder.trainable_variables
        discriminator_vars = self.discriminator.trainable_variables

        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(decoder_loss, decoder_vars), decoder_vars)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(tape.gradient(discriminator_loss, discriminator_vars), discriminator_vars)
        )
        del tape

        _d = decoder_loss

        with tf.GradientTape(persistent=True) as tape:
            v, v_prime, b, b_prime = self.coh_call(x)
            coherence = self.lmse(v, v_prime)/n/2.
            bijector_loss = self.lbce(b, b_prime)
            encoder_loss = coherence
            decoder_loss = coherence + bijector_loss

        encoder_vars = self.encoder.trainable_variables
        decoder_vars = self.decoder.trainable_variables
        bijector_vars = self.bijector.trainable_variables

        self.encoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_loss, encoder_vars), encoder_vars)
        )
        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(decoder_loss, decoder_vars), decoder_vars)
        )
        self.bijector_optimizer.apply_gradients(
           zip(tape.gradient(bijector_loss, bijector_vars), bijector_vars)
        )
        del tape

        self.ed_loss_tracker.update_state(encoder_loss)
        self.de_loss_tracker.update_state(decoder_loss + _d)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.coherence_tracker.update_state(bijector_loss)
        self.closure_tracker.update_state(closure)

        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "de_loss": self.de_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "bij": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result()
        }

    def test_step(self, data):
        x = data
        n = tf.cast(tf.shape(x)[0], tf.float32)
        p_x, p_y = self.clo_call(x)
        v, v_prime, b, b_prime = self.coh_call(x)
        closure = self.lbce(tf.zeros_like(p_y), p_y)/n
        discriminator_loss = self.lbce(tf.concat((
            tf.ones_like(p_y),
            tf.zeros_like(p_x)
        ), axis=0), tf.concat((
            p_y,
            p_x
        ), axis=0))/n/2.
        coherence = self.lmse(v, v_prime)/n/2.
        bijector_loss = self.lbce((b - .1)/.8, b_prime)
        encoder_loss = coherence
        decoder_loss = coherence + bijector_loss

        self.ed_loss_tracker.update_state(encoder_loss)
        self.de_loss_tracker.update_state(decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.coherence_tracker.update_state(bijector_loss)
        self.closure_tracker.update_state(closure)

        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "de_loss": self.de_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "bij": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result()
        }

NO_TRANSFORMER = False

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 127.5
  return (images - 1).astype('float32')#np.where(images > .5, 1.0, 0.0).astype('float32')

if __name__ == "__main__":
  BATCH_SIZE = 128
  _ckpt = sorted(glob("./checkpoints/*.weights.h5"), key=lambda x: int(x.split("/")[-1].split("-")[0]))
  _ckpt = _ckpt[-1] if len(_ckpt) > 0 else None
  with tf.device("/GPU:0"):
    (train_images, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size = 60000
    batch_size = BATCH_SIZE
    test_size = 10000
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(batch_size, drop_remainder=True)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(batch_size, drop_remainder=True)
    )

    model = VSAE(2)
    model.compile(
        eo=tf.keras.optimizers.Adam(1e-4, beta_1=.5),
        deo=tf.keras.optimizers.Adam(1e-4, beta_1=.5),
        dco=tf.keras.optimizers.Adam(1e-4, beta_1=.5),
        bio=tf.keras.optimizers.Adam(1e-4, beta_1=.5)
    )
    model.built = True
    if _ckpt is not None:
        model.load_weights(_ckpt)
    model.summary()
    clear()
    tbcb = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_graph=False)
    ckcb = tf.keras.callbacks.ModelCheckpoint("./checkpoints/{epoch:02d}-.weights.h5", save_weights_only=True, save_freq=20*400)
    rgcb = RGCallback(test_images, lims=RG_LIMS)
    sccb = SCallback(test_images[:8000], test_labels[:8000], xlim=SCATTER_XLIM, ylim=SCATTER_YLIM)
    model.fit(train_dataset, validation_data=test_dataset, epochs=56000, callbacks=[tbcb, ckcb, rgcb, sccb])
