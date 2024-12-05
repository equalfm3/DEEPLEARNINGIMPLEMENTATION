import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Generate simple shapes dataset
def generate_shapes(num_samples=1000, size=28):
    shapes = np.zeros((num_samples, size, size, 1))
    for i in range(num_samples):
        shape_type = np.random.choice(['circle', 'square', 'triangle'])
        if shape_type == 'circle':
            center = (np.random.randint(size//4, 3*size//4), np.random.randint(size//4, 3*size//4))
            radius = np.random.randint(size//8, size//4)
            y, x = np.ogrid[:size, :size]
            dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)
            shapes[i, :, :, 0] = dist_from_center <= radius
        elif shape_type == 'square':
            top_left = (np.random.randint(0, size//2), np.random.randint(0, size//2))
            side = np.random.randint(size//4, size//2)
            shapes[i, top_left[0]:top_left[0]+side, top_left[1]:top_left[1]+side, 0] = 1
        else:  # triangle
            points = np.random.randint(0, size, size=(3, 2))
            x, y = np.meshgrid(range(size), range(size))
            shapes[i, :, :, 0] = (((points[1, 1] - points[0, 1]) * (x - points[0, 0]) - 
                          (points[1, 0] - points[0, 0]) * (y - points[0, 1])) >= 0) & \
                        (((points[2, 1] - points[1, 1]) * (x - points[1, 0]) - 
                          (points[2, 0] - points[1, 0]) * (y - points[1, 1])) >= 0) & \
                        (((points[0, 1] - points[2, 1]) * (x - points[2, 0]) - 
                          (points[0, 0] - points[2, 0]) * (y - points[2, 1])) >= 0)
    return shapes.astype(np.float32)

# Define the VAE model
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = keras.layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = keras.layers.Reshape((7, 7, 64))(x)
        x = keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

# Training function
def train_vae(vae, x_train, epochs, batch_size):
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

# Generate images
def generate_images(vae, n=10):
    z_sample = np.random.normal(size=(n, vae.latent_dim))
    x_decoded = vae.decoder.predict(z_sample)
    return x_decoded

# Interpolate between two points in latent space
def interpolate_images(vae, img1, img2, n_steps=10):
    z_1 = vae.encoder.predict(img1[np.newaxis, ...])[2]
    z_2 = vae.encoder.predict(img2[np.newaxis, ...])[2]
    z_interp = np.zeros((n_steps, vae.latent_dim))
    for i in range(n_steps):
        z_interp[i] = z_1 + (z_2 - z_1) * i / (n_steps - 1)
    x_interp = vae.decoder.predict(z_interp)
    return x_interp

if __name__ == "__main__":
    # Generate dataset
    x_train = generate_shapes(10000)

    # Create and train VAE
    vae = VAE(latent_dim=2)
    train_vae(vae, x_train, epochs=50, batch_size=128)

    # Generate new images
    generated_images = generate_images(vae, n=5)

    # Plot generated images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.savefig('vae_generated_shapes.png')
    plt.close()

    # Interpolate between two images
    img1, img2 = x_train[0], x_train[1]
    interpolated_images = interpolate_images(vae, img1, img2, n_steps=10)

    # Plot interpolated images
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.savefig('vae_interpolated_shapes.png')
    plt.close()

    print("Training completed. Generated and interpolated images saved.")