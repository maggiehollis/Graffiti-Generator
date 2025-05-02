import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

input_directory = "photos"
output_directory = "photos_png_larger"
save_dir = "generated_images_128"
batch_size = 64
epochs = 10000
latent_dim = 100

os.makedirs(output_directory, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


# Image Conversion & Preprocessing
def convert_and_resize_images(input_dir, output_dir, size=(256, 256)):
    images = []
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                img_resized = img.resize(size)
                # Save as PNG
                img_resized.save(output_path, format="PNG")
                images.append(output_path)

    return images


# Function to sample images from the converted images
def sample_images(images, sample_size=655):
    return random.sample(images, min(sample_size, len(images)))


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])  # CHANGED to (128, 128)
    return image


def preprocess_image(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


def load_and_preprocess_images(image_paths):
    images = [preprocess_image(load_image(path)) for path in image_paths]
    return tf.data.Dataset.from_tensor_slices(images)


def build_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(8 * 8 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((8, 8, 1024)))  # (8, 8, 512)

    model.add(
        layers.Conv2DTranspose(
            512, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )  # 16x16
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            256, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )  # 16x16
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )  # 32x32
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )  # 128x128

    return model


def build_discriminator():
    model = tf.keras.Sequential()

    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[128, 128, 3]
        )
    )
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(
        tf.zeros_like(fake_output), fake_output
    )


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Training Step
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_optimizer.apply_gradients(
        zip(
            gen_tape.gradient(gen_loss, generator.trainable_variables),
            generator.trainable_variables,
        )
    )
    discriminator_optimizer.apply_gradients(
        zip(
            disc_tape.gradient(disc_loss, discriminator.trainable_variables),
            discriminator.trainable_variables,
        )
    )

    return gen_loss, disc_loss


# Training Loop
print("Converting and resizing images...")
images = convert_and_resize_images(input_directory, output_directory)
sampled_images = sample_images(images, sample_size=300)
train_dataset = (
    load_and_preprocess_images(sampled_images).shuffle(1000).batch(batch_size)
)

for epoch in range(epochs):
    print(".", end=" ", flush=True)
    for batch in train_dataset:
        gen_loss, disc_loss = train_step(batch)

    if epoch % 10 == 0:
        print(
            f"\nEpoch {epoch}: Gen Loss = {gen_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}"
        )
        noise = tf.random.normal([16, latent_dim])
        generated_images = generator(noise, training=False)
        generated_images = (generated_images + 1) / 2  # Scale to [0, 1]

        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i].numpy())
            plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"generated_image_{epoch}.png"))
        plt.close()
