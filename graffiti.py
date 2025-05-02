import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

input_directory = "photos"
output_directory = "photos_png"
save_dir = "generated_images_655"
batch_size = 64
epochs = 10000
latent_dim = 100

os.makedirs(output_directory, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


# Function to convert images to PNG and resize them
def convert_and_resize_images(input_directory, output_directory, size=(256, 256)):
    images = []
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(
                output_directory, filename.replace(".jpg", ".png")
            )

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                img_resized = img.resize(size)
                # Save as PNG
                img_resized.save(output_path, format="PNG")
                images.append(output_path)

    return images


# Function to sample images from the converted images
def sample_images(images, sample_size=656):
    sampled_images = random.sample(images, min(sample_size, len(images)))
    return sampled_images


# Function to load and decode images
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [32, 32])  # Resize image to desired size (32x32)
    return image


# Function to normalize image pixels to [-1, 1]
def preprocess_image(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


# Load all images from the file paths and apply preprocessing
def load_and_preprocess_images(image_paths):
    images = [load_image(image_path) for image_path in image_paths]
    images = [preprocess_image(image) for image in images]
    return tf.data.Dataset.from_tensor_slices(images)


# Build the generator model
def build_generator():
    model = tf.keras.Sequential()

    model.add(
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,))
    )  # Latent vector (100)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((8, 8, 256)))  # (8x8x256)

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )  # 16x16
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )  # 32x32
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv2DTranspose(
            3, (5, 5), strides=(1, 1), padding="same", use_bias=False, activation="tanh"
        )
    )

    return model


# Build the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[32, 32, 3]
        )
    )
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


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
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


# Training loop
print("Converting and resizing images...")
images = convert_and_resize_images(input_directory, output_directory)
sampled_images = sample_images(images)
train_dataset = (
    load_and_preprocess_images(sampled_images).shuffle(1000).batch(batch_size)
)

for epoch in range(epochs):
    print(".", end=" ", flush=True)
    for image_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch)

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}"
        )
        noise = tf.random.normal([16, 100])
        generated_images = generator(noise, training=False)
        generated_images = (generated_images + 1) / 2  # Scale to [0, 1]

        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i])
            plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"generated_image_{epoch}.png"))
        plt.close()

# Generate 655 final images and save them individually as PNGs
output_dir = "output_655"
os.makedirs(output_dir, exist_ok=True)

print("\nGenerating 655 individual images...")
num_images_to_generate = 655
generated_count = 0
batch_size_gen = 64

while generated_count < num_images_to_generate:
    current_batch_size = min(batch_size_gen, num_images_to_generate - generated_count)
    noise = tf.random.normal([current_batch_size, 100])
    generated_batch = generator(noise, training=False)
    generated_batch = (generated_batch + 1.0) * 127.5  # Scale to [0, 255]
    generated_batch = tf.cast(generated_batch, tf.uint8).numpy()

    for i in range(current_batch_size):
        img = Image.fromarray(generated_batch[i])
        img.save(os.path.join(output_dir, f"image_{generated_count + i + 1}.png"))

    generated_count += current_batch_size

print(f"Saved {generated_count} PNG images in '{output_dir}'.")
