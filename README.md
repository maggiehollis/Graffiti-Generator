# Graffiti-Generator

This repoistory uses a DCGAN model to attempt to generate unique AI-generated images of graffiti. Running `graffiti.py` generates the images.

## Initial Images
Initial images were created on only 300 sample images and are 32x32 pixels. You can view results (generated every 500 epochs of training) in `generated_images_300`

## Current Images
Our most promising images are created on all 656 images we had to train with. They are also 32x32 pixels and results can be viewed in 
- `generated_images_656` generated every 500 epochs
- `output_656` generated after training and used to compute FID value

## Larger Images
We attempted to generate images that were 128x128. Unfortunaely with the computational power we had, this was not feasible. The code we used is in `graffiti_larger.py`

## Blog Images
During training, `generated_images_300` and `generated_images_656` were created from the same block of code

```python
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
```
`output_656` was generated after training was completed in `graffiti.py`
```python
output_dir = "output_656"
os.makedirs(output_dir, exist_ok=True)

print("\nGenerating 656 individual images...")
num_images_to_generate = 656
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
```
In `blog_figures` you can view all the images we included in out blog post. These images are all from one of the previously mentioned folders. We did include one gif from our attempts to generate larger images. These images were generated from the following code block in `graffiti_larger.py` and were turned into a gif using an [online gif generator]([url](https://ezgif.com/maker)).
```python
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
```
Due to the amount of generated images and incomplete training, we were not able to upload them to our repository.
