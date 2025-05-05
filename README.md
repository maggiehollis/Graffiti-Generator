# Graffiti-Generator

This repository uses a DCGAN model to generate unique AI-generated images of graffiti. The main training and generation processes are handled in `graffiti.py`, while larger image experiments are attempted in `graffiti_larger.py`.

---

## Generated Images

### Initial Images (`generated_images_300/`)

* **Training Set**: 300 graffiti samples
* **Resolution**: 32x32 pixels
* **Generated**: Every 500 epochs
* **Code Used**:

  ```python
     if epoch % 500 == 0:
        print(f"Epoch {epoch}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")
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

### Current Images (`generated_images_656/` and `output_656/`)

* **Training Set**: All 656 images
* **Resolution**: 32x32 pixels

#### `generated_images_656/`

* **Generated**: Every 500 epochs
* **Code Used**: *Same as above*

#### `output_656/`

* **Generated**: After Training (used to compute FID score)
* **Code Used**:

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
  ```

## Blog Figures

All figures in the blog post can be found in `blog_figures/` and are sourced from:

* `generated_images_300/`
* `generated_images_656/`
* `output_656/`
* One GIF created from the 128x128 image generation attempts
* Various images used for explaining our methodology 

## Larger Images (`graffiti_larger.py`)

* **Resolution**: 128x128 pixels
* **Limitation**: Computational resources were insufficient for full training
* **Code Used**:

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

The resulting images were turned into a GIF using an [online tool](https://ezgif.com/).

---

## Note

Due to the volume of generated images and storage limitations, not all outputs could be uploaded to this repository.
