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
