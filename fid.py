import os
import shutil

from cleanfid import fid

# Make sure these paths point to folders containing images (not a .png)
real_dir = "photos"  # ← change this from 'photos.png/' to a real folder
gen_dir = "generated_images_655"

# Optional: You can crop/resize the images ahead of time, or let clean-fid handle it
# Set the resolution expected for FID (InceptionV3 uses 299x299 by default)

# Compute FID between two directories
fid_score = fid.compute_fid(gen=gen_dir, real=real_dir)
print(f"FID Score: {fid_score:.2f}")
