import os

from cleanfid import fid
from PIL import Image

real_dir = "photos"  # real images
gen_dir = "output_656"  # generated images

# make a new folder of all photos in "photos" as 32x32 pixel pngs
output_dir = "photos_32x32"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(real_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(real_dir, filename)
        img = Image.open(img_path)
        img = img.resize((32, 32))
        img.save(os.path.join(output_dir, os.path.splitext(filename)[0] + ".png"))

real_dir = output_dir  # Update real_dir to point to the resized images
fid_score = fid.compute_fid(real_dir, gen_dir, device="cpu", num_workers=0)
print(f"FID Score: {fid_score:.2f}")
