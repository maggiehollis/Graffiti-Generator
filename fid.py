from cleanfid import fid

real_dir = "photos"  # real images
gen_dir = "generated_images_655"  # generated images

fid_score = fid.compute_fid(real_dir, gen_dir, device="cpu", num_workers=0)
print(f"FID Score: {fid_score:.2f}")
