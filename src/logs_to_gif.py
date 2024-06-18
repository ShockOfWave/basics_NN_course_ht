import wandb
from PIL import Image
import os
from tqdm.auto import tqdm


api = wandb.Api()
run = api.run("shockofwave90/vae-gan-triplet-mnist/dvvor65w")

for file in tqdm(run.files()):
    if file.name.endswith(".png"):
        file.download(exist_ok=True)

real_images_paths, recon_images_paths, generated_images_paths, recon_val_images_paths, generated_val_images_paths = [], [], [], [], []

for root, dirs, files in os.walk("media"):
    for file in files:
        if file.endswith("png") and file.startswith("original_images"):
            real_images_paths.append(os.path.join(root, file))
        if file.endswith("png") and file.startswith("recon_images"):
            recon_images_paths.append(os.path.join(root, file))
        if file.endswith("png") and file.startswith("recon_val_images"):
            recon_val_images_paths.append(os.path.join(root, file))
        if file.endswith("png") and file.startswith("generated_images"):
            generated_images_paths.append(os.path.join(root, file))
        if file.endswith("png") and file.startswith("generated_val_images"):
            generated_val_images_paths.append(os.path.join(root, file))

def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=0.5, loop=0)


images_to_gif(real_images_paths, "images/real_images")
images_to_gif(recon_images_paths, "images/recon_images")
images_to_gif(recon_val_images_paths, "images/recon_val_images")
images_to_gif(generated_images_paths, "images/generated_images")
images_to_gif(generated_val_images_paths, "images/generated_val_images")
