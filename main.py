import numpy as np
import f

from skimage.io import imread, imsave, imshow
from tqdm import tqdm
from time import time

img_original = imread("data/mlisa.png")[:, :, :3]
C = f.init_C(5)
iterations = 10
alpha = 1

m, n = img_original.shape[:2]
img_reconstructed = np.zeros( (m, n, 3), dtype=np.uint8 )

start = time()
for channel in tqdm([0]):

    img = img_original[:, :, channel]
    img_reconstructed[:, :, channel] = f.reconstruct(img, C, alpha, iterations)

end = time()

imsave("res.png", img_reconstructed[:, :, 0])

print(f"[Input img info] shape: {img_original.shape} | dtype: {img_original.dtype} | min/max: {img_original.min(), img_original.max()}")
print(f"[Output img info] shape: {img_reconstructed.shape} | dtype: {img_reconstructed.dtype} | min/max: {img_reconstructed.min(), img_reconstructed.max()}")
print(f"[Time] : {end - start}")
