import math
import numpy as np

from scipy.ndimage import gaussian_filter1d

# input=[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
#
# kernel = []
# sigma=2
#
# denominator=0
# for i in range(-2,3):
#     value = 1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i**2)/(2*sigma**2))
#     kernel.append(value)
#     denominator+=value
#
# # kernel = [k/denominator for k in kernel]
#
# print(kernel)
# # for i in range(2,7):
# #     results=
#
# base_kernel = [0.0,0.0,1.0,0.0,0.0]
# base_kernel = gaussian_filter1d(base_kernel,sigma,mode="constant")
# print(base_kernel)

img=np.random.random((3,10,10))
print(img.shape)
img=img.astype(np.float32)
print(img.dtype)
