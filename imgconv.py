import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as p
import matplotlib.cm as cm

mod = SourceModule(
        """
#include<stdio.h>
#define INDEX(a, b) a*256+b

__global__ void rgb2gray(float *dest,float *r_img, float *g_img, float *b_img)
{

unsigned int idx = threadIdx.x+blockIdx.x*blockDim.x;

  unsigned int a = idx/256;
  unsigned int b = idx%256;

dest[INDEX(a, b)] = (0.299*r_img[INDEX(a, b)]+0.587*g_img[INDEX(a, b)]+0.114*b_img[INDEX(a, b)]);

}

"""
    )

a = scipy.misc.imread('lena.jpg').astype(np.float32)

r_img = a[:, :, 0].reshape(65536, order='F')
g_img = a[:, :, 1].reshape(65536, order='F')
b_img = a[:, :, 2].reshape(65536, order='F')

dest=r_img
rgb2gray = mod.get_function("rgb2gray")
start = drv.Event()
end = drv.Event()
start.record()
rgb2gray(drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(64, 1, 1))
end.record()
end.synchronize()
secs = start.time_till(end)
print "The image is converted in %f ms and saved as converted.png" % (secs)

dest=np.reshape(dest,(256,256), order='F')
p.imshow(dest, cmap = matplotlib.cm.Greys_r)
p.show()
p.savefig('converted.png')












