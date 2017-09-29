import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

nofblocks = 64
nofthread = 128
arraysize = nofblocks * nofthread

print "Using array size ==", arraysize
n_iter = 100000
print "Calculating %d iterations" % (n_iter)

start = drv.Event()
end = drv.Event()

mod = SourceModule("""
__global__ void gpukernel(float *dest, float *a, int n_iter)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  for(int n = 0; n < n_iter; n++) {
    a[i] = sin(a[i]);
  }
  dest[i] = a[i];
}
""")

gpukernel = mod.get_function("gpukernel")

a = numpy.ones(arraysize).astype(numpy.float32)
dest = numpy.zeros_like(a)

start.record()
gpukernel(drv.Out(dest), drv.In(a), numpy.int32(n_iter), grid=(nofblocks,1), block=(nofthread,1,1) )
end.record()
end.synchronize()
secs = start.time_till(end)
print "GPU kernel time and first three results:"
print "%fms, %s" % (secs, str(dest[:3]))

a = numpy.ones(arraysize).astype(numpy.float32)
start.record()
start.synchronize()
for i in range(n_iter):
    a = numpy.sin(a)
end.record()
end.synchronize()

secs = start.time_till(end)
print "CPU time and first three results:"
print "%fms, %s" % (secs, str(a[:3]))
del a
del dest