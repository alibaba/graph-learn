from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
setup(
    name='ipcservice',
    ext_modules=[
        CUDAExtension('ipc_service', [
            'ipc_service.cpp',
            'helper_multiprocess.cpp',
            'ipc_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })