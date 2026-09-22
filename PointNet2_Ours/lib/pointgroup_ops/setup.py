import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if os.name == 'nt':
    cxx_flags = ['/std:c++20', '/Zc:preprocessor']
    nvcc_flags = ['-O2', '-std=c++20', '-allow-unsupported-compiler', '-Xcompiler=/Zc:preprocessor']
    include_dirs = [os.path.join(sys.prefix, 'Library', 'include', 'windows'), os.path.join(sys.prefix, 'Library', 'include')]
else:
    cxx_flags = ['-O2', '-std=c++20']
    nvcc_flags = ['-O2', '-std=c++20', '-allow-unsupported-compiler']
    include_dirs = []

setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension('PG_OP', [
            'src/pointgroup_ops_api.cpp',
            'src/pointgroup_ops.cpp',
            'src/cuda.cu',
        ], include_dirs=include_dirs,
           extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags})
    ],
    cmdclass={'build_ext': BuildExtension},
)


