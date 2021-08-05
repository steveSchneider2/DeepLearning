# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:39:22 2021

@author: steve
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
https://medium.com/lsc-psd/tensorflow-2-1-doesnt-seem-to-see-my-gpu-even-though-cuda-10-1-with-solution-7b44297843a
https://shawnhymel.com/1961/how-to-install-tensorflow-with-gpu-support-on-windows/



"""
# tensorflow breaks, unless you pip install tensorflow-estimator==2.1.*
import tensorflow as tf

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
import os
print('Conda Envronment:  ', os.environ['CONDA_DEFAULT_ENV'])
print(f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}')
print(f'Cuda Support:       {tf.test.is_built_with_cuda()}')
tf.test.gpu_device_name()
print(f'Tensor Flow:        {tf.version.VERSION}')
tf.version
import sys
import pandas as pd
import numpy as np
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))
print('Python version:      {}.'.format(pver)) 
print('The numpy version:   {}.'.format(np.__version__))
print('The panda version:   {}.'.format(pd.__version__))

print('Cuda compilation tools, release 11.1, V11.1.74'+
'Build cuda_11.1.relgpu_drvr455TC455_06.29069683_0')
print(''+
'cudatoolkit               11.0.3               h3f58a73_6    conda-forge\n' +
'cudnn                     8.0.5.39             hfe7f257_1    conda-forge\n'
)
# GPU information
import GPUtil
from tabulate import tabulate
print("="*40, "GPU Details", "="*40)
gpus = GPUtil.getGPUs()
list_gpus = []
for gpu in gpus:
    # get the GPU id
    gpu_id = gpu.id
    # name of GPU
    gpu_name = gpu.name
    # get % percentage of GPU usage of that GPU
    gpu_load = f"{gpu.load*100}%"
    # get free memory in MB format
    gpu_free_memory = f"{gpu.memoryFree}MB"
    # get used memory
    gpu_used_memory = f"{gpu.memoryUsed}MB"
    # get total memory
    gpu_total_memory = f"{gpu.memoryTotal}MB"
    # get GPU temperature in Celsius
    gpu_temperature = f"{gpu.temperature} °C"
    gpu_uuid = gpu.uuid
    list_gpus.append((
        gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
        gpu_total_memory, gpu_temperature, gpu_uuid
    ))

print(tabulate(list_gpus, 
               headers=("id", "name", "load", "free memory", "used memory", 
                        "total memory", "temperature", "uuid")))

