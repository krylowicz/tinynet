from tinynet.constants import GPU
from tinynet.ops_cpu import *
if GPU:
    from tinynet.ops_gpu import *
