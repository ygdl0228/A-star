# @Time    : 2023/4/12 15:34
# @Author  : ygd
# @FileName: test.py
# @Software: PyCharm

from env import *

env_config = {'road_length': 1, 'YC_interval': 1, 'QC_interval': 1, 'buffer_length': 2, 'node_nums_x': 16,
              'node_nums_YC': 6, 'node_nums_QC': 3, 'max_speed': 6, 'min_speed': 1, 'max_acceleration': 1,
              'min_acceleration': -1}

test = route(**env_config)
state = test.reset()
print(test.AGV['destination'])