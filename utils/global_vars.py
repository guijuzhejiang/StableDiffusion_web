import time
from multiprocessing import cpu_count

import yaml

CONFIG = yaml.safe_load(open("config.yml", 'r'))
UTF_TD = int(time.localtime().tm_gmtoff/60/60)
CPU_NUM = cpu_count()

