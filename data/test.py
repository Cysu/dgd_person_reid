from __future__ import absolute_import
import numpy as np
from argparse import ArgumentParser
from scipy.misc import imsave
from .utils import *
# # add the dgd_root_path to Python environment to "import utils"
# import sys,os
# dgd_root_path = os.path.abspath('.')
# sys.path.insert(0,dgd_root_path)
# print sys.path

mkdir_if_missing()