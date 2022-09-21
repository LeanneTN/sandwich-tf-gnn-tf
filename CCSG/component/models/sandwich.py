"""
Sandwich Model
author: Liang Tian
date: 2022-9-21
"""
import copy
import math
import logging
from tqdm import tqdm
import torch.nn as nn
import torch

import torch
import torch.optim as optim

import component.config
from torch.nn.utils import clip_grad_norm_

import config
from component.config import override_model_args
from component.models.transformer import Transformer
from component.utils.copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from component.utils.misc import tens2sen, count_file_lines


logger = logging.getLogger(__name__)

class Sandwich(nn.Module):

    def __init__(self):
        """
        constructor of the class
        """
        super(Sandwich, self).__init__()
        self.name = 'Sandwich'


