import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from schedule_base import Schedule

class Full(Schedule):
    def __init__(self,
        args=None
    ):
        super(Full, self).__init__(
            args,
        )

    def initialize_labeled_data(self):
        """initialize labeled data as full data"""
        self.labeled_idx[:] = True
