"""
This module reads and transforms the observation data in a python object that
the module of calibration can understand.
"""

import equinox as eqx

class Data(eqx.Module):
    def __init__(obs_info_file_path):
        pass