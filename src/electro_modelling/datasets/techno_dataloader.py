# -*- coding: utf-8 -*-

"""
Module that defines train DataLoader on Techno dataset
"""

from torch.utils.data import DataLoader
from electro_modelling.datasets.techno_dataset import TechnoDatasetWavtoMel


def techno_data_loader(batch_size, data_dir, operator):
    """
    TODO: TO COMPLETE
    Attributes
    ----------

    Parameters
    ----------
    batch_size
    data_dir
    operator

    Returns
    -------

    """
    train_set = TechnoDatasetWavtoMel(
        operator, phase_method="griff", dat_location=data_dir
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
