from enum import Enum

"""
Add new Autoencoder classes here. 
"""


class ModelType(Enum):
    AE = 1
    CNN_AE = 2
    DSAE = 3  # Deep Spatial AE
    VAE = 4  # Variational Autoencoder
