from electro_modelling.models.simple_dcgan import SimpleDCGAN
from electro_modelling.models.hinge_gan import HingeGAN
from electro_modelling.models.least_square_gan import LeastSquareGAN
from electro_modelling.models.wgan_gp import WGANGP


models = {
    "dcgan": SimpleDCGAN,
    "hgan": HingeGAN,
    "lsgan": LeastSquareGAN,
    "wgan": WGANGP,
}
