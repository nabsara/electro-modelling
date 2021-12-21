from electro_modelling.models.dcgan import DCGAN


class HingeGAN(DCGAN):

    def __init__(self, z_dim):
        super().__init__(
            z_dim=z_dim,
            model_name="hinge_dcgan",
            init_weights=True
        )

    def _init_optimizer(self, learning_rate):
        pass

    def _init_criterion(self):
        pass

    def _compute_disc_loss(self, real, fake, disc_real_pred, disc_fake_pred):
        pass

    def _compute_gen_loss(self, disc_fake_pred):
        pass
