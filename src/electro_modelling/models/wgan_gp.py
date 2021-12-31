import torch

from electro_modelling.config import settings
from electro_modelling.models.dcgan import DCGAN


class WGANGP(DCGAN):
    """
    Note: To inherit from DCGAN and benefit from common class methods,
    we keep the discriminator class attributes though it corresponds
    to the critic model here !
    i.e. self.discriminator, self.disc_opt,
    """

    def __init__(self, z_dim, dataset, img_chan, operator=None):
        super().__init__(
            z_dim=z_dim,
            model_name="wgan",
            init_weights=True,
            dataset=dataset,
            img_chan=img_chan,
            operator=operator,
        )

    def _init_optimizer(self, learning_rate, beta_1=0, beta_2=0.9):
        # TODO: Check with RMS Prop cf. W-GAN with weights clipping paper
        self.gen_opt = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )

    def _get_gradient(self, real, fake, epsilon):
        """
        Return the gradient of the critic's scores with respect to mixes of real
        and fake images.

        Parameters
        ----------
        real :
            current batch of real images
        fake :
            current batch of fake images
        epsilon : np.array
            vector of the uniformly random proportions of
            real / fake per mixed image
        """
        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)
        # Calculate the critic's scores on the mixed images
        mixed_scores = self.discriminator(mixed_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    def _compute_disc_loss(
        self,
        real,
        fake,
        disc_real_pred,
        disc_fake_pred,
        c_lambda=10,
        real_score_penalty_weight=0.001,
    ):
        """
        Return the loss of a critic given the critic's scores for fake and
        real images, the gradient penalty, and gradient penalty weight.
        To determine gradient penalty, it computes the magnitude of each
        image's gradient for current batch gradients and penalize the mean
        quadratic distance of each magnitude to 1.

        Parameters
        ----------
        real :
            current batch of real images
        fake :
            current batch of fake images
        disc_real_pred :
            the critic's scores of the real images
        disc_fake_pred :
            the critic's scores of the fake images
        c_lambda : int
            the gradient penalty weight

        Returns
        -------
            a scalar for the critic's loss for the current batch
        """
        # vector of the uniformly random proportions of real/fake per mixed image
        epsilon = torch.rand(
            len(real), 1, 1, 1, device=settings.device, requires_grad=True
        )
        # compute the gradient of the critic's scores with respect to mixes
        # of real and fake images
        gradient = self._get_gradient(real, fake.detach(), epsilon)
        # flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)
        # calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # penalize the mean squared distance of the gradient norms from 1
        gradient_penalty = torch.mean((gradient_norm - 1) * (gradient_norm - 1))
        real_score_penalty = real_score_penalty_weight * disc_real_pred ** 2
        # compute the loss of a critic given the critic's scores for fake and real images,
        # the gradient penalty, and gradient penalty weight
        critic_loss = torch.mean(
            disc_fake_pred
            - disc_real_pred
            + c_lambda * gradient_penalty
            + real_score_penalty
        )
        losses = [
            critic_loss.item(),
            torch.mean(c_lambda * gradient_penalty).item(),
            torch.mean(disc_fake_pred).item(),
            -torch.mean(disc_real_pred).item(),
            torch.mean(real_score_penalty).item(),
        ]
        losses_names = [
            "Total discriminator loss",
            "Gradient penalty",
            "Fake prediction loss",
            "Real prediction loss",
            "Real score penalty",
        ]
        return critic_loss, losses, losses_names

    def _compute_gen_loss(self, disc_fake_pred):
        """
        Return the loss of a generator given the critic's scores of the
        generator's fake images.

        Parameters
        ----------
        disc_fake_pred :
            the critic's scores of the fake images

        Returns
        -------
            a scalar for the generator loss for the current batch
        """
        gen_loss = -torch.mean(disc_fake_pred)
        return gen_loss
