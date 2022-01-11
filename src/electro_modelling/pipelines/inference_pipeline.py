import os
import torch

from electro_modelling.models.models import models
from electro_modelling.datasets.signal_processing import SignalOperators


class InferencePipeline:

    def __init__(self):
        config = {
            "model_filepath": "/home/sarah/Projects/master_atiam/im_ml/electro-modelling/models/vf/generator__wgan_img_size_128_128__init_kernel_2_2_minibatch_std__z_256__lr_0.0001__k_5__e_20.pt",
            "model_name": "wgan",
            "nmels": 128,
            "dataset": "techno",
            "z_dim": 256,
            "img_chan": 1
        }

        self.operator = SignalOperators(nfft=1024, nmels=config.get("nmels"))
        self.model = models[config.get("model_name")](
            z_dim=config.get("z_dim"),
            dataset=config.get("dataset"),
            img_chan=config.get("img_chan"),
            operator=self.operator
        )
        self.generator = self.model.generator
        checkpoint = torch.load(config.get("model_filepath"))
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator.eval()

    def predict(self, n_samples):
        input_noise = self.model.get_noise(n_samples=n_samples)
        fakes = self.generator(input_noise).detach().cpu()
        return fakes
