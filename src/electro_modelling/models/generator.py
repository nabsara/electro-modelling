import torch.nn as nn


class Generator(nn.Module):
    """
    Generator Class

    Parameters
    ----------
    z_dim : int
        the dimension of the noise vector
    img_chan : int
        the number of channels in the images ex : 3 for RGB (MNIST is black-and-white, so default is 1 channel)
    hidden_dim : int
        the inner dimension

    Attributes
    ----------
    z_dim : int
        the dimension of the noise vector
    model : nn.Sequential
        the DCGAN generator model

    """

    def __init__(self,dataset,z_dim, img_chan,hidden_dim,nmel_ratio=0):
        super().__init__()
        self.z_dim = z_dim
        if dataset == 'MNIST':
            hidden_dim=64
            img_chan=1
            self.model = nn.Sequential(
                self.make_gen_block(input_channels=z_dim, output_channels=256, kernel_size=3, stride=2),
                self.make_gen_block(input_channels=hidden_dim * 4, output_channels=hidden_dim * 2, kernel_size=4, stride=1),
                self.make_gen_block(input_channels=hidden_dim * 2, output_channels=hidden_dim, kernel_size=3, stride=2),
                nn.ConvTranspose2d(hidden_dim, img_chan, kernel_size=4, stride=2),
                nn.Tanh()
            )
        elif dataset == 'techno':
            
            self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim *8, (2*nmel_ratio,2), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 8, kernel_size=(3,3), stride=1),
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *8, output_channels=hidden_dim* 4, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *4, output_channels=hidden_dim* 4, kernel_size=(3,3), stride=1),            
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *4, output_channels=hidden_dim* 2, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim *2, output_channels=hidden_dim* 2, kernel_size=(3,3), stride=1),
            nn.Upsample(scale_factor=2),
            self.make_gen_block_techno(input_channels=hidden_dim *2, output_channels=hidden_dim, kernel_size=(3,3), stride=1),
            self.make_gen_block_techno(input_channels=hidden_dim, output_channels=hidden_dim, kernel_size=(3,3), stride=1),

            nn.Conv2d(hidden_dim, img_chan, kernel_size=1, stride=1),
            nn.Tanh()
            )
        else:
            print('data set name is not known')

    def make_gen_block(self, input_channels, output_channels, kernel_size, stride):
        """
        Build the sequence of operations corresponding to a generator block of DCGAN :
        - transposed convolution
        - batchnorm
        - ReLU activation

        Parameters
        ----------
        input_channels : int
            the number of channels of the input feature representation
        output_channels : int
            the number of channels of the output feature representation
        kernel_size : int
            the size of each convolutional filter (kernel_size, kernel_size)
        stride : int
            the stride of the convolution

        Returns
        -------
            nn.Sequential DCGAN generator block
        """

        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True)
        )
    def make_gen_block_techno(self, input_channels, output_channels, kernel_size, stride,padding='same'):
        """
        Build the sequence of operations corresponding to a generator block of DCGAN :
        - transposed convolution
        - batchnorm
        - ReLU activation

        Parameters
        ----------
        input_channels : int
            the number of channels of the input feature representation
        output_channels : int
            the number of channels of the output feature representation
        kernel_size : int
            the size of each convolutional filter (kernel_size, kernel_size)
        stride : int
            the stride of the convolution

        Returns
        -------
            nn.Sequential DCGAN generator block
        """

        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride,padding=padding),
            nn.BatchNorm2d(num_features=output_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
        )
    
    def forward(self, x):
        """
        Function that applies the forward pass of the generator model on a given
        noise tensor and which returns generated images.

        Parameters
        ----------
        x : Tensor
            a noise tensor with dimensions (n_samples, z_dim)

        Returns
        -------
            generated images
        """
        noise = x.view(len(x), self.z_dim, 1, 1)
        output = self.model(noise)
        # output = output.view(x.size(0), 1, 28, 28)
        return output
    
# class pixel_norm(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,x,eps=1e-8):
#         mean = torch.mean(x**2,axis = 3,keepdim=True)
#         return()