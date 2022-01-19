Adversarial modelling of electronic music loops
=================================================

In this project, we will adapt the GANSynth model approach to electronic music modelling, and will evaluate the effect of generating instantaneous frequencies instead
of relying on phase estimation algorithms.


https://user-images.githubusercontent.com/36830545/150036378-7b6b3e63-80e5-4297-b1e9-b57ef23ff96b.mp4

## Install :

- Clone the github repository :
```bash
$ git clone https://github.com/nabsara/electro-modelling.git
```
- Create a virtual environment with Python 3.8
- Activate the environment and install the dependencies with :
```bash
(myenv)$ pip install -r requirements.txt
```

TODO: add .env configuration


## Project Structure :

```bash 
electro-modelling
├── data    # directory to store the data in local /!\ DO NOT COMMIT /!\
├── docs    # to build Sphinx documentation based on modules docstrings
├── models  # directory to store the models checkpoints in local /!\ DO NOT COMMIT /!\
├── notebooks   # jupyter notebooks for data exploration and models analysis
├── README.md
├── requirements.txt   # python project dependencies with versions
├── scripts   # scripts to executes pipelines (data preparation, training, evaluation) 
├── setup.py
├── src
│   └── electro_modelling  # main package
│       ├── config.py      # global settings based on environment variables
│       ├── helpers        # global utility functions
│       │   └── __init__.py
│       ├── __init__.py
│       ├── models         # models architecture defined as class objects
│       │   └── __init__.py
│       └── pipelines      # pipelines for data preparation, training and evaluation for a given model
│           └── __init__.py
└── tests                        # tests package with unit tests
    ├── conftest.py
    └── __init__.py

```


## MNIST Results:

We implemented the Original GAN, the LeastSquareGAN, the HingeGAN and the WGAN-GP relying on the DCGAN architecture.

Train Parameters:
- z_dim = 10
- batch_size = 128
- learning_rate = 0.0002
- k_disc_steps = 1 (also 5)
- n_epochs = 20

**Simple DCGAN :** 

![results_loss__simple_dcgan_MNIST__z_10__lr_0 0002__k_1__e_20](https://user-images.githubusercontent.com/36830545/150169565-65fa9bd0-f12d-42a2-937f-01c80ae920e8.gif)

**LeastSquareGAN :**

![results_loss__least_square_dcgan_MNIST__z_10__lr_0 0002__k_1__e_20](https://user-images.githubusercontent.com/36830545/150170195-8c88b2dd-efe5-4333-9d0f-e9b9afde3706.gif)

**HingeGAN :**

![results_loss__hinge_dcgan_MNIST__z_10__lr_0 0002__k_1__e_20](https://user-images.githubusercontent.com/36830545/150170306-2dbae546-b495-401d-b0f6-e62c5527a798.gif)



**WGAN-GP :**

![results_loss__wgan_MNIST__z_10__lr_0 0002__k_1__e_20](https://user-images.githubusercontent.com/36830545/150170290-58a3fd7f-8643-4535-832d-2f5eae9b7f35.gif)



