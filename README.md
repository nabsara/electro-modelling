Adversarial modelling of electronic music loops
=================================================

In this project, we will adapt the GANSynth model approach to electronic music modelling, and will evaluate the effect of generating instantaneous frequencies instead
of relying on phase estimation algorithms.



https://user-images.githubusercontent.com/36830545/150009712-0b4319ca-bfcb-4c2b-a897-2f7577bc7d80.mp4



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
