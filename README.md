# Research-Fourier-Convolutions
This repository contains code for implementation and testing of fourier convolutions.

![Fourier Conv](https://github.com/rohit3463/Research-Fourier-Convolutions/blob/master/fourier-conv.png)


### Directory Structure
```
-Research-Fourier-Convolutions
    - Data/                           (to store data and their csv files)
    - Model/                          (to store trained model for testing)  
    - train.py                        (driver for training)
    - trainModule.py                  (actual code for training and testing)
    - logging.py                      (a file for logging into tensorboard)
    - config.json                     (a config file required by training driver file)
    - utils.py                        (functions or class required to preprocess)
    - transforms.py                   (contains classes in torch transformation format)
    - datagen_mnist.py                (datagenerator for mnist dataset)
    - test_gen.py                     (testing file for data generator)
    - mobilenet_v3.py                 (standard mobilenet_v3)
    - mobilenet_v3_FC.py              (fourier mobilenet_v3)
    - fourier_conv.py                 (contains fourier conv layer implementation)
    - requirements.txt
    - README.md
    - LICENSE
```