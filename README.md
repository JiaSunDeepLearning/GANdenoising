# GANdenoising
Use generative adversarial networks for image denoising

Dependence:

1. TensorFlow 1.4
2. TensorLayer 1.7.4

Prepare data:

1. DIV2K HR dataset
2. check and modify paths in config.py before training
3. download pre-trained vgg-19 model here (https://github.com/tensorlayer/pretrained-models/tree/master/models)

Dictionary structure:

1. GANdenoising/main.py
2. GANdenoising/config.py
3. GANdenoising/model.py
4. GANdenoising/utils.py
5. GANdenoising/data2019/DIV2K_train_HR/*.png
6. GANdenoising/checkpoint/*.npz



Train:

python main.py --mode=train

or just run main.py in pycharm by changing default settings to "train"

Test:

python main.py --mode=evaluate

or just run main.py in pycharm by changing default settings to "evaluate"


