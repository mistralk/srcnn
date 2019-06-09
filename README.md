# srcnn
* SRCNN tensorflow implementation
* Hanyang University ITE4053 Deep Learning Methods and Applications assignment #1

## Requirements
* Python 3.6+
* TensorFlow 1.13
* NumPy
* imageio

## Detail
* 32x32x1 input (random cropped from original images)
* 32x32x1 output
* 3x3 kernel
* number of epoch=10000
* mini batch size=128

## References
* Original paper
 * [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
* My program architecture is inspired by the tutorials and examples of Stanford cs230
 * https://github.com/cs230-stanford/cs230-code-examples
 * https://cs230-stanford.github.io/tensorflow-input-data.html