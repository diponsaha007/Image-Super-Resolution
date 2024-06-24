# Image Super Resolution Model

This repository contains a deep learning model for image super-resolution tasks using FSRCNN (Fast Super-Resolution Convolutional Neural Network) architecture. On top of that we have added:

- Channel Attention Module: Enhances network representation by focusing on informative features, dynamically adjusting the weighting of each channel.
- Residual Block: Residual Blocks are skip-connection blocks that learn residual functions with reference to the layer inputs, instead of learning unreferenced functions.

It is a very small model that is trained on the Div2K dataset. It is implemented in PyTorch and can be used to upscale images by a factor of 2, 3, or 4. The model weights and the logs can the found in the `src/outputs` folder. 

Here is an example of the model in action:

- Original Image:

![sample_x2/0006x8.png](sample_x2/0006x8.png)

- 2x Upscaled Image:

![sample_x2/0006x8_x2.png](sample_x2/0006x8_x2.png)

- 4x Upscaled Image:

![sample_x4/0006x8_x4.png](sample_x4/0006x8_x4.png)


More examples can be found in the `sample_x2` and `sample_x4` directories. Each image is named according to the scaling factor used. For example, `0006x8.png` is the original image, `0006x8_bicubic_x4.png` is the 4x bicubic upscaled image and `0006x8_x4.png` is the 4x upscaled image by the model.

Reference papers:
- [Image Super-Resolution Using Deep Convolutional Networks
](https://arxiv.org/pdf/1501.00092)
- [Accelerating the Super-Resolution Convolutional Neural Network
](https://arxiv.org/pdf/1608.00367)
- [Image Super-Resolution Using Very Deep Residual Channel Attention Networks
](https://arxiv.org/pdf/1807.02758)