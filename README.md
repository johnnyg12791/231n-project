# 231n-project

A description of all active models:

plankton_caffenet: Fine-tuning from AlexNet (bvlc_reference_caffenet) but with grayscale (1xWxH dimension) input

plankton_mnist_caffenet: Fine-tuning from MNIST. Poor performance (shallow network?)

plankton_xu_caffenet: Network as described by forum post by Xu (see http://www.kaggle.com/c/datasciencebowl/forums/t/11279/public-start-guide-of-deep-network-with-a-score-of-1-382285?limit=all) with random initialization. Okay performance?

plankton_caffenet_color: Fine-tuning from AlexNet (bvlc_reference_caffenet) with color (3xWxH dimension) input. This should help fine-tuning?