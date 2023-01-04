# OcCaMix
Object-centric Contour-aware Data Augmentation Using Superpixels of Varying Granularity (OcCaMix)
-------------------------------------------------------------------------------------------------

![image](https://github.com/DanielaPlusPlus/OcCaMix/blob/main/framework.png)

In the paper, we propose a novel object-centric contour-aware CutMix data augmentation strategy with arbitrary- shape and size superpixel supports, which is hereafter referred to as OcCaMix for short. It not only captures the most discriminative regions, but also effectively preserves the contour details of the objects. Moreover, it enables the search of natural object parts of different sizes. Extensive experiments on a large number of benchmark datasets show that OcCaMix significantly outperforms state-of-the-art CutMix based data augmentation methods in classification tasks. 

The source pytorch codes and some trained models are available here.

The Top.1 accuracy with OcCaMix for classification:

<table align="left">
  <tr><th align="center">Dataset</th><th align="center">ResNet18</th><th align="center">ResNet50</th><th align="center">ResNeXt50</th></tr>
  <tr><th align="center">CIFAR100</th><th align="center">[81.42%](https://github.com/DanielaPlusPlus/OcCaMix/blob/main/CIFAR100_imagesize32_R18_OcCaMix.pt)</th><th align="center">83.69%(https://github.com/DanielaPlusPlus/OcCaMix/blob/main/CIFAR100_imagesize32_R50_OcCaMix.pt)</th><th align="center">84.01%(https://github.com/DanielaPlusPlus/OcCaMix/blob/main/CIFAR100_imagesize32_RX50_OcCaMix.pt)</th></tr>
  <tr><th align="center">CUB200-2011</th><th align="center">78.40%</th><th align="center">82.94%</th><th align="center">83.69%</th></tr>
</table>
