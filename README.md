# Data-Augmentation-Approaches-for-Satellite-Imagery
Code for Data Augmentation Approaches for Satellite Imagery [Hopkins et al., AAAI 2025]. This code is for implementing the satellite-specific image augmentation strategies presented in the paper: **Sat-CutMix**, **Sat-SlideMix**, and **Sat-Trivial**. The code can also be used to reproduce the results within the paper.   

## Abstract
Deep learning models commonly benefit from data augmentation techniques to diversify the set of training images. When working with satellite imagery, it is common for practitioners to apply a limited set of transformations developed for natural images (e.g., flip and rotate) to expand the training set without overly modifying the satellite images. There are many techniques for natural image data augmentation, but given the differences between the two domains, it is not clear whether data augmentation methods developed for natural images are well suited for satellite imagery. This paper presents an extensive experimental study on three classification and three regression tasks over four satellite image datasets. We compare common computer vision data augmentation techniques and propose three novel satellite-specific data augmentation strategies. Across tasks and datasets, we find that geometric transformations are beneficial for satellite imagery while color transformations generally are not. Additionally, our novel **Sat-SlideMix**, **Sat-CutMix**, and **Sat-Trivial** methods all exhibit strong performance across all tasks and datasets.


# Sat-CutMix & Sat-SlideMix
Sat-CutMix is a an extension to the mixing method [CutMix](https://arxiv.org/abs/1905.04899). The main modifications we made to CutMix are 1) to work in the regression setting and 2) for every image in the batch, the batch image is mixed with a variable number (gamma) of images. Rather than mixing an image with other images in the batch, Sat-SlideMix rolls each image in the batch along its height or width axis a variable number (gamma) of times. 

Implementing either method is straightforward and simply requires a call to either of the methods prior to running data through the model, as shown below. See [Mixers.ipynb](Mixers.ipynb) for working examples.

```
from from src.mixing import sat_cutMix, sat_slideMix

# Instantiate either Sat-CutMix or Sat-SlideMix
mixer = sat_cutMix(num_classes, satcutmix_alpha, sat_num_pairs, regression)  
# mixer = sat_slideMix(num_classes, satslidemix_beta, sat_num_pairs, regression)

for imgs, labels in dataloader:
        mixed_imgs, mixed_labels = mixer(imgs, labels) 
        outputs = net(mixed_imgs)
```

![Model](Sat-CutMix_SlideMix.png)

# Sat-Trivial
Sat-Trivial is an extension to [TrivialAugment](https://arxiv.org/abs/2103.10158) with satellite-specific augmentations. For each image in a batch, Sat-Trivial randomly samples one augmentation and an augmentation magnitude (if applicable). The set of possible augmentations are {flip, rotate, horizontal flip, vertical flip, translate, shear, randomErase, randomSaturate, and Gaussian noise}. 

Implementing Sat-Trivial is as simple as replacing any transformation scheme with a call to the sat_trivial() function, as shown below. See [Sat-Trivial.ipynb](SatTrivial.ipynb) for a full working example.

```
from src.augmentation import sat_trivial

img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

# define augmentation parameters
img_size = img.shape[1]  # assumes
bands = img.shape[0]
means = torch.mean(img/255, dim=(1,2)).numpy()  # means need to be in the range 0-1

# perform transformation 
transform = sat_trivial(img_size, bands, means)
img = transform(img)
```
