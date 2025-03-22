# Data-Augmentation-Approaches-for-Satellite-Imagery
Code for Data Augmentation Approaches for Satellite Imagery | [Paper](Hopkins_2025.pdf) | [Supplement](Supplement.pdf). This code is for implementing the satellite-specific image augmentation strategies of **Sat-CutMix**, **Sat-SlideMix**, and **Sat-Trivial**.   

## Abstract
Deep learning models commonly benefit from data augmentation techniques to diversify the set of training images. When working with satellite imagery, it is common for practitioners to apply a limited set of transformations developed for natural images (e.g., flip and rotate) to expand the training set without overly modifying the satellite images. There are many techniques for natural image data augmentation, but given the differences between the two domains, it is not clear whether data augmentation methods developed for natural images are well suited for satellite imagery. This paper presents an extensive experimental study on three classification and three regression tasks over four satellite image datasets. We compare common computer vision data augmentation techniques and propose three novel satellite-specific data augmentation strategies. Across tasks and datasets, we find that geometric transformations are beneficial for satellite imagery while color transformations generally are not. Additionally, our novel **Sat-SlideMix**, **Sat-CutMix**, and **Sat-Trivial** methods all exhibit strong performance across all tasks and datasets.


# Sat-CutMix & Sat-SlideMix
Sat-CutMix is an extension to the mixing method [CutMix](https://arxiv.org/abs/1905.04899). Sat-CutMix extends CutMix by 1) modifying the method to work in the regression setting and 2) for every image in the batch, the batch image is mixed with a variable number (gamma) of images. 

Sat-SlideMix is a mixing method which maintains the original image label (unlike Sat-CutMix). Rather than mixing an image with other images in the batch (i.e. Sat-CutMix), Sat-SlideMix rolls each image in the batch along its height or width axis a variable number (gamma) of times. 

![Model](Sat-CutMix_SlideMix.png)

Implementing either method is straightforward and simply requires a call to either of the methods prior to running data through the model, as shown below. See [Mixers.ipynb](Mixers.ipynb) for working examples.

```
from from src.mixing import sat_cutMix, sat_slideMix
from torchvision import models

model = models.resnet18()

# Instantiate either Sat-CutMix or Sat-SlideMix
mixer = sat_cutMix(num_classes, satcutmix_alpha, sat_num_pairs, regression)  
# mixer = sat_slideMix(num_classes, satslidemix_beta, sat_num_pairs, regression)

for imgs, labels in dataloader:
        mixed_imgs, mixed_labels = mixer(imgs, labels) 
        outputs = model(mixed_imgs)
```

# Sat-Trivial
Sat-Trivial is an extension to [TrivialAugment](https://arxiv.org/abs/2103.10158) with satellite-specific augmentations. For each image in a batch, Sat-Trivial randomly samples one augmentation and an augmentation magnitude (if applicable). The set of possible augmentations are {identity, flip, rotate, horizontal flip, vertical flip, translate, shear, randomErase, randomSaturate, and Gaussian noise}. Below is a sample batch of images in which the transformations selected for each image are shear, random satruate, random erase, Gaussian noise, and flip.    



Implementing Sat-Trivial is as simple as replacing any transformation scheme with a call to the sat_trivial() function, as shown below. See [Sat-Trivial.ipynb](SatTrivial.ipynb) for a full working example.

## Citation

```
@inproceedings{hopkins2025data,
  title={Data Augmentation Approaches for Satellite Imagery},
  author={Hopkins, Laurel M. and Wong, Weng-Keen and Kerner, Hannah and Li, Fuxin and Hutchinson, Rebecca A},
  book title={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
````
