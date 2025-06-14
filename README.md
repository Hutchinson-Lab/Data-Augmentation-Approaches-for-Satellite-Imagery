# Data-Augmentation-Approaches-for-Satellite-Imagery
Code for [Data Augmentation Approaches for Satellite Imagery](http://Hutchinson-Lab.github.io/files/Hopkins_AAAI_2025.pdf) | [Supplement](http://Hutchinson-Lab.github.io/files/Hopkins_AAAI_2025_Supplement.pdf). This code is for implementing the satellite-specific image augmentation strategies of **Sat-CutMix**, **Sat-SlideMix**, and **Sat-Trivial**. 

## Abstract
Deep learning models commonly benefit from data augmentation techniques to diversify the set of training images. When working with satellite imagery, it is common for practitioners to apply a limited set of transformations developed for natural images (e.g., flip and rotate) to expand the training set without overly modifying the satellite images. There are many techniques for natural image data augmentation, but given the differences between the two domains, it is not clear whether data augmentation methods developed for natural images are well suited for satellite imagery. This paper presents an extensive experimental study on three classification and three regression tasks over four satellite image datasets. We compare common computer vision data augmentation techniques and propose three novel satellite-specific data augmentation strategies. Across tasks and datasets, we find that geometric transformations are beneficial for satellite imagery while color transformations generally are not. Additionally, our novel **Sat-SlideMix**, **Sat-CutMix**, and **Sat-Trivial** methods all exhibit strong performance across all tasks and datasets.


# Sat-CutMix & Sat-SlideMix
Sat-CutMix is an extension to the mixing method [CutMix](https://arxiv.org/abs/1905.04899). Sat-CutMix extends CutMix by 1) modifying the method to work in the regression setting and 2) for every image in the batch, the batch image is mixed with a variable number (gamma) of images. 

Sat-SlideMix is a mixing method which maintains the original image label (unlike Sat-CutMix). Rather than mixing an image with other images in the batch (i.e. Sat-CutMix), Sat-SlideMix rolls each image in the batch along its height or width axis a variable number (gamma) of times. 

![Model](http://Hutchinson-Lab.github.io/files/Sat-CutMix_SlideMix.png)

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
Sat-Trivial is an extension to [TrivialAugment](https://arxiv.org/abs/2103.10158) with satellite-specific augmentations. For each image in a batch, Sat-Trivial randomly samples one augmentation and an augmentation magnitude (if applicable). The set of possible augmentations are {identity, flip, rotate, horizontal flip, vertical flip, translate, shear, randomErase, randomSaturate, and Gaussian noise}. Below is a sample batch of images in which the randomly sampled transformations are shear, random satruate, random erase, Gaussian noise, and flip.    

![Model](http://Hutchinson-Lab.github.io/files/Sat-Trivial.png)

Implementing Sat-Trivial is as simple as replacing any transformation scheme with a call to the sat_trivial() function, as shown below. See [Sat-Trivial.ipynb](SatTrivial.ipynb) for a full working example.

```
from src.augmentation import sat_trivial

img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

# define augmentation parameters
img_size = img.shape[1]  # assumes square images
bands = img.shape[0]
means = torch.mean(img/255, dim=(1,2)).numpy()  # means need to be in the range 0-1

# perform transformation 
transform = sat_trivial(img_size, bands, means)
img = transform(img/255)
```

## Instructions
Use the config files to specify run parameters and paths.py to specify the associated data directories. Example config files can be found in the config directory. The supplement lists the configurations that were used to achieve the results in the paper. Test images are provided for euroSAT. 

To train a model use: ```python run.py --c <path_to_config_file>```

## Citation

```
@inproceedings{hopkins2025data,
  title={Data Augmentation Approaches for Satellite Imagery},
  author={Hopkins, L.M. and Wong, W. and Kerner, H. and Li, F. and Hutchinson, R.A.},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, 
  number={27},
  pages={28097-28105},
  year={2025},
  DOI={https://doi.org/10.1609/aaai.v39i27.35028}
}
````
