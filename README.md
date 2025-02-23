# Data-Augmentation-Approaches-for-Satellite-Imagery
Code and data for Data Augmentation Approaches for Satellite Imagery [Hopkins et al., AAAI 2025]. The code in this repository is for implementing the satellite-specific image augmentation strategies (**Sat-CutMix**, **Sat-SlideMix**, and **Sat-Trivial**) presented in the paper. The code can also be used to reproduce the results within the paper.   


# Sat-CutMix & Sat-SlideMix
Both Sat-CutMix and Sat-SlideMix are inspired by [CutMix](https://arxiv.org/abs/1905.04899). Sat-CutMix is a mixing method in which, for every image in the batch, the batch image is mixed with another image within the batch to produce a mixed image and label. Sat-SlideMix, on the other hand, rolls every image in the batch along its height or width axis and maintains the same label. 

Implementing either method is straightforward and simply requires a call to the method prior to running data through the model, as shown below. See Sat-CutMix.ipynb or Sat-SlideMix.ipynb for working examples.

```
from from src.mixing import sat_cutMix, sat_slideMix

# Define either Sat-CutMix or Sat-SlideMix 
mixer = sat_cutMix(num_classes, satcutmix_alpha, sat_num_pairs, regression)  # set regression to True for regresssion and False for classification
# mixer = sat_slideMix(num_classes, satslidemix_beta, sat_num_pairs, regression)

for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = mixer(inputs, labels) 

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

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
