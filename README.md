# Data-Augmentation-Approaches-for-Satellite-Imagery
Code and data for Data Augmentation Approaches for Satellite Imagery [Hopkins et al., AAAI 2025]


--
# Sat-Trivial
```
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
from src.augmentation import sat_trivial

# import image
img = read_image('/content/drive/MyDrive/AAAI25/test_img.png')

# display original image
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

# define augmention type
augment_type = 'Sat-Trivial'

# define augmentation parameters
img_size = img.shape[1]  # assumes
bands = img.shape[0]
means = torch.mean(img/255, dim=(1,2)).numpy()  # means need to be in range 0-1

<span style="background-color: #FFFF00">HERE</span>
# perform transformation 
transform = sat_trivial(img_size, bands, means)
img_transformed = transform(img)

# display transformed image
img_transformed = img_transformed + torch.from_numpy(means).unsqueeze(1).unsqueeze(1)
plt.imshow(img_transformed.permute(1, 2, 0))
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
```
