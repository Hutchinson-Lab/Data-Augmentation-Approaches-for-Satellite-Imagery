# Based on PyTorch's implementation of CutMix:
# https://pytorch.org/vision/main/_modules/torchvision/transforms/v2/_augment.html#CutMix

import math
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2._utils import _parse_labels_getter, has_any, is_pure_tensor, query_size
from torchvision.transforms.v2._transform import Transform, _RandomApplyTransform


class _BaseMixUpCutMix(Transform):
    def __init__(self, *, labels_getter="default") -> None:
        #def __init__(self, *, alpha: float = 1.0, num_classes: int, sat_num_pairs: int, regression,
                    #labels_getter="default") -> None:
        super().__init__()
        #self.alpha = float(alpha)
        #self.num_classes = num_classes
        #self.num_pairs = num_pairs
        #self.regression = regression
        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1:
            raise ValueError(
                f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
            )

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform],
                batch_size=labels.shape[0]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:

        label = one_hot(label,
                        num_classes=self.num_classes)  # one hot encoding of original label: [batch_size, num_classes]
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


class sat_cutMix(_BaseMixUpCutMix):
    def __init__(self, num_classes, alpha, sat_num_pairs, regression):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.num_pairs = sat_num_pairs
        self.regression = regression

    def _get_params(self, flat_inputs: List[Any], batch_size) -> Dict[str, Any]:

        lam = float(np.random.uniform(low=self.alpha, high=0.75))

        H, W = query_size(flat_inputs)

        box_batch = [[]]
        lam_adjusted_batch = [[]]
        for i in range(self.num_pairs):
            for j in range(batch_size):
                r_x = torch.randint(int(lam * W), size=(1,))
                r_y = torch.randint(int(lam * H), size=(1,))

                r = 0.5 * math.sqrt(1.0 - lam)
                r_w_half = int(r * W)
                r_h_half = int(r * H)

                x1 = int(torch.clamp(r_x - r_w_half, min=0))
                y1 = int(torch.clamp(r_y - r_h_half, min=0))
                x2 = int(torch.clamp(r_x + r_w_half, max=W))
                y2 = int(torch.clamp(r_y + r_h_half, max=H))

                if j == 0:
                    box = [(x1, y1, x2, y2)]
                    lam_adjusted = [float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))]
                else:
                    box_indv = (x1, y1, x2, y2)
                    box.append(box_indv)
                    lam_adjusted_indv = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
                    lam_adjusted.append(lam_adjusted_indv)

            if i == 0:
                box_batch[0] = box
                lam_adjusted_batch[0] = lam_adjusted
            else:
                box_batch.append(box)
                lam_adjusted_batch.append(lam_adjusted)
        return dict(box=box_batch, lam_adjusted=lam_adjusted_batch)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # inpt: tensor indicating labels of the two classes to mix; [1, 5]
        # lam_adjusted: weight for label of main image; 0.87
        # returned: [0, 0.87, 0, 0, 0, 0.13]

        if inpt is params["labels"]:
            label_mixed = torch.tensor([]).cuda()
            for i in range(self.num_pairs):
                lam = torch.tensor(params["lam_adjusted"][i]).cuda()
                if self.regression:
                    single = inpt * lam + inpt.roll(i+1) * (1.0 - lam)
                    label_mixed = torch.cat((label_mixed, single), 0)
                else:
                    label = one_hot(inpt, num_classes=self.num_classes)  
                    inpt_rolled = inpt.roll(i+1,0)
                    label_rolled = one_hot(inpt_rolled, num_classes=self.num_classes)
                    if not label.dtype.is_floating_point:
                        label = label.float()
                        label_rolled = label_rolled.float()
                    single = torch.stack([y * a + z * (1.0 - a) for y, z, a in zip(label, label_rolled, lam)])
                    label_mixed = torch.cat((label_mixed, single), 0)
            return label_mixed  # torch tensor w/ shape[num_pairs * batch]

        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            for i in range(self.num_pairs):
                box = params["box"][i]
                rolled = inpt.roll(i + 1, 0)
                for j in range(len(box)):  # batch size
                    rolled_indv = rolled[j]
                    x1, y1, x2, y2 = box[j]
                    mini = inpt[j].clone()
                    mini[..., y1:y2, x1:x2] = rolled_indv[..., y1:y2, x1:x2]
                    mini = mini.unsqueeze(0)
                    if j == 0 and i == 0:
                        output = mini
                    else:
                        output = torch.cat((output, mini), 0)

                    if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                        output = tv_tensors.wrap(output, like=inpt)
            from torchvision.utils import save_image
            i = 0
            for img in output:
                save_image(img, 'output_' + str(i) +'.png')
                i += 1
            return output
        else:
            return inpt


class sat_slideMix(_BaseMixUpCutMix):
    def __init__(self, num_classes, beta, sat_num_pairs, regression):
        super().__init__()
        self.num_classes = num_classes
        self.beta = float(beta)
        self.num_pairs = sat_num_pairs
        self.regression = regression

    def _get_params(self, flat_inputs: List[Any], batch_size):
        H, W = query_size(flat_inputs)
        slide_batch = [[]]
        for i in range(self.num_pairs):
            singe_slide = []
            lam = np.random.uniform(low=0, high=self.beta, size=batch_size)
            for l in lam:
                if torch.randint(2, (1,)):  # randomly select horizontal or vertical slide
                    singe_slide.append((int(l*W), 0))
                else:
                    singe_slide.append((0, int(l*H)))
            if i == 0:
                slide_batch[0] = singe_slide
            else:
                slide_batch.append(singe_slide)
        return dict(slide_batch=slide_batch)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if inpt is params["labels"]:
            inpt = inpt.repeat(self.num_pairs)
            return inpt

        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            for i in range(self.num_pairs):
                slide = params["slide_batch"][i]
                for j in range(len(slide)):
                    slide_x, slide_y = slide[j]
                    img = inpt[j]
                    if slide_x == 0:  # perform vertical slide
                        if torch.randint(2, (1,)):  # randomly slide in negative direction
                            slide_y *= -1
                        rolled_img = torch.roll(img, shifts=slide_y, dims=1)
                    else:  # perform horizontal slide
                        if torch.randint(2, (1,)):  # randomly slide in negative direction
                            slide_x *= -1
                        rolled_img = torch.roll(img, shifts=slide_x, dims=2)
                    if i == 0 and j == 0:
                        output = torch.unsqueeze(rolled_img, 0)

                    else:
                        output = torch.cat((output, torch.unsqueeze(rolled_img, 0)), 0)
            return output
        else:
            return inpt
