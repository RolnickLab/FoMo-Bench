"""
Implements the Generalized R-CNN framework
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
import transformers  # used for DinoV2
from torchvision.models.detection.transform import ImageList

from torchvision.utils import _log_api_usage_once


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(
        self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module, embedding_shapes: dict
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.embedding_shapes = embedding_shapes
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, spectral_keys=None, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
            spectral_keys (list[int]): special keys to fine tune FoMo Bench foundation model

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        if not isinstance(self.backbone, transformers.models.dinov2.modeling_dinov2.Dinov2Model):
            images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        if spectral_keys:
            # Custom backbone: FoMo Net foundation model
            # Requires 'spectral_keys'
            features = self.backbone((images.tensors, spectral_keys))

        elif isinstance(self.backbone, transformers.models.dinov2.modeling_dinov2.Dinov2Model):
            # This is specific to DinoV2
            # Preprocessing is made outside the class
            images = torch.stack(images)
            img_sizes = list(images.size())
            images = ImageList(images, [img_sizes[2:]] * img_sizes[0])
            features = self.backbone(images.tensors, output_hidden_states=True)

            # Method to create squared patches and then resample to iamge size
            # features = features['last_hidden_state'][:, :-1, :]  # from 257 to 256
            features = features["last_hidden_state"]
            # Create patches
            features = features.view(self.embedding_shapes["batch_size"], 16, 16, -1).permute(0, 3, 1, 2)
            features = nn.functional.interpolate(
                features, (self.embedding_shapes["output_size"], self.embedding_shapes["output_size"])
            )

            # Potential extension => re-order the squared features to map the orginal image
            # Ie try to create the inverse transformation from patchinzation of ViT

            """
            # Method to interpolate features to square => too much interpolation
            features = torch.stack(features['hidden_states'], dim=1)
            # for squared features:
            features = nn.functional.interpolate(features, (features.shape[3], features.shape[3]))
            # resize to input image sizes
            features = nn.functional.interpolate(features, (self.embedding_shapes['output_size'],
                                                            self.embedding_shapes['output_size']))
            """
        else:
            features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
