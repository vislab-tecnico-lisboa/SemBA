##########################################################################################################
# Multi-Scale Foveal Mechanism                                                                           #
# Joao Luzio, Institute for Systems and Robotics                                                         #
# Instituto Superior Técnico, Lisbon, 2026                                                               #
##########################################################################################################

import cv2 as cv


class MS_Foveation(object):
    """Build concentric crops with decreasing spatial detail away from fixation.

    Every returned layer has base_dim pixels per side. Layer index zero covers
    the smallest scene region; each subsequent region grows by scale_factor.
    """

    def __init__(self, levels=4, dim=160, scale_factor=2):
        """Store the crop count, output side length, and inter-layer scale."""

        self.levels = levels
        self.base_dim = dim
        self.scale = scale_factor
        self.layers = []

    def foveate(self, img, fixation):
        """Return image crops centered at fixation, given in scene (x, y) pixels."""

        self.layers = []
        x_ctr, y_ctr = fixation

        # Pad the scene so crops near image edges retain their full dimensions.
        pad_size = self.base_dim * (self.scale ** (self.levels - 1))
        img = cv.copyMakeBorder(
            img.copy(),
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Translate scene coordinates into the padded image before cropping.
        for l in range(self.levels):

            y_start = int(y_ctr - (self.base_dim * (self.scale**l)) / 2) + pad_size
            y_end = int(y_ctr + (self.base_dim * (self.scale**l)) / 2) + pad_size
            x_start = int(x_ctr - (self.base_dim * (self.scale**l)) / 2) + pad_size
            x_end = int(x_ctr + (self.base_dim * (self.scale**l)) / 2) + pad_size
            self.layers.append(img[y_start:y_end, x_start:x_end])

        # Downsample larger crops; the innermost layer already has the base size.
        for l in range(1, self.levels):

            self.layers[l] = cv.resize(
                self.layers[l],
                (self.base_dim, self.base_dim),
                interpolation=cv.INTER_LINEAR,
            )

        return self.layers

    def bbox_remapping(self, bbox, lvl, center):
        """Map an xyxy layer box to scene pixels using a one-based layer number.

        The input bbox is scaled in place; the returned list also includes the
        crop origin offset. The caller must copy bbox to preserve its values.
        """

        # Undo resizing, then shift alternating x/y entries by the crop origin.
        for j in range(len(bbox)):
            bbox[j] = bbox[j] * (self.scale ** (lvl - 1))
        box = [
            center[j % 2] - (self.scale ** (lvl - 1)) * self.base_dim / 2 + bbox[j]
            for j in range(len(bbox))
        ]

        return box

    def get_fov_topology(self):
        """Reconstruct a view of the last fixation at the largest crop size."""

        assert (
            self.layers != []
        ), "can't build toplogy without first running self.foveate(...)"

        topology = cv.resize(
            self.layers[self.levels - 1],
            (
                (self.scale ** (self.levels - 1)) * self.base_dim,
                (self.scale ** (self.levels - 1)) * self.base_dim,
            ),
        )
        ctr = int((self.scale ** (self.levels - 1)) * self.base_dim / 2)

        # Paint from coarse to fine so central layers overwrite peripheral detail.
        for l in range(self.levels - 1, -1, -1):

            start = ctr - int((self.scale**l) * self.base_dim / 2)

            for y in range(self.base_dim):

                for x in range(self.base_dim):

                    x_min = start + x * (self.scale**l)
                    y_min = start + y * (self.scale**l)

                    # Expand each layer pixel to its footprint in scene pixels.
                    for i in range(self.scale**l):

                        for j in range(self.scale**l):

                            topology[y_min + i, x_min + j] = self.layers[l][y, x]

        return topology
