# Compatibility of OpenMixup

## MMCV compatibility

In ViT and Swin backbone, we use some Transformer modules (e.g., `PatchEmbed`) of MMCV, and this module is added after MMCV 1.4.2.
Therefore, we need to update the mmcv version to 1.4.2.
