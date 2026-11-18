# Third-party code and release checks

Retain the licenses in `external/licenses/` when redistributing this project.

- BAT: dataset transforms, sampling, crop utilities and legacy checkpoint statistics.
  MIT license, Copyright 2024 SparkTempest.
- CSTNet: the existing DMATrack implementation derives from CSTNet.
  MIT license, Copyright 2024 LiYunfengLYF.
- PreciseRoIPooling: native CUDA operator and Python autograd interface.
  MIT license, Copyright 2017 Megvii Technology Limited.
- CKD/OSTrack: model layers and backbone helpers were consolidated from the
  local CKD code used by the verified inference. No root LICENSE was found in
  that local source snapshot. Its upstream license and redistribution permission
  MUST be verified before public release. Do not assume another project's MIT
  license applies to these files.
- Some ViT and utility files retain their original copyright/attribution text,
  including Ross Wightman and Facebook. Do not remove legal notices.

The project owner's license has not been selected. No new root LICENSE is
invented here. This source tree is a release candidate, not a declaration that
all redistribution permissions have already been cleared. Dataset and model
weights have separate distribution conditions.
