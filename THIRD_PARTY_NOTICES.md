# Third-party sources and acknowledgments

Project contributions use Apache-2.0. That license does not replace the original
licenses or copyright notices of upstream code, libraries, or camera data.

## RT Corporation: CRANE-X7 samples and controller library

- Product and sample-code entry point: https://rt-net.jp/products/crane-x7/
- Source: https://github.com/rt-net/rt_manipulators_cpp
- Library: https://github.com/rt-net/rt_manipulators_cpp/tree/main/rt_manipulators_lib
- Installer revision: `faa425914005b557feb57dce84d42766c7e33d6b`.
- License: [Apache-2.0, original text](licenses/rt_manipulators_cpp-LICENSE.txt).
- Upstream attribution: Copyright 2021 RT Corporation `<support@rt-net.jp>`;
  the samples03 IK files carry Copyright 2022 RT Corporation.

The project partly reuses and partly modifies the samples03 inverse-kinematics
implementation. The affected runtime paths and modification descriptions are in
[NOTICE](NOTICE) and [cpp/NOTICE](cpp/NOTICE). The local original import had no
Git metadata, so its historical import commit cannot be established. The pinned
revision above is the verified build source, not a claim about the import date.
Its library source matches the local library apart from the three feedback files
captured in `third_party/patches/recording-feedback.patch`.

The installer retains the full upstream source archive, license and per-file
notices. The recording-feedback patch adds a separate feedback cache with
sequence numbers and monotonic timestamps; it modifies `hardware.hpp` and
`hardware.cpp` and adds `recording_feedback.hpp`. The runtime builds only the
library sources, not the upstream samples or tests.

## ROBOTIS DynamixelSDK

- Source: https://github.com/ROBOTIS-GIT/DynamixelSDK
- Installer revision: `54cabbf8e6c38fac205bc4dd14e747dd0e32e56f`.
- License: [Apache-2.0, original text](licenses/DynamixelSDK-LICENSE.txt).

CMake compiles the Linux C++ serial implementation and communication support from
the downloaded archive. Original copyright headers and licensing files are
retained. No local SDK source patch is applied.

## OpenTeleVision

- Source: https://github.com/OpenTeleVision/TeleVision
- License reviewed at revision `e6e25afdb16c1b326b5bf37bd0ae79919bf79f26`:
  [original license and acknowledgments](licenses/OpenTeleVision-LICENSE.txt).
- Copyright [2024] [Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang and Xiaolong Wang].

The Python implementation modifies OpenTeleVision code and its teleoperation
workflow. The historical import revision is not recorded. The reviewed revision
identifies the source of the preserved licensing text, not the exact import.
[NOTICE](NOTICE) describes this project's changes.

OpenTeleVision's original license acknowledges the following projects and their
separate terms. These acknowledgments are retained verbatim in its license copy:

| Upstream acknowledgment | Original source/license location | Relationship to this release |
| --- | --- | --- |
| ACT | https://github.com/tonyzhaozh/act | Upstream training component; not copied into this runtime release |
| DETR | https://github.com/facebookresearch/detr | Upstream training dependency; not copied into this runtime release |
| dex-retargeting | https://github.com/dexsuite/dex-retargeting | Upstream hand-retargeting component; not a dependency of this runtime |
| Vuer | https://github.com/vuer-ai/vuer | Installed Python/frontend runtime dependency; retains its own package license |

Upstream citation:

```bibtex
@article{cheng2024tv,
  title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
  author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2407.01512},
  year={2024}
}
```

## Installed dependencies and data

Python packages, their transitive dependencies, system packages and the libraries
bundled in their wheels retain their own licenses. `requirements.txt` installs
packages rather than copying their source into this repository. This includes
NumPy, OpenCV, Vuer, params-proto, aiohttp, aiohttp-cors, aiortc, PyAV and pybind11;
PyAV wheels may bundle FFmpeg libraries with additional notices. Package license
files remain in the installed environment (including `.dist-info/licenses` where
provided). System packages retain notices under `/usr/share/doc`.

The installer does not remove or rewrite upstream license files. If distributing
an environment, container or binary bundle, include the corresponding installed
license files and bundled-library notices as well as this repository's notices.

The ZED factory calibration under `python/calibration/` is serial-specific
Stereolabs calibration data, not project-authored source code. Use the factory
calibration for the actual camera; the project's Apache-2.0 grant applies to
project code, not to ownership of Stereolabs data or product names.
