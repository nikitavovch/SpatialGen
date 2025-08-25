# SpatialGen

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="assets/logo.png" width="60%" alt="SpatialGen" />
</div>
<hr style="margin-top: 0; margin-bottom: 8px;">
<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://github.com/manycore-research/SpatialGen" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-SpatialGen-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/manycore-research/SpatialGen-1.0" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialGen-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<div align="center">

| Image-to-Scene Results                   | Text-to-Scene Results                      |
| :--------------------------------------: | :----------------------------------------: |
| ![Img2Scene](./assets/vis_img2scene.png) | ![Text2Scene](./assets/vis_text2scene.png) |

<p>SpatialGen produces multi-view, multi-modal information from a semantic layout using a multi-view, multi-modal diffusion model.</p>
</div>

## ✨ News

- [Aug, 2025] Initial release of SpatialGen-1.0!

## 📋 Release Plan

- [x] Provide inference code of SpatialGen.
- [ ] Provide training instruction for SpatialGen.
- [ ] Release SpatialGen dataset.

## SpatialGen Models

<div align="center">

| **Model**                | **Download**                                                                        |
| :----------------------: | ----------------------------------------------------------------------------------- |
| SpatialGen-1.0           | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialGen-1.0)           |
| FLUX.1-Layout-ControlNet | [🤗 HuggingFace](https://huggingface.co/manycore-research/FLUX.1-Layout-ControlNet) |

</div>

## Usage

### 🔧 Installation

Tested with the following environment:
* Python 3.10
* PyTorch 2.3.1
* CUDA Version 12.1

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialGen.git
cd SpatialGen

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Optional: fix the [flux inference bug](https://github.com/vllm-project/vllm/issues/4392)
pip install nvidia-cublas-cu12==12.4.5.8
```

### 📊 Dataset

We provide [SpatialGen-Testset](https://huggingface.co/datasets/manycore-research/SpatialGen-Testset) with 48 rooms, which labeled with 3D layout and 4.8K rendered images (48 x 100 views, including RGB, normal, depth maps and semantic maps) for MVD inference.

### Inference

```bash
# Single image-to-3D Scene
bash scripts/infer_spatialgen_i2s.sh

# Text-to-image-to-3D Scene
bash scripts/infer_spatialgen_t2s.sh
```

## License

[SpatialGen-1.0](https://huggingface.co/manycore-research/SpatialGen-1.0) is derived from [Stable-Diffusion-v2.1](https://github.com/Stability-AI/stablediffusion), which is licensed under the [CreativeML Open RAIL++-M License](https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE-MODEL). [FLUX.1-Layout-ControlNet](https://huggingface.co/manycore-research/FLUX.1-Layout-ControlNet) is licensed under the [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

## Acknowledgements

We would like to thank the following projects that made this work possible:

[DiffSplat](https://github.com/chenguolin/DiffSplat) | [SD 2.1](https://github.com/Stability-AI/stablediffusion) | [TAESD](https://github.com/madebyollin/taesd) | [FLUX](https://github.com/black-forest-labs/flux/) | [SpatialLM](https://github.com/manycore-research/SpatialLM)
