import os
import shlex
import subprocess
import imageio
import numpy as np

import gradio as gr
import spaces
import sys
from loguru import logger
current_path = os.path.dirname(os.path.abspath(__file__))



# fail to install RaDe-GS, Continue to try when has quota in Huggingface Space.
# try:
#     import diff_gaussian_rasterization  # noqa: F401
# except ImportError:
#     @spaces.GPU
#     def install_diff_gaussian_rasterization():
#         os.system("pip install ./extensions/RaDe-GS/submodules/diff-gaussian-rasterization")
#     install_diff_gaussian_rasterization()



MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(current_path, 'out')
os.makedirs(TMP_DIR, exist_ok=True)

TAG = {
    "SD15": ["gsdiff_gobj83k_sd15__render", "gsdiff_gobj83k_sd15_image__render"], # Best efficiency
    "PixArt-Sigma": ["gsdiff_gobj83k_pas_fp16__render","gsdiff_gobj83k_pas_fp16_image__render"], 
    "SD3": ["gsdiff_gobj83k_sd35m__render", "gsdiff_gobj83k_sd35m_image__render"] # Best performance
}
MODEL_TYPE = "PixArt-Sigma"

# for PixArt-Sigma
subprocess.run(shlex.split("python3 download_ckpt.py --model_type pas")) # for txt condition
subprocess.run(shlex.split("python3 download_ckpt.py --model_type pas --image_cond")) # for img condition
img_commands = "PYTHONPATH=./ bash scripts/infer.sh src/infer_gsdiff_pas.py configs/gsdiff_pas.yaml {} \
--rembg_and_center --triangle_cfg_scaling --save_ply --output_video_type mp4 --guidance_scale {} \
--image_path {} --elevation {} --prompt {}  --seed {}"

txt_commands = "PYTHONPATH=./ bash scripts/infer.sh src/infer_gsdiff_pas.py  configs/gsdiff_pas.yaml {} \
 --save_ply --output_video_type mp4 \
 --prompt {} --seed {}"


# for SD1.5
# subprocess.run(shlex.split("python3 download_ckpt.py --model_type sd15")) # for txt condition
# subprocess.run(shlex.split("python3 download_ckpt.py --model_type sd15 --image_cond")) # for img condition

# img_commands = "PYTHONPATH=./ bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml {} \
# --rembg_and_center --triangle_cfg_scaling --save_ply --output_video_type mp4 --guidance_scale {} \
# --image_path {} --elevation {} --prompt {} --seed {}"

# txt_commands = "PYTHONPATH=./ bash scripts/infer.sh src/infer_gsdiff_sd.py  configs/gsdiff_sd15.yaml {} \
#  --save_ply --output_video_type mp4 --guidance_scale {} \
#  --elevation {} --prompt {} --seed {}"



# process function
@spaces.GPU
def process(input_image, prompt='a_high_quality_3D_asset', prompt_neg='poor_quality', input_elevation=20, guidance_scale=2., input_seed=0):
    # fail to install RaDe-GS 
    # subprocess.run("cd extensions/RaDe-GS/submodules && pip3 install diff-gaussian-rasterization", shell=True)
    # subprocess.run("cd extensions/RaDe-GS/submodules/diff-gaussian-rasterization && python3 setup.py bdist_wheel ", shell=True)

    if input_image is not None:
        import uuid
        image_path = os.path.join(TMP_DIR, f"{str(uuid.uuid4())}.png")
        image_name = image_path.split('/')[-1].split('.')[0] + "_rgba"
        input_image.save(image_path)
        TAG_DEST = TAG[MODEL_TYPE][1]
        full_command = img_commands.format(TAG_DEST, guidance_scale, image_path, input_elevation, prompt, input_seed)
    else:
        TAG_DEST = TAG[MODEL_TYPE][0]
        # without guidance_scale and input_elevation
        full_command = txt_commands.format(TAG_DEST, prompt, input_seed)
        image_name = ""
    
    os.system(full_command)
    
    # save video and ply files
    ckpt_dir = os.path.join(TMP_DIR, TAG_DEST, "checkpoints")
    infer_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
    MAX_NAME_LEN = 20  # TODO: make `20` configurable
    prompt = prompt.replace("_", " ")
    prompt_name = prompt[:MAX_NAME_LEN] + "..." if prompt[:MAX_NAME_LEN] != "" else prompt
    name = f"[{image_name}]_[{prompt_name}]_{infer_from_iter:06d}"
    output_video_path = os.path.join(TMP_DIR, TAG_DEST, "inference",  name + ".mp4")
    output_ply_path = os.path.join(TMP_DIR, TAG_DEST, "inference",  name + ".ply")
    output_img_path = os.path.join(TMP_DIR, TAG_DEST, "inference",  name + "_gs.png")

    logger.info(full_command, output_video_path, output_ply_path)

    output_image = imageio.imread(output_img_path)
    return output_image, output_video_path, output_ply_path


# gradio UI
_TITLE = '''DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation'''

_DESCRIPTION = '''
### If you find our work helpful, please consider citing our paper 📚 or giving the repo a star 🌟
<div>
<a style="display:inline-block; margin-left: .5em" href="https://chenguolin.github.io/projects/DiffSplat"><img src='https://img.shields.io/badge/Project-Page-brightgreen'/></a>  
<a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2501.16764"><img src='https://img.shields.io/badge/arXiv-2501.16764-b31b1b.svg?logo=arXiv'/></a>  
<a style="display:inline-block; margin-left: .5em" href="https://github.com/chenguolin/DiffSplat"><img src='https://img.shields.io/github/stars/chenguolin/DiffSplat?style=social'/></a>  
<a style="display:inline-block; margin-left: .5em" href="https://huggingface.co/chenguolin/DiffSplat"><img src='https://img.shields.io/badge/HF-Model-yellow'/></a>  
</div>

* Input can be only text, only image, or both image and text. 
* If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
* Upload an image and click "Generate" to create a 3D asset. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)

    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            # input image
            input_image = gr.Image(label="image", type='pil')
            
            # input prompt
            input_text = gr.Textbox(label="prompt",value="a_high_quality_3D_asset")
            
            # negative prompt
            input_neg_text = gr.Textbox(label="negative prompt", value="ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate")

            # guidance_scale
            guidance_scale = gr.Slider(label="guidance scale", minimum=1., maximum=7.5, step=0.5, value=2.0)
            
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=10)

            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
            # gen button
            button_gen = gr.Button("Generate")


        with gr.Column(scale=0.8):
            with gr.Tab("Video"):
                # final video results
                output_video = gr.Video(label="video")
                # ply file
                output_file = gr.File(label="3D Gaussians (ply format)")
            with gr.Tab("Splatter Images"):
                output_image = gr.Image(interactive=False, show_label=False)
                

        button_gen.click(process, inputs=[input_image, input_text, input_neg_text, input_elevation, guidance_scale, input_seed], outputs=[output_image, output_video, output_file])

    gr.Examples(
        examples=[
                f'assets/diffsplat/{image}'
                for image in os.listdir("assets/diffsplat") if image.endswith('.png')
        ],
        inputs=[input_image],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=x),
        run_on_click=True,
        cache_examples=True,
        label='Image-to-3D Examples'
    )

    gr.Examples(
        examples=[
            "a_toy_robot",
            "a_cute_panda",
            "an_ancient_leather-bound_book"
        ],
        inputs=[input_text],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=None, prompt=x),
        run_on_click=True,
        cache_examples=True,
        label='Text-to-3D Examples'
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    block.launch(share=True)