# A diffuser version implementation of CAT3D (https://arxiv.org/pdf/2405.10314)
# by Chuan Fang

import inspect
from dataclasses import dataclass

# from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import numpy as np
import einops
import torch
from packaging import version
from transformers import AutoImageProcessor
import torch.nn.functional as tF

from diffusers import AutoencoderKL, DiffusionPipeline, AutoencoderTiny
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging
from diffusers.configuration_utils import FrozenDict
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers_spatialgen import UNetMVMM2DConditionModel
from src.models.pose_adapter import RayMapEncoder
from src.utils.typing import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
# todo
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""
@dataclass
class SpatialGenDiffusionPipelineOutput(StableDiffusionPipelineOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    mvmm_latents: torch.Tensor    # multi-view multi-modal latents (batch_size, num_channels_latents, height // 8, width // 8)
    input_depths_confi_maps: torch.Tensor    # confidence maps (batch_size, height, width, 1)
    target_depths_confi_maps: torch.Tensor    # confidence maps (batch_size, height, width, 1)

class SpatialGenDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for single view conditioned novel view generation using Zero1to3.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. Stable Diffusion Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder.
        unet ([`UNetMVMM2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.

    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        ray_encoder: RayMapEncoder,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNetMVMM2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: AutoImageProcessor,
        requires_safety_checker: bool = True,
        depth_vae: AutoencoderKL = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # if safety_checker is None and requires_safety_checker:
        #     logger.warning(
        #         f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
        #         " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
        #         " results in services or applications open to the public. Both the diffusers team and Hugging Face"
        #         " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
        #         " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
        #         " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
        #     )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            depth_vae=depth_vae,
            ray_encoder=ray_encoder,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def save_pretrained(self, save_directory, safe_serialization = True, variant = None, max_shard_size = None, push_to_hub = False, **kwargs):
        super().save_pretrained(save_directory, safe_serialization, variant, max_shard_size, push_to_hub, **kwargs)
        # save additional models
        if self.depth_vae is not None:
            self.depth_vae.save_pretrained(f"{save_directory}/depth_vae")
        if self.ray_encoder is not None:
            self.ray_encoder.save_pretrained(f"{save_directory}/ray_encoder")
            

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()
        logger.info(f"enable vae slicing")

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()
        logger.info(f"disable vae slicing")

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()
        logger.info(f"enable vae tiling")

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()
        logger.info(f"disable vae tiling")

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None:
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to" f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents, use_depth_vae=False, return_confidence=False):
        vae = self.depth_vae if use_depth_vae else self.vae
        if use_depth_vae:
            if isinstance(self.depth_vae, AutoencoderKL):
                latents = latents / self.depth_vae.config.scaling_factor
            elif isinstance(self.depth_vae, AutoencoderTiny):
                latents = latents
        else:
            latents = latents / self.vae.config.scaling_factor
        image = vae.decode(latents).sample
        if image.shape[1] == 4:
            print(f"Using 4 channels VAE output, the 4th channel is for depth_confidence")
            confi_map = image[:, 3:4, :, :]
            image = image[:, :3, :, :]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        if return_confidence:
            confi_map = confi_map.cpu().permute(0, 2, 3, 1).float().numpy()
            return image, confi_map
        else:
            return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError("`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is" f" {type(image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_img_latents(self, image, batch_size, dtype, device, generator=None, do_classifier_free_guidance=False):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)]  # sample
            init_latents = [init_latent * self.vae.config.scaling_factor for init_latent in init_latents]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.mode()
            init_latents = init_latents * self.vae.config.scaling_factor

        # init_latents = self.vae.config.scaling_factor * init_latents  # todo in original zero123's inference gradio_new.py, model.encode_first_stage() is not scaled by scaling_factor
        if batch_size > init_latents.shape[0]:
            # init_latents = init_latents.repeat(batch_size // init_latents.shape[0], 1, 1, 1)
            num_images_per_prompt = batch_size // init_latents.shape[0]
            # duplicate image latents for each generation per prompt, using mps friendly method
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1)
            init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        # init_latents = torch.cat([torch.zeros_like(init_latents), init_latents]) if do_classifier_free_guidance else init_latents

        init_latents = init_latents.to(device=device, dtype=dtype)
        return init_latents

    def do_sanity_check(
        self, latent_model_input: Float[Tensor, "Bt C H W"], input_indices: Float[Tensor, "Bt"], target_indices: Float[Tensor, "Bt"], T_in: int, T_out: int, global_step: int
    ):
        """check the input data of model

        Args:
            latent_model_input (Float[Tensor, &quot;Bt C H W&quot;]): input latent featues
            noise_scheduler (DDIMScheduler): noise scheduler
            vae (AutoencoderKL): firist stage encoder
            target_noise (Float[Tensor, &quot;B N C&quot;]): _description_
            timesteps_target (Float[Tensor, &quot;B N&quot;]): _description_
            T_in (int): number of input views
            T_out (int): number of output views

        Returns:
            None
        """
        tmp = einops.rearrange(latent_model_input, "(b t) c h w -> b t c h w", t=T_in + T_out)
        batch_size = tmp.shape[0]
        # input_latents = einops.rearrange(tmp[:, T_out:, :4, :, :], 'b t c h w -> (b t) c h w', t=T_in)
        # denoised_latents = einops.rearrange(tmp[:, :T_out, :4, :, :], 'b t c h w -> (b t) c h w', t=T_out)
        input_latents = latent_model_input[input_indices, :4]
        denoised_latents = latent_model_input[target_indices, :4]

        decoded_input_images = self.decode_latents(input_latents)
        # logger.info(f'decoded_input_images: {decoded_input_images.shape}')
        decoded_input_images = einops.rearrange(decoded_input_images, "(b t) h w c -> b t h w c", b=batch_size, t=T_in)
        for batch_idx in range(decoded_input_images.shape[0]):
            for frame_idx in range(decoded_input_images.shape[1]):
                PIL.Image.fromarray((decoded_input_images[batch_idx, frame_idx] * 255).astype(np.uint8)).save(f"decoded_input_images_step49_b{batch_idx}_f{frame_idx}.png")
        # debug the original image and the noisy image
        decoded_target_images = self.decode_latents(denoised_latents)
        # logger.info(f'decoded_target_images: {decoded_target_images.shape}')
        decoded_target_images = einops.rearrange(decoded_target_images, "(b t) h w c -> b t h w c", b=batch_size, t=T_out)
        for batch_idx in range(decoded_target_images.shape[0]):
            for frame_idx in range(decoded_target_images.shape[1]):
                PIL.Image.fromarray((decoded_target_images[batch_idx, frame_idx] * 255).astype(np.uint8)).save(f"decoded_target_images_step49_b{batch_idx}_f{frame_idx}.png")

        # input_masks = tmp[:, T_out:, 4:5, :, :].float().detach().cpu().numpy()
        # target_masks = tmp[:, :T_out, 4:5, :, :].float().detach().cpu().numpy()
        # for batch_idx in range(input_masks.shape[0]):
        #     for frame_idx in range(input_masks.shape[1]):
        #         PIL.Image.fromarray((input_masks[batch_idx, frame_idx, 0] * 255).astype(np.uint8)).save(f'input_masks_step{global_step}_b{batch_idx}_f{frame_idx}.png')
        # for batch_idx in range(target_masks.shape[0]):
        #     for frame_idx in range(target_masks.shape[1]):
        #         PIL.Image.fromarray((target_masks[batch_idx, frame_idx, 0] * 255).astype(np.uint8)).save(f'target_masks_step{global_step}_b{batch_idx}_f{frame_idx}.png')

    def prepare_camera_embedding(self, task_embedding: Union[float, torch.Tensor], do_classifier_free_guidance, num_images_per_prompt=1):
        # (B, 2)
        task_embedding = task_embedding.to(dtype=self.unet.dtype, device=self.unet.device)

        # (B, 4)
        task_embedding = torch.cat([torch.sin(task_embedding), torch.cos(task_embedding)], dim=-1)
        assert self.unet.config.class_embed_type == "projection"
        assert self.unet.config.projection_class_embeddings_input_dim in [4, 6, 8]
        # else:
        #     raise NotImplementedError

        task_embedding = task_embedding.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            task_embedding = torch.cat([task_embedding, task_embedding], dim=0)

        return task_embedding

    def reshape_to_cd_input(self, input: Float[Tensor, "Bt C H W"]):
        # reshape input for cross-domain attention
        input_rgb_uc, input_other_modal_uc, input_rgb_cond, input_other_modal_cond = torch.chunk(input, dim=0, chunks=4)
        input = torch.cat([input_rgb_uc, input_rgb_cond, input_other_modal_uc, input_other_modal_cond], dim=0)
        return input

    def reshape_to_cfg_output(self, output: Float[Tensor, "Bt C H W"]):
        # reshape input for cfg
        output_rgb_uc, output_rgb_cond, output_other_modal_uc, output_other_modal_cond = torch.chunk(output, dim=0, chunks=4)
        output = torch.cat([output_rgb_uc, output_other_modal_uc, output_rgb_cond, output_other_modal_cond], dim=0)
        return output

    def rescale_cfg(self, modle_pred: Float[Tensor, "Bt C H W"], cfg_weight: float = 7.5, rescale_factor: float = 0.7):
        """
        Rescale the CFG results to prevent the generations frrom over-explosure,
        as methioned in https://arxiv.org/pdf/2305.08891
        """
        # perform normal guidance (as before)
        # TODO: only do CFG and rescaling at output indices
        pred_uncond, pred_cond = modle_pred.chunk(2)
        cfg_pred = pred_uncond + cfg_weight * (pred_cond - pred_uncond)

        # calculate standard deviations (used to rescale the results from guidance)
        # https://arxiv.org/pdf/2305.08891
        std_pos = torch.std(pred_cond, dim=[1, 2, 3], keepdim=True)  # formula 14
        std_cfg = torch.std(cfg_pred, dim=[1, 2, 3], keepdim=True)  # formula 14
        # rescale the results from guidance (fixes overexposure)
        x_rescaled = cfg_pred * (std_pos / std_cfg)  # formula 15
        # mix with the original results from guidance by factor f to avoid plain images
        x_final = rescale_factor * x_rescaled + (1 - rescale_factor) * cfg_pred  # formula 16
        return x_final

    def __encode_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)  # [1,4]
        text_embed = self.text_encoder(text_input_ids)[0].to(self.unet.dtype)  # [1,4,1024]
        return text_embed

    @torch.no_grad()
    def __call__(
        self,
        input_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        input_rgb_indices: Int[Tensor, "Bt"] = None,  # indices for rgb inputs
        condition_indices: Int[Tensor, "Bt"] = None,  # indices for all conditions: input_rgb_indices + input_other_modal_indices
        input_indices: Int[Tensor, "Bt"] = None,  # indices for all inputs: input_rgb_indices + input_other_modal_indices
        target_indices: Int[Tensor, "Bt"] = None,  # indices for all targets: target_rgb_indices + target_other_modal_indices
        output_indices: Int[Tensor, "Bt"] = None,  # indices for all outputs: target_indices + input_other_modal_indices
        input_rays: Optional[torch.FloatTensor] = None,
        target_rays: Optional[torch.FloatTensor] = None,
        task_embeddings: Optional[torch.FloatTensor] = None,
        warpped_target_rgbs: Float[Tensor, "Bt 3 H W"] = None,
        torch_dtype=torch.float32,
        height: Optional[int] = None,
        width: Optional[int] = None,
        T_in: Optional[int] = None,
        T_out: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
        num_tasks: int = 2,
        cond_input_layout_sem_images: Float[Tensor, "Bt 3 H W"] = None,
        cond_target_layout_sem_images: Float[Tensor, "Bt 3 H W"] = None,
        cond_input_layout_depth_images: Float[Tensor, "Bt 3 H W"] = None,
        cond_target_layout_depth_images: Float[Tensor, "Bt 3 H W"] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            input_imgs (`PIL` or `List[PIL]`, *optional*):
                The single input image for each 3D object
            prompt_imgs (`PIL` or `List[PIL]`, *optional*):
                Same as input_imgs, but will be used later as an image prompt condition, encoded by CLIP feature
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(input_imgs, height, width, callback_steps)

        # 2. Define call parameters
        if isinstance(input_imgs, PIL.Image.Image):
            batch_size = 1
        elif isinstance(input_imgs, list):
            batch_size = len(input_imgs)
        else:
            batch_size = input_imgs.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input text prompt
        # normally, input batch size = 1
        real_bsz = batch_size // T_in
        num_sample_views = T_in + T_out
        text_prompt = "indoor scene, high quality, 8k, realistic texture, clean walls"
        prompt_embeds = self.__encode_text(text_prompt)
        prompt_embeds = prompt_embeds.repeat(real_bsz * num_tasks * num_sample_views, 1, 1)
        neg_text_prompt = "plants, greenery, jungle, vines, leaves, potted plants, trees, garden, overgrown, dark, dim, gloomy, shadowy, cluttered, messy, crowded, low quality, blurry, grainy, distorted"
        negative_prompt_embeds = self.__encode_text(neg_text_prompt)
        negative_prompt_embeds = negative_prompt_embeds.repeat(real_bsz * num_tasks * num_sample_views, 1, 1)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        output_batch_size = len(output_indices)
        target_latents = self.prepare_latents(
            # batch_size=(batch_size // T_in) * T_out * num_images_per_prompt, # todo use t_out
            batch_size=output_batch_size * num_images_per_prompt,  # todo use t_out
            num_channels_latents=4,
            height=height,
            width=width,
            dtype=torch_dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        # logger.info(f'target_latents.shape: {target_latents.shape}')

        # 6. Prepare input conditions: image, rays, masks
        input_latents = self.prepare_img_latents(
            image=input_imgs,
            batch_size=batch_size,
            dtype=torch_dtype,
            device=device,
            generator=generator,
        )
        if warpped_target_rgbs is not None:
            warpped_target_rgb_latents = self.prepare_img_latents(
                image=warpped_target_rgbs,
                batch_size=real_bsz,
                dtype=torch_dtype,
                device=device,
                generator=generator,
            )
        
        if cond_input_layout_sem_images is not None and cond_target_layout_sem_images is not None \
        and cond_input_layout_depth_images is not None and cond_target_layout_depth_images is not None:
            input_layout_sem_latents = self.prepare_img_latents(image=cond_input_layout_sem_images,
                batch_size=real_bsz,
                dtype=torch_dtype,
                device=device,
                generator=generator,
            )
            target_layout_sem_latents = self.prepare_img_latents(image=cond_target_layout_sem_images,
                batch_size=real_bsz,
                dtype=torch_dtype,
                device=device,
                generator=generator,
            )
            input_layout_depth_latents = self.prepare_img_latents(image=cond_input_layout_depth_images,
                batch_size=real_bsz,
                dtype=torch_dtype,
                device=device,
                generator=generator,
            )
            target_layout_depth_latents = self.prepare_img_latents(image=cond_target_layout_depth_images,
                batch_size=real_bsz,
                dtype=torch_dtype,
                device=device,
                generator=generator,
            )
            layout_sem_latents = torch.cat([input_layout_sem_latents, target_layout_sem_latents], dim=0)  # B*(T_in+T_out), 4, H//8, W//8
            layout_depth_latents = torch.cat([input_layout_depth_latents, target_layout_depth_latents], dim=0)  # B*(T_in+T_out), 4, H//8, W//8
            layout_latents = torch.cat([layout_sem_latents, layout_depth_latents], dim=0)  # B*2*(T_in+T_out), 4, H//8, W//8
            condition_latents = torch.cat([input_latents, layout_latents], dim=0)  
        else:
            condition_latents = input_latents
        
        if self.ray_encoder is not None:
            rays_in_embeds = self.ray_encoder(input_rays).to(dtype=torch_dtype)  # BxT_inx16xhxw
            rays_in_embeds = einops.rearrange(rays_in_embeds, "B T C H W -> (B T) C H W")  # (B*Ni, 16, 32, 32)
            rays_out_embeds = self.ray_encoder(target_rays).to(dtype=torch_dtype)  # BxT_outx16xhxw
            rays_out_embeds = einops.rearrange(rays_out_embeds, "B T C H W -> (B T) C H W")  # (B*No, 16, 32, 32)
        else:
            input_rays = einops.rearrange(input_rays, "B T C H W -> (B T) C H W")
            rays_in_embeds = tF.interpolate(input_rays, size=(input_latents.shape[-2], input_latents.shape[-1]), mode="bilinear", align_corners=False)
            target_rays = einops.rearrange(target_rays, "B T C H W -> (B T) C H W")
            rays_out_embeds = tF.interpolate(target_rays, size=(input_latents.shape[-2], input_latents.shape[-1]), mode="bilinear", align_corners=False)
        # logger.info(f"rays_in_embeds shape {rays_in_embeds.shape} and rays_out_embeds shape {rays_out_embeds.shape}")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        unet_in_channs = self.unet.config.in_channels
        if unet_in_channs >= 21:
            ray_channel = 16
        elif unet_in_channs == 11:
            ray_channel = 6
        end_ray_chann = 4 + ray_channel
        use_binary_mask = False
        use_warpped_image = False
        if unet_in_channs == 21:
            use_binary_mask = True
            assert warpped_target_rgbs is None, "warpped_target_rgbs should be None when unet_in_channs == 21"
        if unet_in_channs == 24:
            use_binary_mask = False
            use_warpped_image = True
            assert warpped_target_rgbs is not None, "warpped_target_rgbs should be provided when unet_in_channs == 24"
        elif unet_in_channs == 25:
            use_binary_mask = True
            use_warpped_image = True
            assert warpped_target_rgbs is not None, "warpped_target_rgbs should be provided when unet_in_channs == 25"
        
        # (B*num_tasks*(T_in+T_out), 21, Hl, Wl)
        cond_inputs = torch.zeros((real_bsz * num_sample_views * num_tasks, unet_in_channs, 
                                   input_latents.shape[-2], input_latents.shape[-1]), device=input_latents.device, dtype=torch_dtype)
        # latents
        cond_inputs[condition_indices, :4, ...] = condition_latents
        cond_inputs[output_indices, :4, ...] = target_latents
        # ray mebeddings
        cond_inputs[input_indices, 4:end_ray_chann, ...] = rays_in_embeds
        cond_inputs[target_indices, 4:end_ray_chann, ...] = rays_out_embeds
        if use_binary_mask:
            mask_channel = 1
            # use masks to indicate which views are observed, and which are to be generated
            cond_inputs[condition_indices, end_ray_chann:end_ray_chann+mask_channel, ...] = 1.0
            cond_inputs[output_indices, end_ray_chann:end_ray_chann+mask_channel, ...] = 0.0
        if use_warpped_image:
            if use_binary_mask == False:
                mask_channel = 0
            input_rgb_latents_tmp = einops.rearrange(input_latents, "(B Nt) C H W -> B Nt C H W", B=real_bsz)  # B, T_in, 4, H, W
            input_rgb_latents_tmp = input_rgb_latents_tmp.repeat(1, num_tasks, 1, 1, 1)  # B, num_tasks*T_in, 4, H, W
            input_rgb_latents_tmp = einops.rearrange(input_rgb_latents_tmp, "B Nt C H W -> (B Nt) C H W")  # (B*num_tasks*T_in, 4, H, W)
            warpped_target_rgb_latents_tmp = einops.rearrange(warpped_target_rgb_latents, "(B Nt) C H W -> B Nt C H W", B=real_bsz)  # B, T_out, 4, H, W
            warpped_target_rgb_latents_tmp = warpped_target_rgb_latents_tmp.repeat(1, num_tasks, 1, 1, 1)  # B, num_taks*T_out, 4, H, W
            warpped_target_rgb_latents_tmp = einops.rearrange(warpped_target_rgb_latents_tmp, "B Nt C H W -> (B Nt) C H W") # (B*num_tasks, T_out, 4, H, W)
            cond_inputs[input_indices, end_ray_chann+mask_channel:end_ray_chann+mask_channel+4, ...] = input_rgb_latents_tmp
            cond_inputs[target_indices, end_ray_chann+mask_channel:end_ray_chann+mask_channel+4, ...] = warpped_target_rgb_latents_tmp
        # logger.info(f'cond_inputs.shape: {cond_inputs.shape}')

        task_embeddings = self.prepare_camera_embedding(task_embeddings, do_classifier_free_guidance, num_images_per_prompt=num_images_per_prompt)
        # logger.info(f"task_embeddings.shape: {task_embeddings.shape}")

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    uncond_inputs = cond_inputs.detach().clone()
                    uncond_inputs[input_rgb_indices, :4, ...] = 0  # zero out the input latents
                    if use_binary_mask:
                        uncond_inputs[:, end_ray_chann:end_ray_chann+mask_channel, ...] = 0     # zero out the warpped images, keep the rays
                    if use_warpped_image:
                        uncond_inputs[:, end_ray_chann+mask_channel:end_ray_chann+mask_channel+4, ...] = 0
                    latent_model_input = torch.cat([uncond_inputs, cond_inputs], dim=0)
                else:
                    latent_model_input = cond_inputs

                timestep_all = torch.tensor([t], device=target_latents.device).repeat_interleave(real_bsz * num_tasks * num_sample_views)
                timestep_all[condition_indices] = 0
                timestep_all = timestep_all.repeat(2) if do_classifier_free_guidance else timestep_all
                # logger.info(f"timestep_all: {timestep_all}")

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # logger.info(f"latent_model_input: {latent_model_input.shape}")

                # predict the noise residual
                # Input: (B, 21, H, W), Output: (B, 4, H, W)
                noise_pred = self.unet(
                    latent_model_input,
                    timestep_all,
                    encoder_hidden_states=prompt_embeds,  # Bxnum_tasks, 77, 1024
                    class_labels=task_embeddings,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_img = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_img - noise_pred_uncond)

                # only take target views
                target_noise_pred = noise_pred[output_indices]

                # compute the previous noisy sample x_t -> x_t-1
                cond_inputs[output_indices, :4] = self.scheduler.step(
                    model_output=target_noise_pred.to(dtype=torch.float32), timestep=t, sample=cond_inputs[output_indices, :4].to(dtype=torch.float32), **extra_step_kwargs
                ).prev_sample.to(cond_inputs.dtype)


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, cond_inputs[output_indices, :4])

        filtered_latents = cond_inputs[output_indices, :4]

        # 8. Post-processing
        # true prediction tasks are num_tasks-2 if use layout prior
        num_tasks = num_tasks if cond_input_layout_sem_images is None else num_tasks - 2
        has_nsfw_concept = None
        input_depth_confs = None
        target_depth_confs = None
        if output_type == "latent":
            image = filtered_latents
        elif output_type == "pil":
            if self.depth_vae is None:
                # 8. Post-processing
                image = self.decode_latents(filtered_latents)
            else:
                num_target_imgs = real_bsz * num_tasks * T_out
                target_view_latents = filtered_latents[:num_target_imgs]
                input_view_othermodal_latents = filtered_latents[num_target_imgs:]
                # decode rgb and depth separately
                if num_tasks == 2:
                    target_rgb_latents = target_view_latents[:T_out]
                    target_depth_latents = target_view_latents[T_out:]
                    input_depth_latents = input_view_othermodal_latents[:T_in]
                    target_rgbs = self.decode_latents(target_rgb_latents)
                    target_depths = self.decode_latents(target_depth_latents, use_depth_vae=True)
                    input_depths = self.decode_latents(input_depth_latents, use_depth_vae=True)
                    image = np.concatenate([target_rgbs, target_depths, input_depths], axis=0)
                    print(f"target_rgbs: {target_rgbs.shape}, target_depths: {target_depths.shape}, input_depths: {input_depths.shape}, image: {image.shape}")
                    
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            if self.depth_vae is None:
                # 8. Post-processing
                image = self.decode_latents(filtered_latents)
            else:
                num_target_imgs = real_bsz * num_tasks * T_out
                target_view_latents = filtered_latents[:num_target_imgs]
                input_view_othermodal_latents = filtered_latents[num_target_imgs:]
                # decode rgb and depth separately
                if num_tasks == 2:
                    target_rgb_latents = target_view_latents[:T_out]
                    target_depth_latents = target_view_latents[T_out:]
                    input_depth_latents = input_view_othermodal_latents[:T_in]
                    target_rgbs = self.decode_latents(target_rgb_latents)
                    target_depths, target_depth_confs = self.decode_latents(target_depth_latents, use_depth_vae=True, return_confidence=True)
                    input_depths, input_depth_confs = self.decode_latents(input_depth_latents, use_depth_vae=True, return_confidence=True)
                    image = np.concatenate([target_rgbs, target_depths, input_depths], axis=0)
                elif num_tasks == 3:
                    target_rgb_latents = target_view_latents[:T_out]
                    target_depth_latents = target_view_latents[T_out:T_out*2]
                    target_sem_latents = target_view_latents[T_out*2:T_out*3]
                    input_depth_latents = input_view_othermodal_latents[:T_in]
                    input_sem_latents = input_view_othermodal_latents[T_in:T_in*2]
                    target_rgbs = self.decode_latents(target_rgb_latents)
                    target_depths, target_depth_confs = self.decode_latents(target_depth_latents, use_depth_vae=True, return_confidence=True)
                    target_sems = self.decode_latents(target_sem_latents)
                    input_depths, input_depth_confs = self.decode_latents(input_depth_latents, use_depth_vae=True, return_confidence=True)
                    input_sems = self.decode_latents(input_sem_latents)
                    image = np.concatenate([target_rgbs, target_depths, target_sems, input_depths, input_sems], axis=0)
                    

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return SpatialGenDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, mvmm_latents=filtered_latents, input_depths_confi_maps=input_depth_confs, target_depths_confi_maps=target_depth_confs)
