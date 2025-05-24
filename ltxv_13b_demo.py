https://huggingface.co/calcuis/ltxv-gguf

https://github.com/calcuis/gguf-connector

pip install gguf-connector
pip install opencv-python-headless

cp /environment/miniconda3/envs/system/lib/python3.11/site-packages/gguf_connector/vg2.py .

#### ltxv 2b demo

import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
#import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

#model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-2b-0.9.6-distilled-fp32-q8_0.gguf"
model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-13b-0.9.7-distilled-fp32-q8_0.gguf"
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)
pipe = LTXConditionPipeline.from_pretrained(
    #"callgg/ltxv0.9.6-decoder",
    "callgg/ltxv0.9.7-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    image = input_image.convert("RGB")
    condition1 = LTXVideoCondition(image=image, frame_index=0)
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

from PIL import Image
im = Image.open("image_86.jpg")
generate_video(
    im,
    "a beautiful landscape of anime scene",
    "",
    768,
    512,
    81,
    50,
    24
)

#### ltxv 13b demo
import json
with open("cfg097.json", "w") as f:
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }
    json.dump(transformer_13b_config, f)

import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
#import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

#model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-2b-0.9.6-distilled-fp32-q8_0.gguf"
#model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-13b-0.9.7-distilled-fp32-q4_0.gguf"
model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-13b-0.9.7-dev-fp32-q4_0.gguf"
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config = "cfg097.json"
)
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)
pipe = LTXConditionPipeline.from_pretrained(
    #"callgg/ltxv0.9.6-decoder",
    "callgg/ltxv0.9.7-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    image = input_image.convert("RGB")
    condition1 = LTXVideoCondition(image=image, frame_index=0)
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

from PIL import Image
im = Image.open("image_86.jpg")
generate_video(
    im,
    "a beautiful landscape of anime scene",
    "",
    768,
    512,
    81,
    24,
    24
)

vim run_ltxv_app.py

import json
with open("cfg097.json", "w") as f:
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }
    json.dump(transformer_13b_config, f)

import torch
import gradio as gr
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-13b-0.9.7-dev-fp32-q4_0.gguf"
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="cfg097.json"
)
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)
pipe = LTXConditionPipeline.from_pretrained(
    "callgg/ltxv0.9.7-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    image = input_image.convert("RGB")
    condition1 = LTXVideoCondition(image=image, frame_index=0)
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

# Gradio UI
sample_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
sample_prompts = [[x] for x in sample_prompts]

block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🎥 Video Generator (i2v)")
    with gr.Column():
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload Image")
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
        negative_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", visible=False) # disable
        quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt])
        quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt, show_progress=False, queue=False)
        with gr.Row():
            width = gr.Number(label="Width", value=832)
            height = gr.Number(label="Height", value=480)
            num_frames = gr.Number(label="Num Frames", value=81)
            num_inference_steps = gr.Number(label="Num Inference Steps", value=25)
            fps = gr.Number(label="FPS", value=24)
        generate_btn = gr.Button("Generate")
    with gr.Column():
        output_video = gr.Video(label="Generated Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps],
        outputs=output_video,
    )

block.launch(share = True)

vim run_ltxv.py

import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

# Configuration setup
with open("cfg097.json", "w") as f:
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }
    json.dump(transformer_13b_config, f)

# Initialize the model pipeline
model_path = "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-13b-0.9.7-dev-fp32-q4_0.gguf"
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="cfg097.json"
)
text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)
pipe = LTXConditionPipeline.from_pretrained(
    "callgg/ltxv0.9.7-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
).to("cuda")

def generate_video(input_image, prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps, output_path):
    image = input_image.convert("RGB")
    condition1 = LTXVideoCondition(image=image, frame_index=0)
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(
        conditions=[condition1],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]
    export_to_video(video, output_path, fps=fps)
    return output_path

# Process all files in the input directory
input_dir = "Dont_be_your_lover_Images_Captioned"
output_dir = "Dont_be_your_lover_LTXV13b_Videos_Captioned"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all .png files in the input directory
png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# Process each file with tqdm progress bar
for png_file in tqdm(png_files, desc="Processing images"):
    base_name = os.path.splitext(png_file)[0]
    txt_file = f"{base_name}.txt"

    # Check if corresponding text file exists
    if not os.path.exists(os.path.join(input_dir, txt_file)):
        tqdm.write(f"Skipping {png_file}: No corresponding .txt file found")
        continue

    # Read the image and prompt
    image_path = os.path.join(input_dir, png_file)
    prompt_path = os.path.join(input_dir, txt_file)

    try:
        # Load image and prompt
        im = Image.open(image_path)
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()

        # Generate output path
        output_path = os.path.join(output_dir, f"{base_name}.mp4")

        # Generate video
        generate_video(
            im,
            prompt,
            "",
            768,
            512,
            81,
            24,
            24,
            output_path
        )

        # Copy the text file to output directory
        output_txt_path = os.path.join(output_dir, txt_file)
        with open(prompt_path, 'r') as src, open(output_txt_path, 'w') as dst:
            dst.write(src.read())

    except Exception as e:
        tqdm.write(f"Error processing {png_file}: {str(e)}")
        continue

print("Processing complete!")
