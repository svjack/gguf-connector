pip install diffusers transformers peft torch torchvision gradio "httpx[socks]" gguf
pip install gguf-connector
pip install -U transformers ftfy
#git clone https://github.com/ggml-org/llama.cpp
#cd llama.cpp/gguf-py
#pip install --editable .

vim run_ani_gguf_app.py

import torch # optional (need torch, diffusers to work; pip install torch, diffusers)
import gradio as gr # optional (need gradio for lazy ui; pip install gradio)
from transformers import UMT5EncoderModel
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

#model_path = "https://huggingface.co/calcuis/wan-gguf/blob/main/wan2.1_t2v_1.3b-q8_0.gguf"
#model_path = "https://huggingface.co/svjack/hakoniwa_anime_wan2_1_models/blob/main/aniWan2114BFp8E4m3fn_t2v14BGGUFQ4KS.gguf"
model_path = "aniWan2114BFp8E4m3fn_t2v14BGGUFQ4KS.gguf"
transformer = WanTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

text_encoder = UMT5EncoderModel.from_pretrained(
    "chatpig/umt5xxl-encoder-gguf",
    gguf_file="umt5xxl-encoder-q4_0.gguf",
    torch_dtype=torch.bfloat16,
)

vae = AutoencoderKLWan.from_pretrained(
    "callgg/t2v-decoder",
    subfolder="vae",
    torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    "callgg/t2v-decoder",
    transformer=transformer,
    text_encoder=text_encoder,
    vae=vae, torch_dtype=torch.bfloat16
)
# pipe.to("cuda")
pipe.enable_model_cpu_offload()

def generate_video(prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps):
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    export_to_video(video, "output.mp4", fps=fps)
    return "output.mp4"

# Gradio UI

sample_prompts = [
    'anime style ,A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.',
    'anime style ,A pig moving quickly in a beautiful winter scenery nature trees sunset tracking camera.',
]
sample_prompts = [[x] for x in sample_prompts]

block = gr.Blocks(title="gguf").queue()
with block:
    gr.Markdown("## 🎥 Video 2")
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here (or click Sample Prompt)", value="")
        neg_prompt_input = gr.Textbox(label="Negative Prompt", value="blurry ugly bad", visible=False) # disable
        quick_prompts = gr.Dataset(samples=sample_prompts, label='Sample Prompt', samples_per_page=1000, components=[prompt_input])
        quick_prompts.click(lambda x: x[0], inputs=[quick_prompts], outputs=prompt_input, show_progress=False, queue=False)
    with gr.Row():
        width_input = gr.Number(label="Width", value=832)
        height_input = gr.Number(label="Height", value=480)
        num_frames_input = gr.Number(label="Number of Frames", value=81)
        num_steps_input = gr.Number(label="Inference Steps", value=25)
        fps_input = gr.Number(label="FPS", value=16)
    generate_btn = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt_input, neg_prompt_input,
            width_input, height_input,
            num_frames_input, num_steps_input,
            fps_input
        ],
        outputs=output_video,
    )

block.launch(share = True)
