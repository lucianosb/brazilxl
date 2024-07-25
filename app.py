import gradio as gr
import torch
from transformers import pipeline
from diffusers import StableDiffusionXLPipeline
# import spaces

default_model = "lucianosb/acaraje-brazil-xl"
default_prompt = "delicious acarajé"

pipeline = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/lucianosb/sinteticoXL-models/blob/main/sinteticoXL_v1dot2.safetensors",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

def make_description(model):
    """
    Generates a description based on the given model.

    Parameters:
        model (str): The name of the model.

    Returns:
        str: The generated description.

    """
    pipeline.unload_lora_weights()
    model_keywords = {
        "lucianosb/acaraje-brazil-xl": ["acarajé", "acaraje"],
        "lucianosb/boto-brazil-xl": ["boto"],
        "lucianosb/cathedral-of-brasilia-brazil-xl": ["cathedral of brasilia", "cathedral", "house/building"],
        "lucianosb/jkbridge-brazil-xl": ["jkbridge"],
        "lucianosb/mamulengo-brazil-xl": ["mamulengo puppet"],
        "lucianosb/marajoara-brazil-xl": ["marajoara patterns", "marajo patterns", "marajo"],
        "lucianosb/masp-brazil-xl": ["masp", "sampa", "sao_paulo", "masp building"],
        "lucianosb/ofobridge-brazil-xl": ["ofobridge", "bridge", "ponte_estaiada_sp", "ponte"],
        "lucianosb/tacaca-brazil-xl": ["tacacá"],
        "lucianosb/timbalada-brazil-xl": ["timbalada body painting", "body painting"],
        "lucianosb/veropeso-brazil-xl": ["veropeso", "veropa"]
    }
    
    keywords = model_keywords.get(model, ["unknown model"])

    return "Triggered with the following keywords: \n\n" + "- " + "\n- ".join(keywords)

def make_prompt(model):
    """
    Generates a prompt based on the given model.

    Parameters:
        model (str): The name of the model.

    Returns:
        str: The generated prompt.

    """
    prompts = {
        "lucianosb/acaraje-brazil-xl": "delicious acarajé",
        "lucianosb/boto-brazil-xl": "beautiful boto in a river",
        "lucianosb/cathedral-of-brasilia-brazil-xl": "stunning cathedral of brasilia",	
        "lucianosb/jkbridge-brazil-xl": "charming jkbridge",
        "lucianosb/mamulengo-brazil-xl": "enchanting mamulengo puppet in the clouds",
        "lucianosb/marajoara-brazil-xl": "a vase in marajoara patterns",
        "lucianosb/masp-brazil-xl": "elegant masp building, by sunrise in sao_paulo",
        "lucianosb/ofobridge-brazil-xl": "fascinating ofobridge",
        "lucianosb/tacaca-brazil-xl": "delicious tacaca, 4k masterpiece",
        "lucianosb/timbalada-brazil-xl": "beautiful timbalada body painting",
        "lucianosb/veropeso-brazil-xl": "dreamy veropeso landscape"
    }
    
    prompt = prompts.get(model, "")
    return prompt

# @spaces.GPU(duration=120)
def make_image(model, prompt):
    """
    Generates an image based on the given model and prompt.

    Args:
        model (str): The name of the model.
        prompt (str): The prompt to generate the image.

    Returns:
        PIL.Image.Image: The generated image.

    Raises:
        ValueError: If the model is invalid.
    """
    weight_file_names = {
        "lucianosb/acaraje-brazil-xl": "Acarajé_-_Brazil_XL.safetensors",
        "lucianosb/boto-brazil-xl": "Boto_-_Brazil_XL.safetensors",
        "lucianosb/cathedral-of-brasilia-brazil-xl": "Cathedral_of_Brasilia_-_Brazil_XL.safetensors",
        "lucianosb/jkbridge-brazil-xl": "JK_Bridge_-_Brazil_XL.safetensors",
        "lucianosb/mamulengo-brazil-xl": "Mamulengo_-_Brazil_XL.safetensors",
        "lucianosb/marajoara-brazil-xl": "Marajoara_-_Brazil_XL.safetensors",
        "lucianosb/masp-brazil-xl": "MASP_-_Brazil_XL.safetensors",
        "lucianosb/ofobridge-brazil-xl": "OFO_Bridge_-_Brazil_XL.safetensors",
        "lucianosb/tacaca-brazil-xl": "Tacacá_-_Brazil_XL.safetensors",
        "lucianosb/timbalada-brazil-xl": "Timbalada_-_Brazil_XL.safetensors",
        "lucianosb/veropeso-brazil-xl": "VeroPeso_-_Brazil_XL.safetensors"
    }
    
    weight_file_name = weight_file_names.get(model, None)
    
    if weight_file_name is None:
        raise ValueError(f"Invalid model: {model}")
    
    pipeline.load_lora_weights(model, weight_name=weight_file_name)
    prompt = prompt
    image = pipeline(prompt, guidance_scale=6.0, num_inference_steps=20).images[0]
    return image

with gr.Blocks(title="Brazil XL") as demo:
    gr.Markdown("# Brazil XL Demo")
    gr.Markdown('''
Brazil XL is an initiative that brings better representations of Brazilian culture to Stable Diffusion. This demo uses Stable Diffusion models from the [SinteticoXL](https://huggingface.co/lucianosb/sinteticoXL-models) family. 

## How to use

1. Choose a model
2. Describe your input
3. Click "Generate!"
4. See the result
''')
    
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                label="Choose a model", 
                choices=[
                    "lucianosb/acaraje-brazil-xl",
                    "lucianosb/boto-brazil-xl",
                    "lucianosb/cathedral-of-brasilia-brazil-xl",
                    "lucianosb/jkbridge-brazil-xl",
                    "lucianosb/mamulengo-brazil-xl",
                    "lucianosb/marajoara-brazil-xl",
                    "lucianosb/masp-brazil-xl",
                    "lucianosb/ofobridge-brazil-xl",
                    "lucianosb/tacaca-brazil-xl",
                    "lucianosb/timbalada-brazil-xl",
                    "lucianosb/veropeso-brazil-xl"
                ], 
                value=default_model
            )

            description = gr.Markdown("Describe your input for " + default_model)

            prompt = gr.Textbox(
                        label="Prompt",
                        info="use the proper keyword",
                        lines=3,
                        value=default_prompt,
                    )

        
            btn = gr.Button("Generate!")
        with gr.Column():
            output = gr.Image()
    model_dropdown.change(fn=make_description, inputs=model_dropdown, outputs=description)
    model_dropdown.change(fn=make_prompt, inputs=model_dropdown, outputs=prompt)
    btn.click(fn=make_image, inputs=[model_dropdown, prompt], outputs=output)

demo.launch(debug=True)