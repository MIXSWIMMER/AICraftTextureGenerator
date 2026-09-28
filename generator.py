import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_voxel_textures"

# 1. Загрузка пайплайна
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to("cuda")

# 2. Подгрузка обученной LoRA
pipe.load_lora_weights(LORA_PATH)

# 3. Включение бесшовности
for module in pipe.unet.modules():
    if isinstance(module, torch.nn.Conv2d):
        module.padding_mode = 'circular'

# 4. Генерация текстуры по запросу
prompt = "crystallized slime block, glowing, voxel style, seamless texture"
raw_image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

# 5. Приведение к размеру воксельного блока (32x32)
pixel_texture = raw_image.resize((32, 32), Image.NEAREST)
pixel_texture.save("generated_block.png")

print("✓ Текстура успешно сгенерирована и сохранена!")