import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "lora_voxel_textures/checkpoint-2500"  # Путь к папке с обученной LoRA (или к checkpoint-XXXX)
OUTPUT_DIR = "generated_textures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 32  # Итоговый размер текстуры для AICraft

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. ВКЛЮЧЕНИЕ БЕСШОВНОСТИ (CIRCULAR PADDING)
# ==========================================
def enable_circular_padding(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = 'circular'


# ==========================================
# 3. ЗАГРУЗКА ПАЙПЛАЙНА И LORA
# ==========================================
print("--> Загрузка Stable Diffusion и весов LoRA...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
).to(DEVICE)

# Подгружаем обученные веса LoRA в UNet
pipe.load_lora_weights(LORA_PATH)

# Включаем бесшовность
enable_circular_padding(pipe.unet)

# ==========================================
# 4. ТЕСТОВАЯ ГЕНЕРАЦИЯ
# ==========================================
# Текстовые промпты для проверки
prompts = [
    "iron sword item, pixel art, 16x16 texture, voxel item style, seamless"]

print("--> Генерация текстур...")
for i, prompt in enumerate(prompts):
    with torch.no_grad():
        # 1. Генерация полноразмерной картины (512x512)
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 2. Уменьшение до 16x16 с помощью алгоритма NEAREST (пиксель-арт без размытия)
        texture_16x16 = image.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)

        # 3. Сохранение обоих вариантов для сравнения
        orig_path = os.path.join(OUTPUT_DIR, f"texture_{i + 1}_512.png")
        small_path = os.path.join(OUTPUT_DIR, f"texture_{i + 1}_16x16.png")

        image.save(orig_path)
        texture_16x16.save(small_path)

        print(f"✓ Текстура #{i + 1} сохранена: {small_path}")

print(f"\nВсе готово! Проверьте результаты в папке '{OUTPUT_DIR}'.")