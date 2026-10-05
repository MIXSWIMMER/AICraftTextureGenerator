import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel

# ==========================================
# 1. КОНФИГУРАЦИЯ И ГИПЕРПАРАМЕТРЫ
# ==========================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # Базовая модель SD 1.5
OUTPUT_DIR = "lora_voxel_textures"  # Папка для сохранения весов LoRA

# Пути к вашим Parquet-файлам
TRAIN_PARQUET = "data/train.parquet"
VAL_PARQUET = "data/validation.parquet"
TEST_PARQUET = "data/test.parquet"

# Гиперпараметры обучения
RESOLUTION = 512  # Входной размер для SD 1.5
TARGET_TEXTURE_SIZE = 16  # Разрешение воксельной текстуры в игре (16x16 / 32x32)
BATCH_SIZE = 4  # Уменьшите до 2 или 1, если не хватает VRAM
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
LORA_RANK = 32  # Ранг LoRA (16 или 32)
LORA_ALPHA = 32
SAVE_EVERY_N_STEPS = 500  # Интервал валидации и сохранения
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. ВКЛЮЧЕНИЕ БЕСШОВНОСТИ (CIRCULAR PADDING)
# ==========================================
def enable_circular_padding(model):
    """
    Переводит все сверточные слои в режим circular padding.
    Благодаря этому граница левого края подтягивает пиксели правого края,
    что делает сгенерированные текстуры блоков полностью бесшовными.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = 'circular'
    print("✓ Включен Circular Padding для бесшовных текстур (Seamless Tiling).")


# ==========================================
# 3. ПОДГОТОВКА ДАТАСЕТА И ДАТАЛОАДЕРА
# ==========================================
print("--> Загрузка датасетов из Parquet...")
dataset = load_dataset("parquet", data_files={
    "train": TRAIN_PARQUET,
    "validation": VAL_PARQUET,
    "test": TEST_PARQUET
})

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

# Трансформации изображений (ближайший сосед растягивает 16x16 в 512x512 без размытия)
image_transforms = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def preprocess_function(examples):
    # 1. Преобразование изображений из Parquet (PIL) в RGB
    images = [img.convert("RGB") for img in examples["image"]]
    examples["pixel_values"] = [image_transforms(img) for img in images]

    # 2. Формирование динамического промпта из колонок датасета
    prompts = []

    names = examples.get("texture_name", [""] * len(images))
    styles = examples.get("texture_style", [""] * len(images))
    colors = examples.get("primary_colors", [""] * len(images))
    patterns = examples.get("pattern_description", [""] * len(images))
    tileable = examples.get("tileable_direction", [""] * len(images))
    descriptions = examples.get("overall_texture_description", [""] * len(images))

    for name, style, color, pattern, tile, desc in zip(names, styles, colors, patterns, tileable, descriptions):
        prompt_parts = []

        if name:
            prompt_parts.append(f"{name}")
        if style:
            prompt_parts.append(f"{style} style")
        if color:
            prompt_parts.append(f"{color} color palette")
        if pattern:
            prompt_parts.append(f"pattern: {pattern}")
        if tile:
            prompt_parts.append(f"seamless {tile} tiling")
        if desc:
            prompt_parts.append(f"{desc}")

        prompt_parts.append("pixel art, 16x16 texture, voxel block style")

        full_prompt = ", ".join(filter(None, prompt_parts))
        prompts.append(full_prompt)

    # 3. Токенизация собранных промптов
    examples["input_ids"] = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    return examples


train_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
val_dataset = dataset["validation"].map(preprocess_function, batched=True,
                                        remove_columns=dataset["validation"].column_names)


def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 4. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ И LORA
# ==========================================
print("--> Загрузка компонентов Stable Diffusion...")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(DEVICE)

# Замораживаем веса VAE и Text Encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Применяем Circular Padding для бесшовности
enable_circular_padding(unet)

# Конфигурация LoRA для UNet
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# Оптимизатор
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)


# ==========================================
# 5. ФУНКЦИЯ ВАЛИДАЦИИ (ГЕНЕРАЦИЯ ПРИМЕРОВ)
# ==========================================
def run_validation(step, prompt="magma stone block, pixel art, voxel texture, seamless"):
    print(f"\n[Запуск валидации на шаге {step}...] Prompt: '{prompt}'")

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    enable_circular_padding(pipeline.unet)
    pipeline.set_progress_bar_config(disable=True)

    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=25).images[0]

        # Уменьшение размера алгоритмом Nearest Neighbor до воксельного формата (16x16)
        pixel_texture = image.resize((TARGET_TEXTURE_SIZE, TARGET_TEXTURE_SIZE), Image.NEAREST)

        val_save_path = os.path.join(OUTPUT_DIR, f"val_step_{step}.png")
        pixel_texture.save(val_save_path)
        print(f"✓ Валидационное изображение сохранено: {val_save_path}\n")


# ==========================================
# 6. ЦИКЛ ОБУЧЕНИЯ С ПРОГРЕСС-БАРОМ (tqdm)
# ==========================================
print("--> Начало процесса обучения LoRA...")
print(f"DEVICE: {DEVICE}")
global_step = 0
total_steps = NUM_EPOCHS * len(train_dataloader)

# Главный прогресс-бар по эпохам
epoch_progress_bar = tqdm(range(NUM_EPOCHS), desc="Эпохи", position=0)

for epoch in epoch_progress_bar:
    unet.train()

    # Прогресс-бар по батчам внутри эпохи
    step_progress_bar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Эпоха {epoch + 1}/{NUM_EPOCHS}",
        position=1,
        leave=False
    )

    for step, batch in step_progress_bar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)

        # Переводим картинку в латентное пространство VAE
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Генерируем случайный шум
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=DEVICE).long()

        # Добавляем шум к латентам
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Получаем эмбеддинги текста
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Предсказываем шум через UNet с весами LoRA
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Вычисляем MSE Loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        # Обновляем инфо о Loss и Step в статус-строке прогресс-бара
        step_progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Step": f"{global_step}/{total_steps}"})

        if global_step % SAVE_EVERY_N_STEPS == 0:
            run_validation(global_step)
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
            unet.save_pretrained(ckpt_dir)

# ==========================================
# 7. ФИНАЛЬНОЕ СОХРАНЕНИЕ
# ==========================================
print(f"\n--> Обучение завершено! Сохранение итоговой LoRA модели в {OUTPUT_DIR}...")
unet.save_pretrained(OUTPUT_DIR)
print("✓ Модель успешно сохранена!")