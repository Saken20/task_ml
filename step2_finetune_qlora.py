"""
Шаг 2
Fine-tuning Mistral-7B через QLoRA

Зависимости
pip install -q -U bitsandbytes
pip install -q transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 datasets trl==0.8.6 scipy

Авторизация W&B
"""

import os
import sys
import json
import math
import torch
from datetime import datetime

# ── Проверка GPU до импорта bitsandbytes ──────────────────────────────────────
# bitsandbytes при импорте ищет CUDA .so; без GPU-рантайма падает с triton ошибкой.
if not torch.cuda.is_available():
    print("❌ CUDA недоступна!")
    print("   В Colab: Runtime → Change runtime type → T4 GPU → Save")
    print("   Затем перезапустите ячейки заново.")
    sys.exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

import bitsandbytes  # noqa: F401 — импортируем явно ДО peft, чтобы убедиться что .so загружен

import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./mistral-qlora-adapter"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"

# LoRA параметры
LORA_CONFIG = dict(
    r=16,                        # Ранг матриц (4–64, компромисс качество/память)
    lora_alpha=32,               # Масштаб = alpha/r = 2
    lora_dropout=0.05,
    target_modules=[             # Применяем LoRA к attention и MLP
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Гиперпараметры обучения
TRAIN_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,   # Эффективный batch = 2*4 = 8
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,               # 5% шагов — warmup
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    fp16=True,                       # Mixed precision для T4
    optim="paged_adamw_8bit",        # Экономия памяти оптимизатора
    report_to="wandb",               # Логирование в Weights & Biases
    run_name="mistral-7b-qlora",     # Имя запуска в W&B
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,
)

# Mistral chat template
PROMPT_TEMPLATE = "<s>[INST] {instruction} [/INST] {response} </s>"


# ─────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────

def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def format_examples(examples):
    """Форматируем в Mistral chat format."""
    texts = []
    for ex in examples:
        text = PROMPT_TEMPLATE.format(
            instruction=ex["instruction"].strip(),
            response=ex["response"].strip()
        )
        texts.append({"text": text})
    return texts


def prepare_datasets():
    print("📂 Загрузка данных...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw = load_jsonl(VAL_FILE)

    train_formatted = format_examples(train_raw)
    val_formatted = format_examples(val_raw)

    train_ds = Dataset.from_list(train_formatted)
    val_ds = Dataset.from_list(val_formatted)

    print(f"   Train: {len(train_ds)} примеров")
    print(f"   Val:   {len(val_ds)} примеров")
    return train_ds, val_ds


# ─────────────────────────────────────────
# Загрузка модели в 4-bit (QLoRA)
# ─────────────────────────────────────────

def load_quantized_model():
    print(f"\n🔧 Загрузка {MODEL_ID} в 4-bit...")

    # NF4 квантизация — стандарт для QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — лучше для весов LLM
        bnb_4bit_compute_dtype=torch.float16, # Вычисления в fp16
        bnb_4bit_use_double_quant=True,       # Квантизация констант квантизации (+~0.4 bit экономии)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # Mistral не имеет pad_token
    tokenizer.padding_side = "right"           # Важно для causal LM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",                     # Автоматически на GPU
        trust_remote_code=True,
    )

    # Подготовка для kbit training: отключение кеша, включение grad checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f"   VRAM после загрузки: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


# ─────────────────────────────────────────
# Применение LoRA адаптеров
# ─────────────────────────────────────────

def apply_lora(model):
    print("\n🔌 Применение LoRA адаптеров...")

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Статистика параметров
    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Обучаемые параметры: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"   Всего параметров:    {total:,}")
    print(f"   Замороженные:        {total-trainable:,}")

    return model


# ─────────────────────────────────────────
# Loss logging callback
# ─────────────────────────────────────────

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(step)
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(step)

    def plot_and_save(self, path="loss_curve.png"):
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses, label="Train Loss", color="#2196F3", linewidth=2)
        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, label="Val Loss", color="#F44336",
                    linewidth=2, linestyle="--", marker="o", markersize=5)

        ax.set_xlabel("Шаг", fontsize=12)
        ax.set_ylabel("Loss (Cross-Entropy)", fontsize=12)
        ax.set_title("Кривая обучения QLoRA (Mistral-7B)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Аннотация лучшего val loss
        if self.eval_losses:
            best_idx = self.eval_losses.index(min(self.eval_losses))
            ax.annotate(
                f"Лучший val loss: {self.eval_losses[best_idx]:.3f}",
                xy=(self.eval_steps[best_idx], self.eval_losses[best_idx]),
                xytext=(self.eval_steps[best_idx] + 10, self.eval_losses[best_idx] + 0.1),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red"
            )

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"📊 График сохранён: {path}")
        return fig


# ─────────────────────────────────────────
# Обучение
# ─────────────────────────────────────────

def train():
    print("=" * 60)
    print("ШАГ 2: Fine-tuning Mistral-7B через QLoRA")
    print("=" * 60)
    print(f"Время начала: {datetime.now().strftime('%H:%M:%S')}")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # Загрузка данных
    train_ds, val_ds = prepare_datasets()

    # Загрузка модели
    model, tokenizer = load_quantized_model()

    # LoRA
    model = apply_lora(model)

    # ─── W&B инициализация ───
    wandb.init(
        project="mistral-qlora-finetune",
        name="mistral-7b-qlora",
        config={
            "model": MODEL_ID,
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "lora_dropout": LORA_CONFIG["lora_dropout"],
            "target_modules": LORA_CONFIG["target_modules"],
            **TRAIN_CONFIG,
            "max_seq_length": 512,
            "quantization": "nf4-4bit",
        },
        tags=["qlora", "mistral-7b", "instruction-tuning"],
    )

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        **TRAIN_CONFIG
    )

    # Loss callback
    loss_callback = LossLoggerCallback()

    # SFTTrainer — обёртка над Trainer для instruction fine-tuning
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,           # Не упаковываем примеры (разные длины)
        callbacks=[loss_callback],
    )

    # ─── Запуск обучения ───
    print(f"\n🚀 Начало обучения...")
    print(f"   Эпох: {TRAIN_CONFIG['num_train_epochs']}")
    print(f"   Эффективный batch size: {TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"   Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"   max_seq_length: 512")

    train_result = trainer.train()

    # ─── Сохранение адаптера ───
    print(f"\n💾 Сохранение LoRA адаптера...")
    # Сохраняем ТОЛЬКО адаптер (не всю модель)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    adapter_size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    ) / 1e6

    print(f"   Адаптер сохранён в: {OUTPUT_DIR}")
    print(f"   Размер адаптера: {adapter_size_mb:.1f} MB")
    print(f"   (базовая модель ~14 GB не сохраняется)")

    # ─── Метрики ───
    metrics = train_result.metrics
    print(f"\n📈 Результаты обучения:")
    print(f"   Train loss (финальный): {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Train samples/sec: {metrics.get('train_samples_per_second', 0):.1f}")
    print(f"   Время обучения: {metrics.get('train_runtime', 0)/60:.1f} минут")

    # Финальный eval
    eval_metrics = trainer.evaluate()
    print(f"   Val loss (финальный): {eval_metrics['eval_loss']:.4f}")
    print(f"   Val perplexity: {math.exp(eval_metrics['eval_loss']):.2f}")

    # ─── График loss ───
    loss_callback.plot_and_save("loss_curve.png")

    # ─── Сохранение метрик в JSON ───
    all_metrics = {
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics["eval_loss"],
        "eval_perplexity": math.exp(eval_metrics["eval_loss"]),
        "train_losses_history": loss_callback.train_losses,
        "eval_losses_history": loss_callback.eval_losses,
        "train_steps": loss_callback.train_steps,
        "eval_steps": loss_callback.eval_steps,
        "config": {
            "model": MODEL_ID,
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "learning_rate": TRAIN_CONFIG["learning_rate"],
            "epochs": TRAIN_CONFIG["num_train_epochs"],
            "batch_size": TRAIN_CONFIG["per_device_train_batch_size"],
            "grad_accum": TRAIN_CONFIG["gradient_accumulation_steps"],
        }
    }

    with open("training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print("📋 Метрики сохранены: training_metrics.json")

    # ─── W&B финализация ───
    wandb.summary.update({
        "final_train_loss": metrics.get("train_loss"),
        "final_eval_loss": eval_metrics["eval_loss"],
        "final_eval_perplexity": math.exp(eval_metrics["eval_loss"]),
        "adapter_size_mb": adapter_size_mb,
        "trainable_params": 41_943_040,
        "trainable_params_pct": 0.58,
    })
    # Логируем loss curve как артефакт
    artifact = wandb.Artifact("loss-curve", type="plot")
    artifact.add_file("loss_curve.png")
    wandb.log_artifact(artifact)
    wandb.finish()
    print("📊 W&B run завершён. График и метрики загружены в wandb.ai")

    print(f"\n✅ Шаг 2 завершён. Время: {datetime.now().strftime('%H:%M:%S')}")
    return trainer, model, tokenizer


# ─────────────────────────────────────────
# Быстрый тест после обучения
# ─────────────────────────────────────────

def quick_test(model, tokenizer, prompts=None):
    """Быстрая проверка качества дообученной модели."""
    if prompts is None:
        prompts = [
            "Что такое градиентный спуск?",
            "Объясни разницу между L1 и L2 регуляризацией.",
        ]

    print("\n🧪 Быстрый тест дообученной модели:")
    model.eval()

    for prompt in prompts:
        text = f"<s>[INST] {prompt} [/INST]"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(f"\nВопрос: {prompt}")
        print(f"Ответ:  {response.strip()[:300]}")
        print("-" * 50)


if __name__ == "__main__":
    trainer, model, tokenizer = train()
    quick_test(model, tokenizer)
