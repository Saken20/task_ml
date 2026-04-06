"""
Шаг 3: Оценка

Дополнительные зависимости:
pip install -q rouge_score bert_score
"""

import json
import math
import time
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Конфигурация
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "./mistral-qlora-adapter"
RESULTS_FILE = "evaluation_results.json"
PROMPT_TEMPLATE = "<s>[INST] {instruction} [/INST]"

# 15 тестовых примеров
# Вопросы по темам, которые были в датасете дообучения.
TEST_EXAMPLES = [
    {
        "instruction": "Что такое gradient checkpointing и зачем он нужен?",
        "reference": (
            "Gradient checkpointing — техника экономии GPU памяти при обучении. "
            "Промежуточные активации не хранятся, а пересчитываются при backprop. "
            "Это снижает потребление памяти в √N раз ценой ~20% замедления. "
            "Обязательно для обучения 7B+ моделей на T4."
        )
    },
    {
        "instruction": "Что такое LoRA и зачем нужны адаптеры?",
        "reference": (
            "LoRA добавляет к замороженным весам низкоранговое обновление ΔW = A·B. "
            "Параметров в 100–1000 раз меньше чем при полном fine-tuning. "
            "Адаптер сохраняется отдельно (10–100 MB) и применяется поверх базовой модели."
        )
    },
    {
        "instruction": "Объясни понятие overfitting простыми словами.",
        "reference": (
            "Overfitting — модель запоминает обучающие данные вместо закономерностей. "
            "Признак: train loss падает, val loss растёт. "
            "Борьба: dropout, регуляризация, early stopping, аугментация данных."
        )
    },
    {
        "instruction": "Что такое tokenizer в NLP?",
        "reference": (
            "Токенизатор разбивает текст на токены для языковой модели. "
            "Методы: BPE, WordPiece, SentencePiece. "
            "Mistral использует BPE со словарём ~32K токенов. "
            "Правило: 1 токен ≈ 4 символа (английский)."
        )
    },
    {
        "instruction": "Что такое PEFT и какие методы он включает?",
        "reference": (
            "PEFT — Parameter-Efficient Fine-Tuning, семейство методов с минимальным числом обучаемых параметров. "
            "Методы: LoRA, Prefix Tuning, Prompt Tuning, IA³. "
            "LoRA — самый популярный, лучший баланс качество/память."
        )
    },
    {
        "instruction": "Чем отличается batch normalization от layer normalization?",
        "reference": (
            "Batch Norm нормализует по батчу (ось N) — зависит от размера батча. "
            "Layer Norm нормализует по слою (ось C) — не зависит от батча. "
            "В трансформерах используют Layer Norm: батч может быть размером 1."
        )
    },
    {
        "instruction": "Что такое attention mask в трансформерах?",
        "reference": (
            "Attention mask — бинарная маска, указывающая какие токены учитывать (1) и игнорировать (0). "
            "Нужна для padding: все токены-заполнители маскируются. "
            "В causal LM также маскирует будущие токены (causal mask)."
        )
    },
    {
        "instruction": "Что такое hugging face PEFT библиотека?",
        "reference": (
            "HuggingFace PEFT — библиотека для эффективного дообучения больших моделей. "
            "Поддерживает: LoRA, QLoRA, Prefix Tuning, Prompt Tuning. "
            "Интегрирована с transformers и trl. "
            "Позволяет дообучать 7B модели на GPU с 16 GB."
        )
    },
    {
        "instruction": "Что такое perplexity и как её интерпретировать?",
        "reference": (
            "Perplexity = exp(cross-entropy loss) — мера удивлённости модели. "
            "PP=1: идеальная модель. PP=100: неплохо. PP=1000+: слабая. "
            "Mistral-7B на WikiText-2: около 5–6."
        )
    },
    {
        "instruction": "Как работает cosine learning rate scheduler?",
        "reference": (
            "Cosine scheduler плавно снижает lr по форме косинуса: от max до min. "
            "Формула: lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(π*step/total)). "
            "Часто комбинируют с warmup первые N шагов. "
            "Стандарт для fine-tuning LLM."
        )
    },
    {
        "instruction": "Что такое instruction tuning?",
        "reference": (
            "Instruction tuning — дообучение LLM на парах (инструкция, ответ). "
            "Превращает base model в assistant. "
            "Датасеты: Alpaca (52K), Dolly (15K). "
            "Форматы: Mistral [INST]...[/INST], ChatML, Alpaca-style."
        )
    },
    {
        "instruction": "Что такое QLoRA?",
        "reference": (
            "QLoRA = квантизация базовой модели (4-bit NF4) + LoRA адаптеры в полной точности. "
            "Позволяет обучать Mistral-7B на GPU с 16 GB вместо 28 GB. "
            "Качество сопоставимо с полным fine-tuning."
        )
    },
    {
        "instruction": "Что такое weight decay?",
        "reference": (
            "Weight decay — L2 регуляризация в оптимизаторе. "
            "Каждый шаг: w ← w·(1 - lr·λ). "
            "AdamW — стандарт для LLM, типичное значение 0.01–0.1. "
            "Предотвращает слишком большие веса."
        )
    },
    {
        "instruction": "Объясни разницу между Adam и AdamW.",
        "reference": (
            "Adam: адаптивные моменты, L2 регуляризация через loss (некорректно). "
            "AdamW: исправляет weight decay — применяет его напрямую к весам, не через градиент. "
            "AdamW — стандарт для трансформеров, стабильнее при обучении."
        )
    },
    {
        "instruction": "Что такое BERTScore?",
        "reference": (
            "BERTScore — метрика качества текста через эмбеддинги BERT. "
            "Вычисляет косинусное сходство токенов prediction и reference. "
            "Преимущество над ROUGE: понимает семантику, а не точное совпадение слов."
        )
    },
]


# Загрузка моделей
def load_models():
    """Загружаем базовую и дообученную модель."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("📥 Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("📥 Загрузка базовой модели (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.eval()

    print("📥 Загрузка дообученной модели (base + LoRA адаптер)...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    finetuned_model.eval()

    print("✅ Обе модели загружены")
    return tokenizer, base_model, finetuned_model


# Генерация
GENERATION_CONFIG = dict(
    max_new_tokens=250,
    temperature=0.3,
    do_sample=True,
    repetition_penalty=1.1,
)

def generate(model, tokenizer, instruction):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


# Метрики
def compute_rouge(predictions, references):
    """ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rL),
    }


def compute_bertscore(predictions, references, lang="ru"):
    """BERTScore F1."""
    print("   Вычисление BERTScore (может занять 1-2 мин)...")
    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_avg_length(texts):
    return np.mean([len(t.split()) for t in texts])


# Основная функция оценки
def evaluate():
    print("=" * 60)
    print("ШАГ 3: Оценка базовой vs дообученной модели")
    print("=" * 60)

    tokenizer, base_model, finetuned_model = load_models()

    instructions = [ex["instruction"] for ex in TEST_EXAMPLES]
    references = [ex["reference"] for ex in TEST_EXAMPLES]

    # Генерация ответов
    print(f"\n🤖 Генерация ответов для {len(TEST_EXAMPLES)} примеров...")

    base_predictions = []
    ft_predictions = []

    for i, instruction in enumerate(instructions):
        print(f"   [{i+1}/{len(instructions)}] {instruction[:50]}...")

        base_resp = generate(base_model, tokenizer, instruction)
        base_predictions.append(base_resp)

        ft_resp = generate(finetuned_model, tokenizer, instruction)
        ft_predictions.append(ft_resp)

    # ROUGE
    print("\n📊 Вычисление метрик...")
    print("   ROUGE...")
    base_rouge = compute_rouge(base_predictions, references)
    ft_rouge = compute_rouge(ft_predictions, references)

    # BERTScore
    print("   BERTScore (базовая модель)...")
    base_bert = compute_bertscore(base_predictions, references)
    print("   BERTScore (дообученная)...")
    ft_bert = compute_bertscore(ft_predictions, references)

    # Длина ответов
    base_len = compute_avg_length(base_predictions)
    ft_len = compute_avg_length(ft_predictions)

    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)

    print(f"\n{'Метрика':<25} {'Базовая':>12} {'Дообученная':>14} {'Δ':>8}")
    print("-" * 62)

    metrics_to_compare = [
        ("ROUGE-1", base_rouge["rouge1"], ft_rouge["rouge1"]),
        ("ROUGE-2", base_rouge["rouge2"], ft_rouge["rouge2"]),
        ("ROUGE-L", base_rouge["rougeL"], ft_rouge["rougeL"]),
        ("BERTScore F1", base_bert["bertscore_f1"], ft_bert["bertscore_f1"]),
        ("BERTScore Prec", base_bert["bertscore_precision"], ft_bert["bertscore_precision"]),
        ("BERTScore Rec", base_bert["bertscore_recall"], ft_bert["bertscore_recall"]),
        ("Avg длина (токен)", base_len, ft_len),
    ]

    for name, base_val, ft_val in metrics_to_compare:
        delta = ft_val - base_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        indicator = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{name:<25} {base_val:>12.4f} {ft_val:>14.4f} {delta_str:>6} {indicator}")

    print("\n" + "=" * 60)
    print("ПРИМЕРЫ ОТВЕТОВ (первые 3)")
    print("=" * 60)

    for i in range(min(3, len(TEST_EXAMPLES))):
        print(f"\n📌 Вопрос: {instructions[i]}")
        print(f"\n  Базовая модель:\n  {base_predictions[i][:300]}")
        print(f"\n  Дообученная:\n  {ft_predictions[i][:300]}")
        print(f"\n  Эталон:\n  {references[i][:200]}")
        print("-" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_examples": len(TEST_EXAMPLES),
        "base_model": {
            "rouge": base_rouge,
            "bertscore": base_bert,
            "avg_response_length": base_len,
        },
        "finetuned_model": {
            "rouge": ft_rouge,
            "bertscore": ft_bert,
            "avg_response_length": ft_len,
        },
        "per_example": [
            {
                "instruction": instructions[i],
                "reference": references[i],
                "base_response": base_predictions[i],
                "finetuned_response": ft_predictions[i],
            }
            for i in range(len(TEST_EXAMPLES))
        ]
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: {RESULTS_FILE}")

    # Визуализация
    plot_comparison(base_rouge, ft_rouge, base_bert, ft_bert)

    print("\n✅ Шаг 3 завершён.")
    return results


# Визуализация
def plot_comparison(base_rouge, ft_rouge, base_bert, ft_bert):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Базовая vs Дообученная модель (Mistral-7B + QLoRA)", fontsize=14, fontweight="bold")

    # График ROUGE
    ax1 = axes[0]
    rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    base_vals = [base_rouge["rouge1"], base_rouge["rouge2"], base_rouge["rougeL"]]
    ft_vals = [ft_rouge["rouge1"], ft_rouge["rouge2"], ft_rouge["rougeL"]]

    x = np.arange(len(rouge_labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, base_vals, width, label="Базовая", color="#90CAF9", edgecolor="black")
    bars2 = ax1.bar(x + width/2, ft_vals, width, label="Дообученная", color="#1976D2", edgecolor="black")

    ax1.set_title("ROUGE метрики")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rouge_labels)
    ax1.set_ylabel("F1 Score")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#1565C0")

    # График BERTScore
    ax2 = axes[1]
    bert_labels = ["Precision", "Recall", "F1"]
    base_bert_vals = [base_bert["bertscore_precision"], base_bert["bertscore_recall"], base_bert["bertscore_f1"]]
    ft_bert_vals = [ft_bert["bertscore_precision"], ft_bert["bertscore_recall"], ft_bert["bertscore_f1"]]

    x2 = np.arange(len(bert_labels))
    bars3 = ax2.bar(x2 - width/2, base_bert_vals, width, label="Базовая", color="#A5D6A7", edgecolor="black")
    bars4 = ax2.bar(x2 + width/2, ft_bert_vals, width, label="Дообученная", color="#388E3C", edgecolor="black")

    ax2.set_title("BERTScore")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bert_labels)
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.set_ylim(0.7, 1.0)  # BERTScore обычно в диапазоне 0.8–1.0
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#1B5E20")

    plt.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=150)
    print("📊 График сохранён: evaluation_comparison.png")


# Анализ результатов
def analyze_results(results_path=RESULTS_FILE):
    """Загружает сохранённые результаты и выводит аналитику."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    base = results["base_model"]
    ft = results["finetuned_model"]

    print("\n" + "=" * 60)
    print("АНАЛИТИКА И ВЫВОДЫ")
    print("=" * 60)

    rouge_delta = ft["rouge"]["rougeL"] - base["rouge"]["rougeL"]
    bert_delta = ft["bertscore"]["bertscore_f1"] - base["bertscore"]["bertscore_f1"]

    print(f"\n📈 Изменение ROUGE-L:       {rouge_delta:+.4f} ({'улучшение' if rouge_delta > 0 else 'ухудшение'})")
    print(f"📈 Изменение BERTScore F1:  {bert_delta:+.4f} ({'улучшение' if bert_delta > 0 else 'ухудшение'})")

    len_change = ft["avg_response_length"] - base["avg_response_length"]
    print(f"📏 Изменение длины ответа:  {len_change:+.1f} токенов")

    print("\n💡 Интерпретация:")
    if rouge_delta > 0.02:
        print("  • ROUGE: дообученная модель лучше совпадает с эталонами по n-граммам.")
    elif rouge_delta > -0.02:
        print("  • ROUGE: результаты сопоставимы. ROUGE не улавливает стилистических улучшений.")
    else:
        print("  • ROUGE: снижение. Возможно, модель генерирует более разнообразные ответы.")

    if bert_delta > 0.01:
        print("  • BERTScore: семантическая близость к эталонам выросла — стиль ближе к целевому.")
    elif bert_delta > -0.01:
        print("  • BERTScore: без значимых изменений. 200 примеров — минимум для стабильного сдвига.")
    else:
        print("  • BERTScore: снижение. Проверьте качество датасета и параметры обучения.")

    print("\n⚠️  Ограничения оценки:")
    print("  • 15 тестовых примеров — недостаточно для статистической значимости")
    print("  • ROUGE/BERTScore не оценивают структурность и лаконичность ответов")
    print("  • Для полной оценки нужна human evaluation или LLM-as-judge")
    print("  • 200 примеров и 3 эпохи — минимальный fine-tuning; больше данных → лучше результат")


if __name__ == "__main__":
    results = evaluate()
    analyze_results()