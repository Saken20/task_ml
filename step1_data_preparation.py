"""
Шаг 1
Подготовка датасета для дообучения Mistral-7B
Стиль: лаконичные, структурированные ответы в духе технического наставника.
Датасетт: tatsu-lab/alpaca (HuggingFace)
"""

import json
import random
import re
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def load_base_dataset():
    """Загружаем tatsu-lab/alpaca — 52K instruction-following примеров от Stanford.

    Поля датасета:
      - instruction: задание/вопрос
      - input:       дополнительный контекст (часто пустой)
      - output:      ответ модели
      - text:        полный промпт в Alpaca-формате (не используем напрямую)
    """
    print("📥 Загрузка датасета tatsu-lab/alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"   Загружено {len(ds)} примеров")
    return ds

# Фильтрация качества
MIN_INSTRUCTION_LEN = 20
MAX_INSTRUCTION_LEN = 400
MIN_RESPONSE_LEN = 30
MAX_RESPONSE_LEN = 800

def quality_filter(example):
    # Alpaca: поля instruction, input, output
    instruction = example.get("instruction", "")
    # Если есть input — добавляем к инструкции через перенос
    inp = example.get("input", "").strip()
    if inp:
        instruction = f"{instruction}\n{inp}"
    response = example.get("output", "")

    # Длина
    if not (MIN_INSTRUCTION_LEN <= len(instruction) <= MAX_INSTRUCTION_LEN):
        return False
    if not (MIN_RESPONSE_LEN <= len(response) <= MAX_RESPONSE_LEN):
        return False

    # Нет специальных символов-мусора
    if re.search(r"[<>{}]", instruction):
        return False

    # Не пустая инструкция/ответ
    if not instruction.strip() or not response.strip():
        return False

    # Нет ссылок (добавляют шум)
    if "http://" in response or "https://" in response:
        return False

    return True

# Синтез нового стиля ответов
# Целевой стиль: краткий, структурированный, с bullet-points, без воды
STYLE_TEMPLATES = [
    # Шаблон 1: Прямой ответ + детали
    lambda inst, resp: {
        "instruction": inst,
        "response": _make_concise(resp)
    },
    # Шаблон 2: Структурированный список
    lambda inst, resp: {
        "instruction": inst,
        "response": _make_structured(resp)
    },
]

def _make_concise(text):
    """Убираем вводные слова, делаем ответ прямым."""
    # Удаляем вводные фразы
    fillers = [
        r"^Sure,?\s*", r"^Of course,?\s*", r"^Certainly,?\s*",
        r"^Great question[!.]?\s*", r"^I'm happy to help[!.]?\s*",
        r"^Absolutely[!.]?\s*", r"^Yes[,!.]?\s*",
    ]
    result = text
    for filler in fillers:
        result = re.sub(filler, "", result, flags=re.IGNORECASE)

    # Обрезаем до 2-3 предложений если длинный
    sentences = result.split(". ")
    if len(sentences) > 4:
        result = ". ".join(sentences[:3]) + "."

    return result.strip()

def _make_structured(text):
    """Преобразуем в структурированный формат с ключевыми моментами."""
    clean = _make_concise(text)

    # Если есть перечисления — структурируем
    if any(w in clean.lower() for w in [" first", " second", " third", " also", " additionally"]):
        lines = re.split(r"\.\s+", clean)
        if len(lines) >= 2:
            header = lines[0] + "."
            bullets = "\n".join(f"• {l.strip()}." for l in lines[1:] if l.strip())
            return f"{header}\n\n{bullets}"

    return clean


# Синтетические примеры (технический стиль)
SYNTHETIC_PAIRS = [
    {
        "instruction": "Что такое градиентный спуск?",
        "response": "Градиентный спуск — алгоритм оптимизации, который итеративно движется в направлении наибольшего убывания функции потерь.\n\n• Вычисляем градиент ∇L по параметрам\n• Обновляем: θ = θ - α·∇L\n• α — learning rate, контролирует размер шага\n\nОстановка: когда изменение потерь < порога или исчерпан лимит итераций."
    },
    {
        "instruction": "Объясни разницу между L1 и L2 регуляризацией.",
        "response": "Обе добавляют штраф к функции потерь, но по-разному.\n\nL1 (Lasso): штраф = λ·Σ|wᵢ| — порождает разреженность, обнуляет незначимые веса.\nL2 (Ridge): штраф = λ·Σwᵢ² — равномерно уменьшает все веса, не обнуляет.\n\nПрактика: L1 — когда нужен отбор признаков. L2 — когда все признаки потенциально значимы."
    },
    {
        "instruction": "Что такое переобучение и как с ним бороться?",
        "response": "Переобучение — модель запоминает обучающие данные вместо обобщения закономерностей.\n\nПризнаки: train loss ↓, val loss ↑.\n\nМетоды борьбы:\n• Dropout — случайное отключение нейронов при обучении\n• L1/L2 регуляризация — штраф за большие веса\n• Early stopping — остановка при росте val loss\n• Аугментация данных — увеличение разнообразия выборки\n• Уменьшение сложности модели"
    },
    {
        "instruction": "Что такое attention mechanism в трансформерах?",
        "response": "Attention позволяет модели взвешивать важность разных токенов при генерации каждого следующего.\n\nФормула: Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V\n\n• Q (Query) — что ищем\n• K (Key) — с чем сравниваем\n• V (Value) — что берём\n\nMulti-head attention запускает несколько голов параллельно — каждая улавливает разные типы зависимостей."
    },
    {
        "instruction": "Чем отличается val loss от train loss при нормальном обучении?",
        "response": "При корректном обучении: val loss немного выше train loss, оба убывают.\n\nОпасные паттерны:\n• val loss растёт, train loss падает → переобучение\n• оба не убывают → слишком большой lr или архитектурная ошибка\n• val loss < train loss → нетипично, проверьте разметку"
    },
    {
        "instruction": "Что такое QLoRA и зачем оно нужно?",
        "response": "QLoRA — техника дообучения больших моделей с минимальным потреблением памяти.\n\nКак работает:\n1. Базовая модель квантуется до 4-bit (NF4) — экономия памяти ×4\n2. LoRA-адаптеры (малые матрицы ΔW) обучаются в полной точности\n3. Только адаптеры сохраняются и переносятся\n\nРезультат: Mistral-7B обучается на T4 (16GB) вместо требуемых ~28GB."
    },
    {
        "instruction": "Что такое ROUGE метрика?",
        "response": "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — метрика качества генерации текста.\n\nОсновные варианты:\n• ROUGE-1: перекрытие унограмм (отдельных слов)\n• ROUGE-2: перекрытие биграмм (пар слов)\n• ROUGE-L: наибольшая общая подпоследовательность\n\nВычисляется как F1 между prediction и reference. Выше = лучше. Диапазон: 0–1."
    },
    {
        "instruction": "Как правильно разбить датасет на train/val/test?",
        "response": "Стандартные пропорции: 80/10/10 или 70/15/15.\n\nПравила:\n• Разбивать до любой предобработки (иначе утечка данных)\n• Стратифицировать по классам для классификации\n• Test трогать только один раз — финальная оценка\n• Val — для подбора гиперпараметров и early stopping\n\nПри малом датасете (<1000 примеров): k-fold cross-validation."
    },
    {
        "instruction": "Что такое learning rate scheduler?",
        "response": "Scheduler изменяет learning rate по ходу обучения.\n\nПопулярные стратегии:\n• Cosine annealing: lr плавно убывает по косинусу\n• Linear warmup: lr растёт первые N шагов, затем убывает\n• Step decay: lr делится на γ каждые N эпох\n• ReduceLROnPlateau: снижает lr при стагнации val loss\n\nДля fine-tuning: warmup 5–10% шагов + cosine decay — стандартная практика."
    },
    {
        "instruction": "Что такое batch normalization?",
        "response": "Batch Norm нормализует активации каждого слоя по батчу: μ=0, σ=1.\n\nЗатем масштабирует: y = γ·x̂ + β (γ, β обучаемые).\n\nЭффекты:\n• Стабилизирует обучение — можно использовать больший lr\n• Частичный эффект регуляризации\n• Ускоряет сходимость\n\nАльтернативы: Layer Norm (в трансформерах), Group Norm (малые батчи)."
    },
    {
        "instruction": "Объясни разницу между precision и recall.",
        "response": "Precision: из всего, что модель назвала позитивным — сколько реально позитивное.\nPrecision = TP / (TP + FP)\n\nRecall: из всего реально позитивного — сколько модель нашла.\nRecall = TP / (TP + FN)\n\nКомпромисс: повышение порога → precision ↑, recall ↓.\n\nF1 = 2·P·R / (P+R) — гармоническое среднее, баланс двух метрик."
    },
    {
        "instruction": "Как работает токенизация в языковых моделях?",
        "response": "Токенизация разбивает текст на токены — единицы, с которыми работает модель.\n\nМетоды:\n• BPE (Byte-Pair Encoding): итеративно объединяет частые пары байт\n• WordPiece: аналогично, используется в BERT\n• SentencePiece: работает напрямую с Unicode\n\nMistral/LLaMA используют BPE. Словарь ~32K токенов.\nПравило: 1 токен ≈ 4 символа (для английского), ≈ 2–3 для русского."
    },
    {
        "instruction": "Что такое temperature при генерации текста?",
        "response": "Temperature контролирует «случайность» генерации.\n\nФормула: P'(x) = softmax(logits / T)\n\n• T → 0: детерминированный выбор (argmax)\n• T = 1: стандартное распределение\n• T > 1: более случайный, творческий вывод\n\nПрактика:\n• Для кода/фактов: T = 0.1–0.3\n• Для диалога: T = 0.7–0.9\n• Для творческих задач: T = 1.0–1.2"
    },
    {
        "instruction": "Что такое LoRA и как устроены адаптеры?",
        "response": "LoRA (Low-Rank Adaptation) добавляет к замороженным весам модели малоранговое обновление.\n\nИдея: вместо ΔW (огромная матрица) учим ΔW = A·B, где A и B малы.\n• W_new = W_frozen + α·A·B\n• rank r << d (обычно 4–64)\n• Параметров в 100–1000 раз меньше\n\nАдаптер сохраняется отдельно (~10–100 MB) и применяется поверх базовой модели."
    },
    {
        "instruction": "Объясни принцип early stopping.",
        "response": "Early stopping останавливает обучение, когда val loss перестаёт улучшаться.\n\nАлгоритм:\n1. Следим за val loss каждые N шагов\n2. Если нет улучшения за patience эпох — останавливаемся\n3. Восстанавливаем веса из лучшей чекпоинты\n\nПараметры:\n• patience: сколько эпох ждём (обычно 3–10)\n• min_delta: минимальное значимое улучшение\n\nЗащищает от переобучения без ручного контроля."
    },
    {
        "instruction": "Что такое BERTScore?",
        "response": "BERTScore — метрика качества текста на основе эмбеддингов BERT.\n\nПринцип:\n1. Кодируем prediction и reference через BERT\n2. Считаем косинусное сходство между токенами\n3. Вычисляем Precision, Recall, F1 на уровне токен-пар\n\nПреимущество над ROUGE: понимает семантику, а не только точное совпадение слов.\n«Быстрый» ≈ «скорый» → BERTScore высокий, ROUGE-1 низкий."
    },
    {
        "instruction": "Как работает dropout?",
        "response": "Dropout случайно обнуляет p% нейронов при каждом forward pass во время обучения.\n\nЭффект:\n• Предотвращает co-adaptation (нейроны учатся независимо)\n• Эквивалентен ансамблю многих архитектур\n• Стандартные значения: p = 0.1–0.5\n\nВажно: при инференсе dropout отключается (model.eval()), веса масштабируются на (1-p)."
    },
    {
        "instruction": "Что такое cross-entropy loss?",
        "response": "Cross-entropy — стандартная функция потерь для задач классификации.\n\nФормула: L = -Σ yᵢ · log(p̂ᵢ)\n• yᵢ — истинная метка (0 или 1)\n• p̂ᵢ — предсказанная вероятность\n\nДля языковых моделей (causal LM): усредняем по всем токенам.\nPerplexity = exp(L) — интерпретируемая версия: сколько токенов модель «путает»."
    },
    {
        "instruction": "Чем отличается fine-tuning от обучения с нуля?",
        "response": "Обучение с нуля: случайные веса → обучение на большом датасете → недели/месяцы на кластере.\n\nFine-tuning: предобученные веса → дообучение на малом датасете → часы на одной GPU.\n\nКлючевые отличия:\n• Данных нужно в 100–1000 раз меньше\n• Меньший lr (1e-5 vs 1e-3)\n• Риск catastrophic forgetting — модель «забывает» прежние знания\n\nLoRA/QLoRA решают проблему забывания: базовые веса заморожены."
    },
    {
        "instruction": "Что такое perplexity языковой модели?",
        "response": "Perplexity (PP) — мера «удивлённости» модели при виде текста.\n\nФормула: PP = exp(cross-entropy loss)\n\nИнтерпретация: модель в среднем выбирает из PP одинаково вероятных токенов.\n• PP = 1: идеальная модель (знает следующий токен)\n• PP = 100: неплохо для языковой модели\n• PP = 1000+: слабая модель или неподходящий домен\n\nMistral-7B на WikiText-2: ~5–6."
    },
    {
        "instruction": "Как правильно выбрать learning rate для fine-tuning?",
        "response": "Для LoRA fine-tuning стандартные значения: 1e-4 – 3e-4.\n\nПодход:\n1. Начните с 2e-4 — работает в большинстве случаев\n2. Используйте warmup (5–10% шагов)\n3. Смотрите на train loss: должен стабильно убывать\n4. Если loss взрывается — lr слишком велик (делите на 10)\n5. Если loss не двигается — lr слишком мал (умножайте на 3)\n\nlr finder (fastai) автоматизирует этот процесс."
    },
    {
        "instruction": "Что такое инструктивная настройка (instruction tuning)?",
        "response": "Instruction tuning — дообучение LLM на парах (инструкция, ответ) для следования командам пользователя.\n\nБаз. модель: продолжает текст (next-token prediction).\nПосле instruction tuning: отвечает на вопросы, выполняет задачи.\n\nФорматы:\n• Alpaca: ### Instruction / ### Response\n• ChatML: <|im_start|>user ... <|im_start|>assistant\n• Mistral: [INST] ... [/INST]\n\nДатасеты: Alpaca (52K), Dolly (15K), FLAN (1M+)."
    },
    {
        "instruction": "Что такое gradient checkpointing?",
        "response": "Gradient checkpointing — техника экономии GPU памяти при обучении.\n\nОбычно: все промежуточные активации хранятся для backprop → много памяти.\n\nС checkpointing:\n• Часть активаций не хранится\n• При backprop пересчитываются заново (forward pass)\n• Память: ×√N меньше, скорость: ×~20% медленнее\n\nВключение: model.gradient_checkpointing_enable()\nОбязательно для обучения 7B+ моделей на T4."
    },
    {
        "instruction": "Как интерпретировать confusion matrix?",
        "response": "Confusion matrix — таблица N×N для N классов.\n\nДля бинарной классификации:\n         Pred 0  Pred 1\nTrue 0 [  TN  |  FP  ]\nTrue 1 [  FN  |  TP  ]\n\nМетрики:\n• Accuracy = (TP+TN) / всего\n• Precision = TP / (TP+FP) — качество позитивных предсказаний\n• Recall = TP / (TP+FN) — покрытие позитивного класса\n• F1 = 2PR/(P+R)\n\nПри несбалансированных классах Accuracy вводит в заблуждение — смотрите F1."
    },
    {
        "instruction": "Что такое weight decay в оптимизаторах?",
        "response": "Weight decay — L2 регуляризация, встроенная в оптимизатор.\n\nФормула обновления: w ← w·(1 - lr·λ) - lr·∇L\n\nЭффект: веса уменьшаются на каждом шаге пропорционально своей величине.\n\nAdamW (Adam + weight decay): стандарт для LLM.\nТипичные значения: 0.01–0.1\n\nОтличие от L2 в loss: в AdamW weight decay не адаптируется моментами → корректнее."
    },
    {
        "instruction": "Что такое PEFT и какие методы в него входят?",
        "response": "PEFT (Parameter-Efficient Fine-Tuning) — семейство методов дообучения, изменяющих малую долю параметров.\n\nОсновные методы:\n• LoRA: низкоранговые матрицы обновлений\n• Prefix Tuning: обучаемые prefix-токены в каждом слое\n• Prompt Tuning: обучаемые soft-токены только на входе\n• IA³: масштабирует активации тремя векторами\n\nLoRA — самый популярный: лучший баланс качество/память.\nБиблиотека: huggingface/peft."
    },
]

# Дополнительные общие пары (не только ML)
GENERAL_SYNTHETIC = [
    {
        "instruction": "Что такое рекурсия в программировании?",
        "response": "Рекурсия — вызов функцией самой себя для решения подзадач.\n\nКомпоненты:\n• Base case: условие остановки (без него — бесконечный цикл)\n• Recursive case: вызов себя с меньшей задачей\n\nПример: factorial(n) = n · factorial(n-1), factorial(0) = 1\n\nСложность: стек вызовов растёт O(n). При n > ~1000 → Stack Overflow.\nАльтернатива: хвостовая рекурсия или итерация."
    },
    {
        "instruction": "Объясни разницу между TCP и UDP.",
        "response": "TCP: надёжный, с подтверждением доставки.\n• Устанавливает соединение (3-way handshake)\n• Гарантирует порядок и доставку пакетов\n• Медленнее из-за подтверждений\n• Использование: HTTP, SSH, БД\n\nUDP: быстрый, без гарантий.\n• Нет соединения, нет подтверждений\n• Пакеты могут теряться или приходить не по порядку\n• Использование: видео/аудио стриминг, DNS, игры"
    },
    {
        "instruction": "Что такое индекс в базе данных?",
        "response": "Индекс — структура данных для ускорения поиска в таблице.\n\nАналогия: оглавление книги вместо чтения всех страниц.\n\nБез индекса: Full Table Scan, O(n).\nС индексом (B-Tree): O(log n).\n\nКогда добавлять:\n• Колонки в WHERE, JOIN, ORDER BY\n• Высокая кардинальность (много уникальных значений)\n\nКогда не нужен:\n• Малые таблицы (<1000 строк)\n• Частые INSERT/UPDATE (индекс замедляет запись)"
    },
    {
        "instruction": "Что такое Docker и зачем он нужен?",
        "response": "Docker — инструмент контейнеризации: упаковывает приложение со всеми зависимостями.\n\nКонтейнер vs VM:\n• VM: полноценная ОС, тяжёлый (~GBs), медленный старт\n• Контейнер: разделяет ядро хоста, лёгкий (~MBs), старт за секунды\n\nОсновные понятия:\n• Image: шаблон (Dockerfile → docker build)\n• Container: запущенный экземпляр image\n• Registry: хранилище образов (Docker Hub)\n\nПрактика: воспроизводимая среда на любом хосте."
    },
    {
        "instruction": "Что такое REST API?",
        "response": "REST — архитектурный стиль для API поверх HTTP.\n\nПринципы:\n• Stateless: сервер не хранит состояние клиента\n• Ресурсы: URL идентифицирует сущность (/users/42)\n• HTTP-методы: GET (чтение), POST (создание), PUT/PATCH (обновление), DELETE\n\nПример:\nGET  /users      → список пользователей\nPOST /users      → создать пользователя\nGET  /users/42   → пользователь с id=42\nDELETE /users/42 → удалить\n\nОтвет в JSON, статус-коды: 200, 201, 404, 500."
    },
]

# Удаление дубликатов
def dedup(examples):
    seen = set()
    unique = []
    for ex in examples:
        key = hashlib.md5(ex["instruction"].lower().strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


# Сборка финального датасета
def build_dataset():
    ds = load_base_dataset()

    print("🔍 Фильтрация качества...")
    filtered = [ex for ex in tqdm(ds) if quality_filter(ex)]
    print(f"   После фильтрации: {len(filtered)} примеров")

    # Преобразуем в целевой формат и применяем стилизацию
    print("✏️  Стилизация ответов...")
    styled = []
    for ex in filtered:
        # Alpaca: поля instruction, input (доп. контекст), output
        inp = ex.get("input", "").strip()
        inst = ex["instruction"].strip()
        if inp:
            inst = f"{inst}\n{inp}"
        resp = ex["output"].strip()
        template = random.choice(STYLE_TEMPLATES)
        result = template(inst, resp)
        styled.append({
            "instruction": result["instruction"],
            "response": result["response"]
        })

    # Добавляем синтетические примеры
    all_examples = styled + SYNTHETIC_PAIRS + GENERAL_SYNTHETIC
    print(f"   + {len(SYNTHETIC_PAIRS + GENERAL_SYNTHETIC)} синтетических примеров")

    # Дедупликация
    print("🔁 Удаление дубликатов...")
    unique = dedup(all_examples)
    print(f"   После дедупликации: {len(unique)} примеров")

    # Перемешиваем и берём нужное количество
    random.seed(42)
    random.shuffle(unique)

    # Берём 220 примеров (с запасом)
    final = unique[:220]
    print(f"   Финальный датасет: {len(final)} примеров")

    return final


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"💾 Сохранено: {path}")


def validate_jsonl(path):
    """Проверяем корректность JSONL файла."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert "instruction" in obj, "Отсутствует поле instruction"
            assert "response" in obj, "Отсутствует поле response"
            assert len(obj["instruction"]) >= MIN_INSTRUCTION_LEN
            assert len(obj["response"]) >= MIN_RESPONSE_LEN
            count += 1
    print(f"✅ Валидация пройдена: {count} примеров, все поля корректны")


if __name__ == "__main__":
    print("=" * 60)
    print("ШАГ 1: Подготовка датасета")
    print("=" * 60)

    dataset = build_dataset()

    # Сохраняем полный датасет
    save_jsonl(dataset, "dataset.jsonl")

    # Train/val split: 200 train, 20 val
    train_data = dataset[:200]
    val_data = dataset[200:]

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")

    # Валидация
    print("\n🔎 Валидация файлов...")
    validate_jsonl("dataset.jsonl")

    # Статистика
    instructions_len = [len(ex["instruction"]) for ex in dataset]
    responses_len = [len(ex["response"]) for ex in dataset]

    print(f"\n📊 Статистика датасета:")
    print(f"   Всего примеров: {len(dataset)}")
    print(f"   Инструкции — avg: {sum(instructions_len)//len(instructions_len)} символов")
    print(f"   Ответы      — avg: {sum(responses_len)//len(responses_len)} символов")
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    print("\n✅ Шаг 1 завершён.")