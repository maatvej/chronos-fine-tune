# --- Импорт необходимых библиотек ---
import pandas as pd  # Для работы с табличными данными
import json          # Для работы с форматом JSON
from pathlib import Path  # Для удобной работы с путями
from tqdm.auto import tqdm  # Для создания прогресс-баров

# --- 1. КОНФИГУРАЦИЯ ---
# Блок с основными параметрами для легкой настройки.

# Путь к исходному файлу с полными данными.
INPUT_FILE = "20250628_dm_by_clusters_train_preprocessed.parquet"

# Папка, куда будут сохранены подготовленные для теста данные.
OUTPUT_DIR = Path("chronos_univariate_dataset")

# Имя столбца с целевой переменной.
TARGET = "sales_qty"

# ВАЖНО: Горизонт прогнозирования.
# ЗАЧЕМ ЭТО НУЖНО: Это количество "хвостовых" точек, которые мы отрежем от каждого
# временного ряда, чтобы использовать их как "экзамен" для модели.
# Это значение должно быть АБСОЛЮТНО ИДЕНТИЧНО тому `prediction_length`,
# с которым вы обучали вашу модель.
PREDICTION_LENGTH = 8

# ВАЖНО: Минимальная общая длина ряда для включения в тест.
# ЗАЧЕМ ЭТО НУЖНО: Чтобы можно было отрезать "хвост" длиной `PREDICTION_LENGTH` и
# при этом оставить в "истории" достаточно данных для генерации прогноза.
# Это значение должно быть равно `context_length + prediction_length` из вашего
# скрипта обучения, или, как минимум, `MIN_SERIES_LENGTH` из скрипта подготовки.
MIN_TOTAL_LENGTH = 102


# --- ОСНОВНОЙ КОД ---
print("--- Шаг 1: Загрузка данных ---")
# Безопасная загрузка Parquet файла.
try:
    df = pd.read_parquet(INPUT_FILE)
    print(f"DataFrame успешно загружен. Форма: {df.shape}")
except FileNotFoundError:
    print(f"Ошибка: Файл '{INPUT_FILE}' не найден.")
    exit()

print("\n--- Шаг 2: Нарезка данных на историю (context) и будущее (actuals) ---")
# Создаем папку для вывода, если она не существует.
OUTPUT_DIR.mkdir(exist_ok=True)

# Определяем пути для двух выходных файлов.
context_file = OUTPUT_DIR / "test_context.jsonl" # Сюда пойдут данные для генерации прогноза (без "хвоста").
actuals_file = OUTPUT_DIR / "test_actuals.jsonl" # Сюда пойдут полные ряды для последующей оценки.

# Мы должны тестировать модель на данных, которые имеют ту же структуру,
# что и обучающие данные. Так как мы обучали модель на отдельных ClipID,
# то и тестировать мы должны, отрезая "хвост" от каждого ClipID отдельно.
grouped = df.groupby(['product_id', 'cluster_id', 'clip_id'])
print(f"Найдено {len(grouped)} уникальных непрерывных рядов (ClipID) для обработки.")

# `with` позволяет безопасно открыть сразу два файла для записи.
with open(context_file, 'w') as f_context, open(actuals_file, 'w') as f_actuals:
    # Итерируемся по каждому непрерывному ряду.
    for name, group in tqdm(grouped, desc="Нарезка временных рядов"):
        # Сортируем на всякий случай, чтобы гарантировать правильный порядок.
        group = group.sort_values('time_idx')
        target = group[TARGET].tolist()

        # Пропускаем ряды, которые слишком коротки для нашего теста.
        if len(target) < MIN_TOTAL_LENGTH:
            continue

        # --- Разделение данных ---
        # `context_data` - это все данные, КРОМЕ последних `PREDICTION_LENGTH` точек.
        # Эти данные мы покажем модели, чтобы она сделала прогноз.
        context_data = target[:-PREDICTION_LENGTH]
        
        # Создаем уникальный ID для этого конкретного ряда (ClipID)
        product_id, cluster_id, clip_id = name
        item_id = f"{product_id}_{cluster_id}_{clip_id}"
        
        # 1. Запись в файл с контекстом (для модели)
        context_entry = {
            "start": 0,
            "target": context_data,
            "item_id": item_id
        }
        f_context.write(json.dumps(context_entry) + '\n')
        
        # 2. Запись в файл с полными данными (для оценки)
        # Здесь мы сохраняем ПОЛНЫЙ ряд `target`. Это нужно, чтобы на этапе
        # оценки мы могли сравнить прогноз модели с реальным "хвостом".
        actuals_entry = {
            "start": 0,
            "target": target,
            "item_id": item_id
        }
        f_actuals.write(json.dumps(actuals_entry) + '\n')

print(f"\nГотово! Данные для теста сохранены в папке: {OUTPUT_DIR}")