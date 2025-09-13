# --- Импорт необходимых библиотек ---
import logging  # Для настройки логирования
import random   # Для установки random seed
from pathlib import Path  # Для удобной работы с путями к файлам
from typing import Iterator, Dict  # Для аннотации типов (подсказок для IDE и анализаторов кода)
import json     # Для чтения JSON Lines файлов

import typer    # Для создания красивого и удобного интерфейса командной строки (CLI)
import numpy as np  # Фундаментальная библиотека для числовых операций
import torch    # Основная библиотека для глубокого обучения (PyTorch)
import transformers  # Библиотека от Hugging Face для работы с Трансформерами
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments  # Ключевые классы для загрузки модели и обучения
from torch.utils.data import IterableDataset  # Класс для создания "потоковых" датасетов, которые не загружают все данные в память

from gluonts.transform import InstanceSplitter, ExpectedNumInstanceSampler  # Инструменты из GluonTS для нарезки временных рядов

from chronos import ChronosConfig, ChronosTokenizer  # Специальные классы из библиотеки Chronos для конфигурации и токенизации

# Создаем приложение командной строки
app = typer.Typer(pretty_exceptions_enable=False)


# --- Класс для подготовки данных к обучению ---
# IterableDataset - это специальный тип датасета PyTorch, который работает как "поток".
# Он генерирует данные "на лету", а не хранит их все в памяти, что идеально для больших файлов.
class ChronosUnivariateDataset(IterableDataset):
    """
    Этот класс читает файл с временными рядами, нарезает каждый ряд на обучающие примеры
    (контекст + будущее) и преобразует их в формат, понятный модели Chronos (токены).
    """
    def __init__(self, dataset_path: Path, tokenizer, context_length, prediction_length):
        super().__init__()
        # Сохраняем переданные параметры как атрибуты класса
        self.dataset_path = dataset_path      # Путь к .jsonl файлу с данными
        self.tokenizer = tokenizer            # Объект токенизатора Chronos
        self.context_length = context_length  # Длина истории, которую видит модель
        self.prediction_length = prediction_length  # Длина прогноза, который модель учится делать

        # Инициализируем 'InstanceSplitter' из библиотеки GluonTS.
        # Это мощный инструмент, который будет нарезать наши временные ряды.
        self.instance_splitter = InstanceSplitter(
            target_field="target",              # Имя поля с целевой переменной в .jsonl файле
            is_pad_field="is_pad",              # Имя поля, куда будет записана информация о padding'е
            start_field="start",                # Имя поля с датой начала
            forecast_start_field="forecast_start", # Имя поля, куда будет записана дата начала прогноза
            
            # Сэмплер, который говорит, сколько примеров нужно нарезать из каждого ряда.
            # ExpectedNumInstanceSampler(num_instances=1.0) пытается нарезать в среднем по одному примеру.
            instance_sampler=ExpectedNumInstanceSampler(num_instances=1.0),
            
            past_length=self.context_length,    # Длина прошлого (контекста) для каждого примера
            future_length=self.prediction_length, # Длина будущего (прогноза) для каждого примера
        )

    # __iter__ - это главный метод IterableDataset. Он вызывается, когда DataLoader запрашивает данные.
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Этот генератор читает файл, обрабатывает каждый временной ряд и 'yield'-ит 
        готовые обучающие примеры один за другим.
        """
        # Открываем наш .jsonl файл для чтения
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            # Читаем файл построчно, чтобы не загружать весь в память
            for line in f:
                # Каждая строка - это JSON. Преобразуем ее в Python-словарь.
                raw_ts = json.loads(line)
                
                # Преобразуем список чисел в numpy-массив нужного типа.
                # Это необходимо для корректной работы GluonTS.
                raw_ts['target'] = np.asarray(raw_ts['target'], dtype=np.float32)
                
                # Подаем один временной ряд в наш 'InstanceSplitter'.
                # Он может нарезать его на несколько обучающих примеров.
                # iter([raw_ts]) - трюк, чтобы передать один элемент как итератор.
                training_examples = self.instance_splitter.apply(iter([raw_ts]), is_train=True)
                
                # Проходим по всем примерам, которые нарезал сплиттер
                for example in training_examples:
                    
                    # КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА:
                    # Убеждаемся, что "будущее" (future_target) имеет ровно ту длину,
                    # которую ожидает модель. Если оно короче (например, это самый конец
                    # временного ряда), мы пропускаем этот неполноценный пример.
                    if len(example["future_target"]) != self.prediction_length:
                        continue
                        
                    # --- Преобразование данных в тензоры и токены ---
                    
                    # 1. Контекст (прошлое)
                    # Преобразуем numpy-массив с историей в тензор PyTorch
                    context = torch.tensor(example["past_target"])
                    # Подаем тензор в токенизатор. Он масштабирует данные и превращает их в последовательность токенов (чисел).
                    # .unsqueeze(0) добавляет фиктивное измерение batch_size=1, т.к. токенизатор ожидает батч.
                    input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context.unsqueeze(0))

                    # 2. Labels (будущее, которое модель должна предсказать)
                    # Аналогично, превращаем будущее в тензор
                    labels_context = torch.tensor(example["future_target"])
                    # Токенизируем его, используя тот же `scale`, что был вычислен для контекста.
                    # Это гарантирует, что прошлое и будущее находятся в одном масштабе.
                    labels, _ = self.tokenizer.label_input_transform(labels_context.unsqueeze(0), scale)
                    
                    # 'yield' возвращает один готовый к обучению пример и "замораживает" выполнение функции,
                    # пока DataLoader не запросит следующий.
                    yield {
                        "input_ids": input_ids.squeeze(0),      # Входные токены для модели (убираем batch_size=1)
                        "attention_mask": attention_mask.squeeze(0), # Маска, показывающая, где реальные токены, а где padding
                        "labels": labels.squeeze(0),            # Токены, которые модель должна научиться генерировать
                    }


# --- Основная функция, которая запускает весь процесс ---
# Декоратор @app.command() делает эту функцию доступной из командной строки.
@app.command()
def main(
    # --- Параметры путей и данных ---
    training_data_dir: str = "./chronos_univariate_dataset/",
    model_id: str = "amazon/chronos-t5-large",  # ID модели на Hugging Face Hub
    output_dir: str = "./output_102/",          # Папка для сохранения результатов и чекпоинтов
    
    # --- Ключевые параметры модели и данных ---
    context_length: int = 94,                   # Длина истории, на которую смотрит модель
    prediction_length: int = 8,                 # Горизонт прогнозирования
    
    # --- Параметры процесса обучения ---
    max_steps: int = 20000,                     # Общее количество шагов обучения
    save_steps: int = 5000,                     # Как часто сохранять чекпоинт
    log_steps: int = 200,                       # Как часто выводить лог с 'loss'
    per_device_train_batch_size: int = 32,      # Количество примеров в одном батче
    learning_rate: float = 5e-5,                # Скорость обучения (важный гиперпараметр)
    seed: int = 42,                             # Зерно для генератора случайных чисел для воспроизводимости
):
    # Устанавливаем seed для всех библиотек, чтобы результаты были воспроизводимыми
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("--- Шаг 1: Загрузка модели и токенизатора ---")
    # Загружаем предобученную модель из Hugging Face Hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Создаем конфигурацию Chronos вручную.
    # Это необходимо, так как у ChronosConfig нет метода .from_pretrained().
    print("Создание конфигурации Chronos вручную...")
    config = ChronosConfig(
        # Здесь задаются все параметры, необходимые для работы токенизатора и пайплайна
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -20.0, "high_limit": 20.0},
        n_tokens=4096,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=context_length,
        prediction_length=prediction_length,
        n_special_tokens=2,
        # Параметры ниже используются при генерации прогнозов, но нужны для инициализации
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    # На основе конфигурации создаем объект токенизатора
    tokenizer = config.create_tokenizer()

    print("\n--- Шаг 2: Подготовка датасета ---")
    # Собираем полный путь к файлу с данными
    train_file_path = Path(training_data_dir) / "univariate_train.jsonl"

    # Создаем экземпляр нашего кастомного датасета
    train_dataset = ChronosUnivariateDataset(
        dataset_path=train_file_path,
        tokenizer=tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    
    # Создаем папку для вывода, если она не существует
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print("\n--- Шаг 3: Настройка и запуск обучения ---")
    # Создаем объект TrainingArguments, который содержит все настройки для Trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir_path),          # Куда сохранять модель
        per_device_train_batch_size=per_device_train_batch_size, # Размер батча
        learning_rate=learning_rate,              # Скорость обучения
        max_steps=max_steps,                      # Сколько всего шагов делать
        logging_steps=log_steps,                  # Частота логирования
        save_steps=save_steps,                    # Частота сохранения
        save_total_limit=2,                       # Хранить не более 2 последних чекпоинтов
        optim="adamw_torch",                      # Оптимизатор
        lr_scheduler_type="cosine",               # Планировщик скорости обучения (плавно снижает ее)
        warmup_ratio=0.05,                        # Доля шагов для "прогрева" (плавного повышения LR в начале)
        report_to=["tensorboard"],                # Куда отправлять логи (для визуализации)
        remove_unused_columns=False,              # Не удалять лишние столбцы (важно для кастомных датасетов)
    )

    # Создаем объект Trainer - главный "дирижер" процесса обучения
    trainer = Trainer(
        model=model,                        # Модель, которую обучаем
        args=training_args,                 # Настройки обучения
        train_dataset=train_dataset,        # Наш кастомный датасет
    )
    
    print("\n--- НАЧАЛО ОБУЧЕНИЯ (UNIVARIATE FINE-TUNING) ---")
    # Запускаем процесс обучения. Эта команда будет выполняться долго.
    trainer.train()

    print("\n--- Обучение завершено. Сохранение финальной модели. ---")
    # Сохраняем финальную версию модели в отдельную папку
    final_checkpoint_dir = output_dir_path / "checkpoint-final"
    trainer.save_model(str(final_checkpoint_dir))
    # Вся информация о токенизаторе уже сохранена внутри config.json модели,
    # поэтому отдельное сохранение не требуется.
    print(f"Финальная модель сохранена в {final_checkpoint_dir}")

# Эта конструкция гарантирует, что код ниже выполнится только тогда,
# когда скрипт запускается напрямую (а не импортируется как модуль).
if __name__ == "__main__":
    # Настраиваем базовый формат логирования
    logging.basicConfig(level=logging.INFO)
    # Запускаем наше typer-приложение
    app()