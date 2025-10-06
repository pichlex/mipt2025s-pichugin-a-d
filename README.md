# Проект: Локальный автоуточнитель уголков для проективно искажённых штрих‑/QR‑кодов

Этот репозиторий сфокусирован на задаче локального автоуточнения четырёх углов прямоугольной области перед нормализацией/приведением перспективы. В нём:

- Реализован модуль автоуточнения углов и единым API.
- Добавлены скрипты для локального запуска, визуализаций и сравнения методов.
- Подготовлены конфиги и данные.

## Актуальный отчёт
- Актуальный отчёт можно найти здесь:

[ОТЧЁТ](https://github.com/pichlex/mipt2025s-pichugin-a-d/blob/main/report/report.pdf)

## Установка

- Требования: Python 3.10+.
- Установка зависимостей:

```
uv pip install -e .
```

- Скачивание датасета:
```
wget -O data.zip "$(curl -s 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.360.yandex.ru/d/cKDZp5q991bglQ' | jq -r '.href')"

```

## Структура

- `src/barcode_refiner/geometry.py` — геометрические утилиты: упорядочивание углов, гомографии, IoU, отрисовка quad.
- `src/barcode_refiner/refine.py` — автоуточнение углов: `refine_corners(image, pts, config)`.
- `src/barcode_refiner/viz.py` — визуализация quad и легенд.
- `src/barcode_refiner/eval.py` — метрики и помощники для оценки.
- `src/barcode_refiner/scripts/*` — CLI: `refine-image`, `evaluate-refiners`.
- `configs/*` — примеры конфигов для запуска.
- `data/*` — размеченные изображения.
- `results/*` — папка для результатов (создаётся скриптами).
- `report/*` - отчёт о проделанной работе

## Быстрый старт

1) Оценка метода с результатами и визуализациями:

```
make eval-detector
```

2) Локальное автоуточнение для одного изображения и исходной разметки:

```
refine-image --image images/qr.png --points "10,10 200,20 210,210 20,200" --method edge --out results/single
```
