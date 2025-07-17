# 🎮 Video Game Sales Prediction

Проект по анализу и предсказанию региональных продаж видеоигр с помощью нейросети на PyTorch.
##📂 Содержимое
- `vgsales.csv` — датасет с Kaggle (жанр, платформа, год, продажи по регионам)
- `vgsales_game.py` — основной код: обучение нейросети
- `eda_graphs.py` — анализ данных и визуализация
- Модель: полносвязная нейросеть (128 → 64 → 32 → 4)
- Функции: ReLU, Dropout(0.3)
- Оптимизатор: Adam, LR=0.001
- Потери: MSE
- Эпохи: 100
- Предсказываются: NA, EU, JP, Other Sales


## Для запуска

1. Установить зависимости:

PyTorch
pandas
matplotlib
seaborn
scikit-learn
streamlit

2. Запустить
python vgsales_game.py

##Для визуализации данных

python eda_graphs.py
