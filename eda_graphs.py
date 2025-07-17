import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


try:
    data = pd.read_csv("vgsales.csv")
except Exception as e:
    print("Ошибка загрузки файла: vgsales.csv")
    print("Подробности:", e)
    exit()

# === 1. Глобальные продажи по годам ===
plt.figure(figsize=(12, 6))
data.groupby('Year')['Global_Sales'].sum().plot(kind='bar', colormap='viridis')
plt.title('Глобальные продажи по годам')
plt.xlabel('Год')
plt.ylabel('Продажи (млн копий)')
plt.tight_layout()
plt.savefig("graph.png")
plt.show()

# === 2. Корреляционная матрица продаж ===
plt.figure(figsize=(8, 6))
corr = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].corr()
sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f")
plt.title('Корреляционная матрица продаж')
plt.tight_layout()
plt.savefig("graph.png")
plt.show()

# === 3. Распределение по платформам, годам, жанрам ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


data['Platform'].value_counts().head(10).plot(kind='bar', ax=axes[0])
axes[0].set_title('Топ 10 платформ по количеству игр')
axes[0].set_ylabel('Количество')


data['Year'].value_counts().sort_index().plot(kind='bar', ax=axes[1])
axes[1].set_title('Распределение игр по годам')
axes[1].set_ylabel('Количество')


data['Genre'].value_counts().plot(kind='bar', ax=axes[2])
axes[2].set_title('Распределение по жанрам')
axes[2].set_ylabel('Количество')

plt.tight_layout()
plt.savefig("graph.png")
plt.show()

# === 4. Продажи по жанру Action по годам ===
plt.figure(figsize=(12,6))
action_sales = data[data['Genre'] == 'Action'].groupby('Year')['Global_Sales'].sum()
action_sales.plot(kind='line', marker='o')
plt.title('Продажи игр жанра Action по годам')
plt.xlabel('Год')
plt.ylabel('Продажи (млн копий)')
plt.grid(True)
plt.tight_layout()
plt.savefig("graph.png")
plt.show()
