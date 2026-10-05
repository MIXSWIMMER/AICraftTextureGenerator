import pandas as pd

# Укажите путь к вашему parquet-файлу
df = pd.read_parquet("data/train.parquet")

print("--- Названия всех колонок ---")
print(df.columns.tolist())

print("\n--- Типы данных колонок ---")
print(df.dtypes)

print("\n--- Пример первой строки ---")
print(df.head(1))