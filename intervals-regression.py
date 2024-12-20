import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Функція для побудови графіків залежності та лінійної регресії
def analyze_excel(file_path):
    # Завантаження Excel-файлу
    df = pd.read_excel(file_path)
    df = df.drop('Файл', axis=1)

    # Перевірка наявності потрібних стовпців
    required_columns = ['LOC', 'NOM', 'SIZE2', 'WAC', 'NOCC', 'DIT',
                        'WPMC1', 'WMPC2', 'RFC', 'CBO', 'MPC', 'LCOM']

    # Функція для розділення пар чисел
    def split_pairs(value):
        if isinstance(value, str) and '-' in value:
            value = value.replace(',', '.')
            first, second = map(float, value.split(' - '))
            return first, second
        else:
            return None, None

    # Розділення пар чисел у всіх стовпцях, окрім 'К-ть класів'
    for col in required_columns:
        df[f'{col}_min'] = df[col].apply(lambda x: split_pairs(x)[0])
        df[f'{col}_max'] = df[col].apply(lambda x: split_pairs(x)[1])

    # Ініціалізація поліноміальної регресії (2-го порядку)
    poly = PolynomialFeatures(degree=1)
    lin_reg = LinearRegression()

    # Визначення кількості підграфіків
    num_cols = 6 #2
    num_rows = 4 #(num_plots + num_cols - 1) // num_cols

    # Створення фігури для всіх графіків
    fig, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()

    # Цикл по стовпцях для перших і других чисел
    plot_idx = 0
    for col in required_columns:
        for suffix in ['min', 'max']:
            new_col = f'{col}_{suffix}'
            if new_col in df.columns and df[new_col].notnull().any():
                # Підготовка даних
                X = df[['К-ть класів']]
                y = df[new_col].values
                print(y)

                # Трансформація даних для поліноміальної регресії
                X_poly = poly.fit_transform(X)

                # Навчання моделі
                lin_reg.fit(X_poly, y)

                # Прогнозування
                y_pred = lin_reg.predict(X_poly)

                # Побудова графіка
                ax = axes[plot_idx]
                sns.scatterplot(x=df['К-ть класів'], y=df[new_col], ax=ax)
                ax.plot(df['К-ть класів'], y_pred, color='red')
                ax.set_title(f"Залежність {new_col} \nвід К-ть класів", fontsize=10)
                ax.set_xlabel(' ', fontsize=10)
                ax.set_ylabel(new_col, fontsize=10)
                ax.grid(True)

                plot_idx += 1

    # Видалення порожніх підграфіків
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    # Відображення всіх графіків
    #fig.tight_layout()
    plt.subplots_adjust(
                        wspace=0.4, hspace=0.4)
    plt.savefig("all_plots.png")
    plt.show()

# Використання функції
analyze_excel("confidence-intervals.xlsx")