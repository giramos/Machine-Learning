import pandas as pd
import numpy as np

df = pd.read_csv("datasets/datos01_train.csv")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = {0: "#4A249D", 1: "#0D7C66"}

def visualizar_datos(df_datos: pd.DataFrame, df_labels: pd.DataFrame, title: str, x_label: str, y_label: str, legend_dict: dict = None, new_point: np.ndarray = None) -> tuple:
    if isinstance(df_datos, pd.DataFrame):
        df_datos = df_datos.to_numpy()

    if isinstance(df_labels, pd.DataFrame):
        df_labels = df_labels.to_numpy()

    fig, ax = plt.subplots(figsize=(5,5))
    
    # Visualizar los puntos de entrenamiento
    ax.scatter(df_datos[:, 0], df_datos[:, 1], c=[colors[c] for c in df_labels],
        s=100,
        edgecolor="k",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], color="red", marker="*", s=200, label="Nueva persona")

    if legend_dict is not None:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0], markersize=10, label=legend_dict[0]),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], markersize=10, label=legend_dict[1]),
        ]

        if new_point is not None:
            legend_elements.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=10, label="Nuevo punto"))

        plt.legend(handles=legend_elements)
    
    plt.show()

    return ax.get_xlim(), ax.get_ylim()

x_range, y_range =  visualizar_datos(df.iloc[:,:-1], df.iloc[:,-1], "datos01_train", "Variable1", "Variable2", {0: "Clase 0", 1: "Clase 1"})