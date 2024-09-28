import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Pandas
data = pd.read_csv(r"C:\Users\bruno\OneDrive\Faculdade\F259 - Experimental II\Experimento 3 - Tubos\aberto_aberto\aberto_aberto_data.csv")

data["frequency (Hz)"] = data["frequency (Hz)"].apply(lambda x: int(x)) #Eixo X
data["Amplitude Normalizada"] = data["Amplitude Normalizada"].str.replace(",",".").apply(lambda x: float(x)) #Eixo Y_1
data["Nivel Sonoro (db)"] = data["Nivel Sonoro (db)"].str.replace(",",".").apply(lambda x: float(x)) #Eixo Y_2

#Matplotlib
fig, ax1 = plt.subplots(figsize=(20, 6))
ax1.plot(data["frequency (Hz)"],data["Amplitude Normalizada"],"o", label = "Amplitude Normalizada (u.m)", color = "red")
ax1.plot(data["frequency (Hz)"],data["Amplitude Normalizada"],linestyle = 'dashed', color = "red",alpha = 0.3)
ax2 = ax1.twinx()
ax2.plot(data["frequency (Hz)"],data["Nivel Sonoro (db)"],linestyle = 'dashed', color = "green",alpha = 0.3)
ax2.plot(data["frequency (Hz)"],data["Nivel Sonoro (db)"],"o", label = "Nivel Sonoro(db)", color = "green")
ax1.set_xlabel("Frequencia (Hz)")
ax1.set_ylabel("Amplitude")
ax2.set_ylabel("Nivel Sonoro (db)")
plt.title("Gráfico de frequência para o tubo aberto-aberto") 

# Obter handles e labels de ambos os eixos
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combinar handles e labels
handles = handles1 + handles2
labels = labels1 + labels2

# Adicionar a legenda combinada
ax1.legend(handles, labels, loc="upper right", fontsize="small", shadow=True)

plt.show()
