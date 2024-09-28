import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\bruno\OneDrive\Faculdade\F259 - Experimental II\Experimento 3 - Tubos\aberto_fechado\aberto_fechado_data.csv")

data["frequency (Hz)"] = data["frequency (Hz)"].apply(lambda x: int(x)) #Eixo X
data["Amplitude Normalizada"] = data["Amplitude Normalizada"].str.replace(",",".").apply(lambda x: float(x)) #Eixo Y_1
data["Nivel Sonoro (db)"] = data["Nivel Sonoro (db)"].str.replace(",",".").apply(lambda x: float(x)) #Eixo Y_2

# Usar SciPy para encontrar os picos (máximos locais)
amplitudes = data["Amplitude Normalizada"].values
frequencies = data["frequency (Hz)"].values

# Encontra os picos (máximos locais)
peaks, _ = find_peaks(amplitudes)

# Exibir as frequências correspondentes aos picos
peak_frequencies = frequencies[peaks]
print(peak_frequencies)
#Pandas
num_harmonico = np.linspace(3,23,11)
print(num_harmonico)

#Matplotlib 
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(num_harmonico,peak_frequencies,"o", label = "Experimental Data", color = "black")
plt.xlabel("Número do Harmônico (η)")
plt.ylabel("Frequências (Hz)")
plt.title("Linearização de Frequencia por Número de Harmônicos para o tubo Aberto-Fechado") 
plt.xticks(np.linspace(3,23,11))


#sklearn - Gera a Regressão Linear associada aos dados de aceleração de Diferença de massa obtidos
x = pd.DataFrame(num_harmonico)
y = peak_frequencies
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 1)
model = LinearRegression()
model.fit(x_train,y_train)
x_range = np.linspace(x.min(),x.max(),10).reshape(-1,1)
y_pred = model.predict(x_range)

#Adiciona linearização ao gráfico
plt.plot(x_range,y_pred,label = "Linear Fit", color= "blue")
plt.legend(loc = "upper left", fontsize = "small", shadow = True)

#Cálcula as incertezas associadas a cada parâmetro, retornadas na variável Incertezas.
y_test_pred = model.predict(x_test)
residuals = y_test - y_test_pred
n = len(y)
p = x.shape[1] + 1
residual_variance = np.sum(residuals**2) / (n - p)
x_design = np.hstack([np.ones((x.shape[0], 1)), x])
cov_matrix = residual_variance * np.linalg.inv(x_design.T @ x_design)
incerteza = np.sqrt(np.diag(cov_matrix))

a = model.coef_[0]
b = model.intercept_

incerteza[0] = round(incerteza[0],5)
incerteza[1] = round(incerteza[1],5)

print(f"Freq = ({a:.3f}±{incerteza[0]})*Num_Harmonico")

plt.show()

