import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\bruno\OneDrive\Faculdade\F259 - Experimental II\Experimento 3 - Tubos\Ressoador\Ressoador.csv.csv")

freq_quadrada = data["Frequencia encontrada (usar essa)"]**2. #Eixo X
freq_quadrada = freq_quadrada.dropna()
freq_quadrada = freq_quadrada.drop(freq_quadrada.index[-1])
inverso_volume = (1/(data["volume"].str.replace(",",".").apply(lambda x: float(x)))).dropna()
inverso_volume = inverso_volume.drop(inverso_volume.index[-1])
print(freq_quadrada, "\n", inverso_volume)

#Matplotlib 
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(inverso_volume,freq_quadrada,"o", label = "Experimental Data", color = "black")
plt.xlabel("1/V (1/m³)")
plt.ylabel("Frequencias ao quadrado (Hz²)")
plt.title("Linearização de Frequencia ao quadrado por inverso do Volume (Ressoador)") 


#sklearn - Gera a Regressão Linear associada aos dados de aceleração de Diferença de massa obtidos
x = pd.DataFrame(inverso_volume)
y = freq_quadrada
model = LinearRegression()
model.fit(x,y)
x_range = np.linspace(x.min(),x.max(),100).reshape(-1,1)
y_pred = model.predict(x_range)

#Adiciona linearização ao gráfico
plt.plot(x_range,y_pred,label = "Linear Fit", color= "blue")
plt.legend(loc = "upper left", fontsize = "small", shadow = True)

#Cálcula as incertezas associadas a cada parâmetro, retornadas na variável Incertezas.
y_test_pred = model.predict(x)
residuals = y - y_test_pred
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

print(f"Freq² = ({a:.3f}±{incerteza[0]})*Inv_Volume")

plt.show()

