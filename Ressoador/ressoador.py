import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
data = pd.read_csv(r"C:\Users\bruno\OneDrive\Faculdade\F259 - Experimental II\Experimento 3 - Tubos\Ressoador\Ressoador.csv.csv")

# Pré-processamento dos dados
freq_quadrada = data["Frequencia encontrada (usar essa)"]**2  # Eixo X (Frequência ao quadrado)
freq_quadrada = freq_quadrada.dropna()
freq_quadrada = freq_quadrada.drop(freq_quadrada.index[-1])  # Remove último elemento

inverso_volume = (1/(data["volume"].str.replace(",", ".").apply(lambda x: float(x)))).dropna()  # Eixo Y (1/Volume)
inverso_volume = inverso_volume.drop(inverso_volume.index[-1])  # Remove último elemento

# Verificação dos dados
print(freq_quadrada, "\n", inverso_volume)

# Plotar os dados experimentais
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(inverso_volume, freq_quadrada, "o", label="Dados Experimentais", color="black")
plt.xlabel("1/V (1/m³)")
plt.ylabel("Frequências ao quadrado (Hz²)")
plt.title("Linearização de Frequência ao Quadrado por Inverso do Volume (Ressoador)")

# Aplicar Regressão Linear
x = pd.DataFrame(inverso_volume)
y = freq_quadrada
model = LinearRegression()
model.fit(x, y)

# Gerar valores preditos para o ajuste linear
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)  # Gera um intervalo para o ajuste
y_pred = model.predict(x_range)

# Plotar o ajuste linear
plt.plot(x_range, y_pred, label="Ajuste Linear", color="blue")
plt.legend(loc="upper left", fontsize="small", shadow=True)

# Cálculo das incertezas
y_test_pred = model.predict(x)
residuals = y - y_test_pred
n = len(y)
p = x.shape[1] + 1
residual_variance = np.sum(residuals**2) / (n - p)
x_design = np.hstack([np.ones((x.shape[0], 1)), x])
cov_matrix = residual_variance * np.linalg.inv(x_design.T @ x_design)
incerteza = np.sqrt(np.diag(cov_matrix))

# Coeficientes da reta ajustada
a = model.coef_[0]
b = model.intercept_

# Exibir os coeficientes com incertezas
incerteza[0] = round(incerteza[0], 5)
incerteza[1] = round(incerteza[1], 5)
print(f"Freq² = ({a:.3f}±{incerteza[0]})*Inv_Volume + ({b:.3f}±{incerteza[1]})")

# Exibir o gráfico
plt.show()
