import numpy as np
import moocore
import matplotlib.pyplot as plt

x = np.array([[5, 1], [1, 5], [4, 2], [4, 4], [5, 1]])
reference_point = np.array([6, 6])

contributions = moocore.hv_contributions(x, ref=reference_point)
print("Contribuciones hipervolumen:", contributions)

# Graficar los puntos y el punto de referencia
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], c=contributions, cmap='viridis', s=100, label='Puntos')
plt.scatter(reference_point[0], reference_point[1], c='red', marker='*', s=200, label='Referencia')

# Etiquetas de contribución
for i, (pt, hv) in enumerate(zip(x, contributions)):
	plt.text(pt[0]+0.1, pt[1]+0.1, f"{hv:.2f}", fontsize=10)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contribución hipervolumen de cada punto')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
