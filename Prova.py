import matplotlib.pyplot as plt
import pandas as pd


reviews = pd.DataFrame({
    "country": ["Italy", "France", "Italy", "Spain", "France", "USA", "Italy", "USA", "Spain", "France"],
    "points": [88, 92, 85, 90, 94, 87, 91, 89, 86, 93],
    "price": [15, 35, 12, 20, 45, 18, 27, 22, 14, 50]
})


dati = reviews["points"].head(20)

plt.plot(dati)

plt.title("Punteggi dei primi 20 vini")
plt.xlabel("Indice del vino")
plt.ylabel("Punteggio")

plt.show()