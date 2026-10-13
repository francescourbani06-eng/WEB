import pandas as pd
import matplotlib.pyplot as plt

# Creo il DataFrame
reviews = pd.DataFrame({
    "country": ["Italy", "France", "Italy", "Spain", "France",
                "USA", "Italy", "USA", "Spain", "France"],
    "points": [88, 92, 85, 90, 94, 87, 91, 89, 86, 93],
    "price": [15, 35, 12, 20, 45, 18, 27, 22, 14, 50]
})

# -----------------------------
# 1. PANDAS: preparo i dati
# -----------------------------

# Tengo solo vini con almeno 88 punti
good_reviews = reviews.loc[reviews["points"] >= 88]

# Calcolo il prezzo medio per Paese
mean_price = reviews.groupby("country")["price"].mean()

print(mean_price)

# -----------------------------
# 2. MATPLOTLIB: grafico a barre
# -----------------------------

plt.bar(mean_price.index, mean_price.values)

plt.title("Prezzo medio dei vini per Paese")
plt.xlabel("Paese")
plt.ylabel("Prezzo medio (€)")

plt.show()

# -----------------------------
# 3. Scatter plot
# -----------------------------

plt.scatter(reviews["price"], reviews["points"])

plt.title("Relazione tra prezzo e punteggio")
plt.xlabel("Prezzo (€)")
plt.ylabel("Punteggio")

plt.show()

# -----------------------------
# 4. Istogramma
# -----------------------------

plt.hist(reviews["points"], bins=5)

plt.title("Distribuzione dei punteggi")
plt.xlabel("Punteggio")
plt.ylabel("Frequenza")

plt.show()