#   Esperimento sugli embedding: similarità del coseno, analogia re/regina
#   e proiezione 2D con PCA.


import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.decomposition import PCA
 
# 1. Carica i vettori pre-addestrati (400.000 parole, 50 dimensioni ciascuna)
model = api.load("glove-wiki-gigaword-50")
print("Dimensione di ogni vettore:", model.vector_size)
print("Vettore di 'king' (primi 8 valori):", model["king"][:8])
 
# 2. Similarità del coseno, calcolata a mano come in numpy
def cos_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
 
print("\nSimilarità del coseno")
for w1, w2 in [("cat", "dog"), ("cat", "refrigerator"), ("king", "queen")]:
    print(f"  {w1:>6} - {w2:<13} {cos_sim(model[w1], model[w2]):.3f}")
 
# 3. L'analogia: king - man + woman ≈ queen
print("\nking - man + woman ≈")
for word, score in model.most_similar(positive=["king", "woman"], negative=["man"], topn=3):
    print(f"  {word:<10} {score:.3f}")
 
# 4. Proiezione 2D con PCA di parole appartenenti a gruppi diversi
gruppi = {
    "animali":  ["cat", "dog", "horse", "lion", "bird", "fish"],
    "paesi":    ["italy", "france", "germany", "spain", "japan", "china"],
    "capitali": ["rome", "paris", "berlin", "madrid", "tokyo", "beijing"],
    "oggetti":  ["table", "chair", "car", "phone", "computer", "book"],
}
colori = {"animali": "tab:green", "paesi": "tab:blue",
          "capitali": "tab:red", "oggetti": "tab:orange"}
 
parole = [w for lista in gruppi.values() for w in lista]
X = np.array([model[w] for w in parole])          # matrice (n_parole, 50)
 
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)                          # matrice (n_parole, 2)
print("\nVarianza spiegata dalle 2 componenti:", pca.explained_variance_ratio_.sum())
 
# Equivalente "alla MATLAB": SVD della matrice centrata
#   Xc = X - X.mean(axis=0)
#   U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
#   X2 = Xc @ Vt[:2].T
 
plt.figure(figsize=(9, 7))
i = 0
for nome, lista in gruppi.items():
    for w in lista:
        plt.scatter(X2[i, 0], X2[i, 1], color=colori[nome])
        plt.annotate(w, (X2[i, 0], X2[i, 1]), textcoords="offset points", xytext=(5, 5))
        i += 1
plt.title("Embedding GloVe (50 dim) proiettati in 2D con PCA")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("embedding_pca.png", dpi=150)
plt.show()