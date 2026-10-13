import numpy as np

a = np.array([0,3,6,9])
b = np.array([0,5,6,1])

prod = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)


cosine = prod/(norm_a*norm_b)
print("Cosine Similarity:", cosine)