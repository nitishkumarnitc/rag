import  math
import numpy as np
def cosine_similarity(a, b):
    """Computes the cosine similarity between two vectors"""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot_product = 0.0
    norm_a=0.0
    norm_b=0.0

    for x,y in zip(a,b):
        dot_product += x*y
        norm_a += x*x
        norm_b += y*y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


def cosine_similarty_via_numpy(a, b):
    """Computes the cosine similarity between two vectors"""
    a = np.array(a,dtype=float)
    b=np.array(b,dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same length")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a,b) / (norm_a*norm_b))
