import heapq
from typing import Iterable, List, Tuple

def top_k_stream(candidates: Iterable[Tuple[float, str]], k: int) -> List[Tuple[float, str]]:
    """
    candidates: iterable of (score, item_id) - higher score = better
    returns: top-k list sorted descending by score
    """
    if k <= 0:
        return []

    # min-heap keeps the smallest of top-k at index 0
    heap: List[Tuple[float, str]] = []

    for score, item in candidates:
        if len(heap) < k:
            heapq.heappush(heap, (score, item))
        else:
            # if new item better than smallest in heap, replace it
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, item))

    # return sorted descending
    return sorted(heap, key=lambda x: x[0], reverse=True)

# Example usage
if __name__ == "__main__":
    items = [(0.2, "a"), (0.9, "b"), (0.5, "c"), (0.95, "d"), (0.7, "e")]
    print(top_k_stream(items, k=3))  # -> [(0.95,'d'), (0.9,'b'), (0.7,'e')]
