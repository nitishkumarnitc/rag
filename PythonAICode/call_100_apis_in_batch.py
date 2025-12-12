import asyncio
import aiohttp
import time
from typing import List, Tuple

async def fetch_one(session: aiohttp.ClientSession, url: str) -> Tuple[str, int, float]:
    """Fetch a single URL and return (url, status_or_error, latency_seconds)."""
    start = time.perf_counter()
    try:
        async with session.get(url, timeout=10) as resp:
            status = resp.status
            # optionally read body: await resp.text()
    except Exception as e:
        status = f"ERR:{type(e).__name__}"
    latency = time.perf_counter() - start
    return url, status, latency

def chunkify(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def fetch_in_batches(urls: List[str], batch_size: int = 10):
    total_start = time.perf_counter()
    results = []  # list of (url, status, latency)
    async with aiohttp.ClientSession() as session:
        for batch_idx, batch in enumerate(chunkify(urls, batch_size), start=1):
            batch_start = time.perf_counter()
            tasks = [fetch_one(session, u) for u in batch]
            batch_res = await asyncio.gather(*tasks, return_exceptions=False)
            batch_time = time.perf_counter() - batch_start
            print(f"Batch {batch_idx}: {len(batch_res)} requests done in {batch_time:.3f}s")
            results.extend(batch_res)
    total_time = time.perf_counter() - total_start
    print(f"Total: {len(results)} requests in {total_time:.3f}s\n")
    # print per-request summary (first N for brevity)
    for url, status, latency in results:
        print(f"{status:10} {latency:6.3f}s  {url}")
    return results

# Example usage:
if __name__ == "__main__":
    # Build 100 test URLs (replace with real endpoints)
    urls = ["https://httpbin.org/delay/0.2"] * 100  # example: each endpoint delays 0.2s
    asyncio.run(fetch_in_batches(urls, batch_size=10))
