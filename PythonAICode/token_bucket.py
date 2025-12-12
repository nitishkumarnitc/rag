"""Token Bucket Algorithm Synchronous (thread-safe) Python implementation"""
import time
import threading

class TokenBucket:
    def __init__(self, capacity:float, refill_rate:float):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_ts=time.monotonic()

    def _refill(self):
        now=time.monotonic()
        elapsed=now-self.last_ts
        if elapsed<=0:
            return
        self.tokens=min((self.tokens+elapsed*self.refill_rate), self.capacity)
        self.last_ts=now

    def try_consume(self,tokens=1):
        with self.lock:
            self._refill()
            if tokens<=self.tokens:
                self.tokens-=tokens
                return True
            else: return False

    def wait_for(self, tokens:float=1.0, time_out:float | None=None) -> bool:
        """Blocking: wait until tokens available or timeout (seconds)."""
        deadline=None if time_out is None else time.monotonic()+time_out
        while True:
            with self.lock:
                self._refill()
                if self.tokens>=tokens:
                    self.tokens-=tokens
                    return True
            if deadline is not None and time.monotonic()>=deadline:
                return False

            time.sleep(0.01)


"""Asyncio variant"""
import asyncio

class AsyncTokenBucket():
    def __init__(self, capacity:float, refill_rate:float):
        self.capacity = capacity
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_ts=time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now=time.monotonic()
        elapsed=now-self.last_ts
        if elapsed<=0:
            return
        self.tokens=min((self.tokens+elapsed*self.refill_rate), self.capacity)
        self.last_ts=now

    async def try_consume(self,tokens=1)->bool:
        async with self._lock:
            self._refill()
            if tokens<=self.tokens:
                self.tokens-=tokens
                return True
            else: return False

    async def wait_for(self, tokens:float=1.0, time_out:float | None=None) -> bool:
        deadline=None if time_out is None else time.monotonic()+time_out
        while True:
            async with self._lock:
                self._refill()
                if self.tokens>=tokens:
                    self.tokens-=tokens
                    return True
                if deadline is not None and time.monotonic()>=deadline:
                    return False
                await asyncio.sleep(0.01)