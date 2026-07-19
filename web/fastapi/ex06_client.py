# ex06_client.py
import time, requests, asyncio, aiohttp

def call_sync(url, times):
    start_time = time.time()
    for _ in range(times):
        requests.get(url)
    elapsed_time = time.time() - start_time
    print(f"[Sync] elapsed_time: {elapsed_time:.2f}")

async def call_async(url, times):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for _ in range(times)]
        await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time
    print(f"[Sync] elapsed_time: {elapsed_time:.2f}")

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000/"
    call_sync(api_url + "sync", 10)
    asyncio.run(call_async(api_url + "async", 10))
