# ex04.py
import asyncio

async def func1():
    print("func1: Start")
    await asyncio.sleep(2)
    print("func1: End")

async def func2():
    print("func2: Start")
    await asyncio.sleep(1)
    print("func2: End")

async def main():
    await asyncio.gather(func1(), func2())
    
asyncio.run(main())