import asyncio
import random
strings = ['a', 'b', 'c']


async def async_func(string):
    sleep_time = random.randint(0, 3)
    print(f'async_func sleeps {sleep_time}')
    await asyncio.sleep(sleep_time)
    return string


async def async_main():
    # t1 = asyncio.create_task(async_func('x'))
    # t2 = asyncio.create_task(async_func2('x2'))
    #
    # x1 = await t1
    # x2 = await t2
    #
    # print(f'{x1} {x2} ')
    # tasks = [asyncio.create_task(async_func(string)) for string in strings]
    # for t in tasks:
    #     res = await t
    #     print(res)

    coroutines = [async_func(string) for string in strings]
    # done, pending = await asyncio.wait(coroutines)
    # for coroutine in done:
    #     print(coroutine.result())
    results = await asyncio.gather(*coroutines)
    for r in results:
        print(r)


if __name__ == '__main__':
    asyncio.run(async_main())
