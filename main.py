import random
import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

cap_elp = UndistortedVideoCapture(0, fisheye=True)
cap_elp.camera_calibration.load_calibration('elp')
cap_logitech = UndistortedVideoCapture(2, fisheye=False)
cap_logitech.camera_calibration.load_calibration('logitech')

image_processor_elp = ImageProcessor()
image_processor_logitech = ImageProcessor()

executor = ThreadPoolExecutor(max_workers=3)
image_processors = [(image_processor_elp, cap_elp), (image_processor_logitech, cap_logitech)]


def main():
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    while True:
        # futures = []
        # for (img_proc, cap) in image_processors:
        #     futures.append(executor.submit(process, img_proc, cap))
        #
        # cnt = 0
        # for future in futures:
        #     image = future.result()
        #     cv.imshow(str(cnt), image)
        #     cnt += 1

        # print('starting...')
        future_to_processors = \
            {executor.submit(process, img_proc, cap): (img_proc, cap) for (img_proc, cap) in image_processors}
        # done, pending = concurrent.futures.wait(future_to_processors)
        # for future in done:
        for future in concurrent.futures.as_completed(future_to_processors):
            (img_proc, cap) = future_to_processors[future]
            try:
                image = future.result()
            except Exception as exc:
                print(f"exception: {exc}")
            else:
                cv.imshow(str(cap.camera_calibration.camera_id), image)

        # coroutines = [process(img_proc, cap) for (img_proc, cap) in image_processors]
        # results = await asyncio.gather(*coroutines)
        # for i in range(len(results)):
        #     cv.imshow(str(image_processors[i][1].camera_calibration.camera_id), results[i])
        # done, pending = await asyncio.wait(coroutines)
        # print(done)
        # for future in done:
        #     cv.imshow('a', future.result())

        # coroutines = [process(img_proc, cap) for (img_proc, cap) in image_processors]
        # results = await asyncio.gather(*coroutines)
        # # print(len(results))
        # for i in range(len(results)):
        #     # cv.imshow(str(i), results[i])
        #     cv.imshow(str(image_processors[i][1].camera_calibration.camera_id), results[i])

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        print(f'fps:{fps}')

        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


# async def process(image_processor, cap):
#     ret, image = cap.read()
#     images = await image_processor.process_image_async(image)
#     # print(f'processed {threading.currentThread().ident}')
#     return images


def process(image_processor, cap):
    # print(f'process {threading.currentThread().ident}')
    ret, image = cap.read()
    image = image_processor.process_image_paralell(image)
    print(f'processed {cap.camera_calibration.camera_id} {threading.currentThread().ident}')
    return image


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
    main()
    # asyncio.run(main())

