import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
from threading import Thread
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

        print('starting...')
        future_to_processors = \
            {executor.submit(process, img_proc, cap): (img_proc, cap) for (img_proc, cap) in image_processors}
        for future in concurrent.futures.as_completed(future_to_processors):
            (img_proc, cap) = future_to_processors[future]
            try:
                image = future.result()
            except Exception as exc:
                print(f"exception: {exc}")
            else:
                cv.imshow(str(cap.camera_calibration.camera_id), image)

        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


def process(image_processor, cap):
    ret, image = cap.read()
    image = image_processor.process_image(image)
    print(f'processed {threading.currentThread().ident}')
    return image


if __name__ == '__main__':
    main()
