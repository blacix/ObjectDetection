import cv2 as cv
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
from utils import OpsMeter


cap_elp = UndistortedVideoCapture(0, fisheye=True)
cap_elp.camera_calibration.load_calibration('elp')
cap_logitech = UndistortedVideoCapture(2, fisheye=False)
cap_logitech.camera_calibration.load_calibration('logitech')

image_processor_elp = ImageProcessor()
image_processor_logitech = ImageProcessor()

executor = ThreadPoolExecutor(max_workers=3)
image_processors = [(image_processor_elp, cap_elp), (image_processor_logitech, cap_logitech)]

ops_meter = OpsMeter()


def main():
    ops = 0
    while True:
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
                # display_text = f'ops: {ops}'
                # image = cv.putText(image, display_text, (00, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                #                    cv.LINE_AA)
                cv.imshow(str(cap.camera_calibration.camera_id), image)

        ops = ops_meter.loop()
        print(ops)
        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


def process(image_processor, cap):
    ret, image = cap.read()
    images = image_processor.process_image_paralell(image)
    # print(f'processed {threading.currentThread().ident}')
    return images


if __name__ == '__main__':
    main()
