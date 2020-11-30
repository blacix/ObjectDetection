import cv2 as cv
from image_processing import ImageProcessor


def main():
    # image_processor1 = ImageProcessor(0)
    image_processor2 = ImageProcessor(2)
    while True:
        # image1 = image_processor1.process_image()
        image2 = image_processor2.process_image()

        # cv.imshow('camera 0', image1)
        cv.imshow('camera 2', image2)

        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
