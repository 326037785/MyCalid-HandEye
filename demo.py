import numpy as np
import cv2

def main():
    '''
    A simple test demo
    '''
    a = np.abs(np.random.rand(300, 300,3) * 255)
    a = a.astype(np.uint8)
    cv2.imshow('test', a)
    cv2.waitKey(1000)
    print("Hello World.")

if __name__ == '__main__':
    main()