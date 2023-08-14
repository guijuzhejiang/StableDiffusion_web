import cv2

def get_bgr(filename):
    image = cv2.imread(filename)
    target = [int(image.shape[0] / 2), int(image.shape[1] / 2)]
    print(image.size)
    b, g, r = image[target[0], target[1]]
    image = cv2.circle(image, [target[1], target[0]], 50, (int(b), int(g), int(r)), 20)

    print(f'colour = {b}, {g}, {r}')

    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    get_bgr('/home/ray/Downloads/image.png')