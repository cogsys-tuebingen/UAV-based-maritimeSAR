import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_structures.data_frame import DataFrame, SubFrame


def constructor_DataFrame(image, scale):
    global frames
    # gets the current frame
    main_img = image
    main_shape = image.shape
    main_dtype = image.dtype
    n_subs = np.random.randint(10)
    sub_frames = []
    main = DataFrame(main_img, main_shape, main_dtype, scale, n_subs, sub_frames)
    # creates the main frame and looks for a random amount of subframes
    frames = [i for i in frames if i[1] != 0]
    for i in range(main.n_subs):
        xmin = np.random.randint(main.main_shape[1] - 100)
        ymin = np.random.randint(main.main_shape[0] - 100)
        xmax = np.random.randint(xmin + 100, main.main_shape[1])
        ymax = np.random.randint(ymin + 100, main.main_shape[0])
        sub_orig_coords = (xmin, ymin, xmax, ymax)
        frames.append([sub_orig_coords, 3])
    for i in frames:
        sub_img = main.main_img[i[0][1]:i[0][3], i[0][0]:i[0][2]]
        sub_shape = sub_img.shape
        sub_dtype = sub_img.dtype
        frame = SubFrame(sub_img, sub_shape, sub_dtype, i[0])
        main.sub_frames.append(frame)
        i[1] -= 1
    main.n_subs = len(sub_frames)
    # downscale
    width = int(main.main_shape[1] * main.scale)
    height = int(main.main_shape[0] * main.scale)
    dim = (width, height)
    main.main_img = cv2.resize(main.main_img, dim, interpolation=cv2.INTER_AREA)
    main.main_shape = main.main_img.shape
    return main


def getFrame(cap):
    ret, img = cap.read()
    if img is None:
        return None
    image = np.stack(img, axis=0)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def createRoI(coords):
    global frames
    frames.append([coords, 10])


def overlay(Dframe):
    width = int(Dframe.main_shape[1] / Dframe.scale)
    height = int(Dframe.main_shape[0] / Dframe.scale)
    dim = (width, height)  # upscale / downscale function seperat
    Dframe.main_img = cv2.resize(Dframe.main_img, dim, interpolation=cv2.INTER_AREA)
    Dframe.main_shape = Dframe.main_img.shape
    if Dframe.n_subs != 0:
        for i in range(Dframe.n_subs):
            sub = Dframe.sub_frames[i]
            xmin = sub.sub_orig_coords[1]
            xmax = sub.sub_orig_coords[3]
            ymin = sub.sub_orig_coords[0]
            ymax = sub.sub_orig_coords[2]
            sub.sub_img = cv2.rectangle(sub.sub_img, (0, 0), (sub.sub_shape[1] - 1, sub.sub_shape[0] - 1), (0, 0, 0), 1)
            Dframe.main_img[xmin:xmax,
            ymin:ymax] = sub.sub_img[0:(xmax - xmin),
                         0:(ymax - ymin)]
    return Dframe.main_img


if __name__ == "__main__":
    cap = cv2.VideoCapture('../mockup/example.mp4')
    plt.figure()
    #subplot = plt.subplot(1, 1, 1)
    #frame = subplot.imshow(getFrame(cap))
    #while cap.isOpened():
     #   frame.set_data(overlay(constructor_DataFrame(getFrame(cap), 0.05)))
      #  plt.pause(0.2)
    #cap.release()
