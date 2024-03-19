import argparse
import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from model import get_model
from defaults import _C as cfg

categories = [
    ['male', 'female'],
    ['afroamerican', 'caucasian', 'asian'],
    ['nomakeup', 'verysubtle', 'makeup', 'notclear'],
    ['modernphoto', 'oldphoto'],
    ['happy', 'other', 'slightlyhappy', 'neutral']
]



def get_args():
    parser = argparse.ArgumentParser(description="Age estimation demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=str, default=None,
                        help="Model weight to be tested")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="Margin around detected face for age-gender estimation")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to which resulting images will be stored if set")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def draw_label(image, point, labels, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    x, y_start = point
    for label in labels:
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        y = y_start - size[1]
        cv2.rectangle(image, (x, y_start - size[1] - 3), (x + size[0], y_start), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x, y_start), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        y_start += size[1] + 10  # Move to the next label's start, adjust spacing as needed



@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img, None


def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)

    for img_path in img_dir.glob("*.*"):
        img = cv2.imread(str(img_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), img_path.name


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> loading model ")
    model = torch.load('trained_model.pth', map_location=torch.device('cpu'))


    device = "cpu"
    model = model.to(device)

    model.eval()
    margin = args.margin
    img_dir = args.img_dir
    detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()

    with torch.no_grad():
        for img, name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)

                # predict ages
                outputs = F.softmax(model(inputs[:,:101]), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs[:,:101] * ages).sum(axis=-1)

                # predict other categories
                predicted_genders_ = outputs[:,101:103].max(1)
                predicted_races_ = outputs[:,103:106].max(1)
                predicted_makeups_ = outputs[:,106:110].max(1)
                predicted_times_ = outputs[:,110:112].max(1)
                predicted_happinesses_ = outputs[:,112:].max(1)
                
                predicted_genders = [categories[0][i] for i in range(len(predicted_genders_))] 
                predicted_races = [categories[1][i] for i in range(len(predicted_races_)) ]
                predicted_makeups = [categories[2][i] for i in range(len(predicted_makeups_)) ]
                predicted_times = [categories[3][i] for i in range(len(predicted_times_)) ]
                predicted_happinesses = [categories[4][i] for i in range(len(predicted_happinesses_))]

                # draw results
                for i, d in enumerate(detected):
                    label_1 = "{}".format(int(predicted_ages[i]))
                    label_2 = "{}".format(predicted_genders[i])
                    label_3 = "{}".format(predicted_races[i])
                    label_4 = "{}".format(predicted_makeups[i])
                    label_5 = "{}".format(predicted_times[i])
                    label_6 = "{}".format(predicted_happinesses[i])
                    # inside the loop where you process each detected face
                    labels = [
                        "Age: {}".format(int(predicted_ages[i])),
                        "Gender: {}".format(predicted_genders[i]),
                        "Race: {}".format(predicted_races[i]),
                        "Makeup: {}".format(predicted_makeups[i]),
                        "Photo Time: {}".format(predicted_times[i]),
                        "Happiness: {}".format(predicted_happinesses[i])
                    ]
                    draw_label(img, (d.left()+160, d.top() - 30), labels)  # Adjust starting point as needed


            if args.output_dir is not None:
                output_path = output_dir.joinpath(name)
                cv2.imwrite(str(output_path), img)
            else:
                cv2.imshow("result", img)
                key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

                if key == 27:  # ESC
                    break


if __name__ == '__main__':
    main()
