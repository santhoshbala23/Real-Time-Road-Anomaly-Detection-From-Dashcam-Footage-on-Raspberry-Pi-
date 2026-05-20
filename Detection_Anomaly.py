# YOLOv5 Frame Skip Detection Script (FINAL OPTIMIZED VERSION)
# Optimized for Raspberry Pi 4 CPU + USB webcam
# Improvements:
# - Frame skipping
# - Lower confidence for small potholes
# - Supports img 416
# - Compatible with latest YOLOv5

import argparse
import cv2
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


def run(weights='best.pt', source='0', img=416):
    # ===== Device =====
    device = select_device('cpu')

    # ===== Model =====
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img, s=stride)

    # ===== Webcam =====
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)

    # ===== Detection tuning =====
    conf_thres = 0.15   # lower for potholes
    iou_thres = 0.45

    # ===== Frame skip tuning =====
    frame_count = 0
    skip_frames = 2     # frequent updates for small objects
    last_det = None

    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        run_detect = (frame_count % skip_frames == 0)

        # ===== Preprocess =====
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        # ===== Inference with frame skip =====
        if run_detect or last_det is None:
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            last_det = pred
        else:
            pred = last_det

        # ===== Draw results =====
        for i, det in enumerate(pred):
            im0 = im0s[i].copy()

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    x1, y1, x2, y2 = map(int, xyxy)

                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Road Detection', im0)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--img', type=int, default=416)
    return parser.parse_args()


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
