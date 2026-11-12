#!/usr/bin/env python3
"""Label GelSight ball-indenter contacts with OpenCV instead of NanoGUI."""

import argparse
import os

import cv2
import numpy as np
import yaml

from calibration.utils import load_csv_as_dict


def detect_circle(image, bg_img, default_center, default_radius):
    diff_image = image.astype(np.float32) - bg_img.astype(np.float32)
    color_mask = np.linalg.norm(diff_image, axis=-1) > 15
    color_mask = cv2.dilate(color_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    color_mask = cv2.erode(color_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return default_center, default_radius
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return default_center, default_radius
    moments = cv2.moments(largest)
    if moments["m00"] < 1e-6:
        return default_center, default_radius
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    _, radius = cv2.minEnclosingCircle(largest)
    return [cx, cy], max(int(radius) - 2, 5)


def overlay_circle(image, center, radius):
    overlay = image.copy()
    cv2.circle(
        overlay, (int(center[0]), int(center[1])), int(radius), (128, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, overlay)
    cv2.circle(
        overlay, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 1
    )
    return overlay


def difference_image(image, bg_img):
    diff_img = (image.astype(np.float32) - bg_img.astype(np.float32)) * 3
    diff_img = np.clip(diff_img, -127, 128) + 127
    return diff_img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Label GelSight calibration contacts with OpenCV."
    )
    parser.add_argument("-b", "--calib_dir", required=True)
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-d", "--display_difference", action="store_true")
    parser.add_argument("-r", "--detect_circle", action="store_true")
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["imgh"]
        imgw = config["imgw"]

    catalog = load_csv_as_dict(os.path.join(args.calib_dir, "catalog.csv"))
    fnames = [
        os.path.join(args.calib_dir, rel_dir)
        for rel_dir in catalog["experiment_reldir"]
    ]
    bg_img = cv2.imread(os.path.join(args.calib_dir, "background.png"))
    if bg_img is None:
        raise SystemExit("Missing background.png in the calibration directory.")

    center = [imgw / 2, imgh / 2]
    radius = 40
    step = 2
    idx = 0
    load_new = True
    orig_img = None

    window = "GelSight label (Enter=save  arrows=move  m/p=radius  f/c=step  a=back  q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print(f"{len(fnames)} images. Enter saves, arrows move, m/p radius, f/c step, a previous, q quit.")
    while 0 <= idx < len(fnames):
        if load_new:
            image_path = os.path.join(fnames[idx], "gelsight.png")
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                print(f"Skipping missing image: {image_path}")
                idx += 1
                continue
            label_path = os.path.join(fnames[idx], "label.npz")
            if os.path.isfile(label_path):
                labeled = np.load(label_path)
                center = labeled["center"].astype(float).tolist()
                radius = float(labeled["radius"])
            elif args.detect_circle:
                center, radius = detect_circle(orig_img, bg_img, center, radius)
            load_new = False
            print(f"[{idx + 1}/{len(fnames)}] {fnames[idx]}")

        display = (
            difference_image(orig_img, bg_img)
            if args.display_difference
            else orig_img.copy()
        )
        display = overlay_circle(display, center, radius)
        display = cv2.resize(
            display, (imgw * args.scale, imgh * args.scale), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            display,
            f"{idx + 1}/{len(fnames)}  r={int(radius)}  step={step}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key_ex = cv2.waitKeyEx(0)
        key = key_ex & 0xFF
        qt_left, qt_up, qt_right, qt_down = 2424832, 2490368, 2555904, 2621440

        if key in (ord("q"), 27):
            break
        if key in (13, 10, ord("w")):
            save_dir = os.path.join(fnames[idx], "label.npz")
            np.savez(save_dir, center=center, radius=radius)
            labeled_img = overlay_circle(orig_img, center, radius)
            cv2.imwrite(os.path.join(fnames[idx], "labeled.png"), labeled_img)
            print(f"Saved {save_dir} center={center} radius={radius}")
            idx += 1
            load_new = True
            continue
        if key == ord("a"):
            idx = max(idx - 1, 0)
            load_new = True
            continue
        if key in (81, 2) or key_ex == qt_left:
            center[0] -= step
        elif key in (83, 3) or key_ex == qt_right:
            center[0] += step
        elif key in (82, 0) or key_ex == qt_up:
            center[1] -= step
        elif key in (84, 1) or key_ex == qt_down:
            center[1] += step
        elif key == ord("m"):
            radius = max(radius - 1, 3)
        elif key == ord("p"):
            radius += 1
        elif key == ord("f"):
            step = max(step / 2, 0.25)
        elif key == ord("c"):
            step *= 2

        center[0] = float(np.clip(center[0], 0, imgw - 1))
        center[1] = float(np.clip(center[1], 0, imgh - 1))

    cv2.destroyAllWindows()
    labeled = sum(
        1 for path in fnames if os.path.isfile(os.path.join(path, "label.npz"))
    )
    print(f"Done. {labeled}/{len(fnames)} images labeled.")


if __name__ == "__main__":
    main()
