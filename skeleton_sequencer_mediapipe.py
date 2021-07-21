#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # load model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while True:
        # camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # draw
        if results.pose_landmarks is not None:
            debug_image = draw_landmarks(
                debug_image,
                results.pose_landmarks,
            )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('MediaPipe Pose Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]
    face_size = int(image_width * 0.25)
    body_line_size = int(image_width * 0.01)

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # nose
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), face_size)

    if len(landmark_point) > 0:
        # shoulder
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), body_line_size)

        # right arm
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), body_line_size)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), body_line_size)

        # left arm
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), body_line_size)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), body_line_size)

        # right hand
        # if landmark_point[15][0] > visibility_th and landmark_point[17][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[15][1], landmark_point[17][1],
        #             (0, 255, 0), 2)
        # if landmark_point[17][0] > visibility_th and landmark_point[19][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[17][1], landmark_point[19][1],
        #             (0, 255, 0), 2)
        # if landmark_point[19][0] > visibility_th and landmark_point[21][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[19][1], landmark_point[21][1],
        #             (0, 255, 0), 2)
        # if landmark_point[21][0] > visibility_th and landmark_point[15][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[21][1], landmark_point[15][1],
        #             (0, 255, 0), 2)

        # left hand
        # if landmark_point[16][0] > visibility_th and landmark_point[18][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[16][1], landmark_point[18][1],
        #             (0, 255, 0), 2)
        # if landmark_point[18][0] > visibility_th and landmark_point[20][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[18][1], landmark_point[20][1],
        #             (0, 255, 0), 2)
        # if landmark_point[20][0] > visibility_th and landmark_point[22][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[20][1], landmark_point[22][1],
        #             (0, 255, 0), 2)
        # if landmark_point[22][0] > visibility_th and landmark_point[16][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[22][1], landmark_point[16][1],
        #             (0, 255, 0), 2)

        # body
        # if landmark_point[11][0] > visibility_th and landmark_point[23][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[11][1], landmark_point[23][1],
        #             (0, 255, 0), 2)
        # if landmark_point[12][0] > visibility_th and landmark_point[24][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[12][1], landmark_point[24][1],
        #             (0, 255, 0), 2)
        # if landmark_point[23][0] > visibility_th and landmark_point[24][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[23][1], landmark_point[24][1],
        #             (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # right foot
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # left foot
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return image


if __name__ == '__main__':
    main()
