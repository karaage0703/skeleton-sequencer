#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import time

import pygame
import pygame.midi

pygame.init()
pygame.midi.init()

# time param
start_time = 0.0
dot_line = 0

# music setting
volume = 127
note_list = []

HUMAN_COLOR = (255, 0, 0)

# midi setup
for i in range(pygame.midi.get_count()):
    interf, name, input_dev, output_dev, opened = pygame.midi.get_device_info(i)
    if output_dev and b'NSX-39 ' in name:
        print('midi id=' + str(i))
        midi_output = pygame.midi.Output(i)

midi_output.set_instrument(1, 2)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--fullscreen', type=bool, default=False)

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


def get_pentatonic_scale(note):
    # C
    if note % 5 == 0:
        out_note = note // 5 * 12

    # D#
    if note % 5 == 1:
        out_note = note // 5 * 12 + 3

    # F
    if note % 5 == 2:
        out_note = note // 5 * 12 + 5

    # G
    if note % 5 == 3:
        out_note = note // 5 * 12 + 7

    # A#
    if note % 5 == 4:
        out_note = note // 5 * 12 + 10

    out_note += 60
    while out_note > 127:
        out_note -= 128

    return out_note


def skeleton_sequencer(src):
    global start_time
    global dot_line
    global note_list

    # parameters
    speed = 0.5
    d_circle = 30

    image_h, image_w = src.shape[:2]

    h_max = int(image_h / d_circle)
    w_max = int(image_w / d_circle)

    # create blank image
    npimg_target = np.zeros((image_h, image_w, 3), np.uint8)
    dot_color = [[0 for i in range(h_max)] for j in range(w_max)]

    # make dot information from ndarray
    for y in range(0, h_max):
        for x in range(0, w_max):
            dot_color[x][y] = src[y * d_circle][x * d_circle]

    # move dot
    while time.time() - start_time > speed:
        start_time += speed
        dot_line += 1
        if dot_line > w_max - 1:
            dot_line = 0

        # sound off
        for note in note_list:
            midi_output.note_off(note, volume, 2)

        # sound on
        note_list = []

        for y in range(0, h_max):
            if dot_color[dot_line][y].tolist() == list(HUMAN_COLOR):
                note_list.append(get_pentatonic_scale(y))

        for note in note_list:
            midi_output.note_on(note, volume, 2)

    # draw dot
    for y in range(0, h_max):
        for x in range(0, w_max):
            center = (int(x * d_circle + d_circle * 0.5), int(y * d_circle + d_circle * 0.5))
            if x == dot_line:
                if dot_color[dot_line][y].tolist() == list(HUMAN_COLOR):
                    cv.circle(npimg_target, center, int(d_circle / 2), [
                        255 - (int)(dot_color[x][y][0]), 255 - (int)(dot_color[x][y][1]), 255 - (int)(dot_color[x][y][2])],
                        thickness=-1, lineType=8, shift=0)
                else:
                    cv.circle(npimg_target, center, int(d_circle / 2), [
                        255, 255, 255], thickness=-1, lineType=8, shift=0)
            else:
                cv.circle(npimg_target, center, int(d_circle / 2), [
                    (int)(dot_color[x][y][0]), (int)(dot_color[x][y][1]), (int)(dot_color[x][y][2])],
                    thickness=-1, lineType=8, shift=0)

    return npimg_target


def main():
    global start_time

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

    window_name = 'Skeleton Sequencer'
    if args.fullscreen:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    start_time = time.time()
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

        image_ss = skeleton_sequencer(debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # cv.imshow(window_name, debug_image)
        cv.imshow(window_name, image_ss)

    for note in note_list:
        midi_output.note_off(note, volume, 2)

    midi_output.close()
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
    face_size = int(image_width * 0.3)
    face_offset = int(image_width * 0.05)
    body_line_size = int(image_width * 0.1)
    hand_line_size = int(image_width * 0.1)

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # Head
            cv.circle(image, (landmark_x, landmark_y - face_offset), 5, HUMAN_COLOR, face_size)

    if len(landmark_point) > 0:
        # shoulder
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    HUMAN_COLOR, body_line_size)

        # right arm
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    HUMAN_COLOR, body_line_size)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    HUMAN_COLOR, body_line_size)

        # left arm
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    HUMAN_COLOR, body_line_size)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    HUMAN_COLOR, body_line_size)

        # right hand
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    HUMAN_COLOR, hand_line_size)

        # left hand
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    HUMAN_COLOR, hand_line_size)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    HUMAN_COLOR, hand_line_size)

        # body shape
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    HUMAN_COLOR, body_line_size)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    HUMAN_COLOR, body_line_size)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    HUMAN_COLOR, body_line_size)

        # fill body
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th and landmark_point[23][
                0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.rectangle(image, landmark_point[11][1], landmark_point[24][1],
                         HUMAN_COLOR, -1)

        if len(landmark_point) > 25:
            # right foot
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        HUMAN_COLOR, body_line_size)

            # left foot
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        HUMAN_COLOR, body_line_size)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        HUMAN_COLOR, body_line_size)
    return image


if __name__ == '__main__':
    main()
