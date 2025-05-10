# coding:utf-8
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import copy


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


image = cv2.imread('cherry.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import sys

sys.path.append("../../0-ending/1-分割叶片")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.array([[128, 128]])  # apple blueberry
# input_point = np.array([[1221.8, 2121.1]])  # B2
# input_point = np.array([[1529, 1496]])  # B3
# input_point = np.array([[2200, 1200]])  # B4
# input_point = np.array([[1692.4, 1110.1]])  # B5
# input_point = np.array([[2200, 1100]])  # B6
input_label = np.array([1])
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(5.12, 5.12))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = mask + 255
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    plt.Axes(fig, [0., 0., 1., 1.])  # 设置子图占满整个画布
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    # plt.savefig('pic-{}.png'.format(i + 1))
    plt.show()

for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = ~mask
    mask = mask + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(image, mask)
    res[res == 0] = 255
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    plt.Axes(fig, [0., 0., 1., 1.])  # 设置子图占满整个画布
    plt.imshow(res)
    plt.axis('off')
    plt.savefig('res-cherry-{}.png'.format(i + 1))
    plt.show()
