import argparse
import os

import numpy as np
import torch
from image_representation import ImageRepresenter
from PIL import Image
from tqdm import tqdm


def get_label_from_path(path):
    with open(path, "r") as f:
        labels = [int(i) - 1 for i in f.read().split("\n")]
    return labels


def run(args, data_path, image_representer):
    image_paths = os.listdir(data_path)
    answer_paths = sorted([i for i in image_paths if i.find("answer") != -1])
    question_paths = sorted([i for i in image_paths if i.find("query") != -1])
    question_images = [
        Image.open(os.path.join(data_path, i)).convert("RGB") for i in question_paths
    ]
    answer_images = [
        Image.open(os.path.join(data_path, i)).convert("RGB") for i in answer_paths
    ]

    if args.model == "pixel_rep" or args.model == "resnet_rep":
        question_represents = [image_representer(i) for i in question_images]
        answer_represents = [image_representer(i) for i in answer_images]
    else:
        all_represents = image_representer(question_images + answer_images)
        question_represents = all_represents[: len(question_images)]
        answer_represents = all_represents[len(question_images) :]

    if len(question_represents) == 3:
        target_represent = (
            question_represents[1] - question_represents[0] + question_represents[2]
        )
    elif len(question_represents) == 4:
        target_represent1 = 2 * question_represents[3] - question_represents[2]
        target_represent2 = 2 * question_represents[2] - question_represents[0]
        target_represent = (target_represent1 + target_represent2) / 2
    elif len(question_represents) == 5:
        target_represent1 = (
            question_represents[2] + question_represents[4] - question_represents[1]
        )
        target_represent2 = (
            question_represents[2] + question_represents[3] - question_represents[0]
        )
        target_represent = (target_represent1 + target_represent2) / 2
    elif len(question_represents) == 7:
        target_represent1 = (
            question_represents[3] + question_represents[6] - question_represents[2]
        )
        target_represent2 = (
            question_represents[3] + question_represents[5] - question_represents[1]
        )
        target_represent3 = (
            question_represents[3] + question_represents[4] - question_represents[0]
        )
        target_represent = (
            target_represent1 + target_represent2 + target_represent3
        ) / 3
    elif len(question_represents) == 8:
        target_represent1 = (
            question_represents[7] + question_represents[5] - question_represents[4]
        )
        target_represent2 = (
            question_represents[7] + question_represents[2] - question_represents[1]
        )
        target_represent3 = (
            question_represents[5] + question_represents[6] - question_represents[3]
        )
        target_represent4 = (
            question_represents[6] + question_represents[2] - question_represents[0]
        )

        target_represent = (
            target_represent1
            + target_represent2
            + target_represent3
            + target_represent4
        ) / 4
    else:
        raise ValueError("not correct number of question images")

    distances = [np.linalg.norm(target_represent - i) for i in answer_represents]
    return np.argmin(distances)


@torch.inference_mode
def main(args):
    rep = ImageRepresenter()
    rep_functions = {
        "pixel_rep": rep.pixel_rep,
        "resnet_rep": rep.resnet_rep,
        "vit_rep": rep.vit_rep,
    }

    image_representer = rep_functions[args.model]

    items = os.listdir(args.data_folder)

    if args.data_folder.find("iq50") != -1:
        items = sorted(items, key=lambda x: int(x.split("-")[-1]))
    elif args.data_folder.find("raven") != -1:
        items = sorted(items, key=lambda x: int(x.split("-")[-1]))

    predictions = []
    for item in tqdm(items, desc=f"Running on {args.model}", ncols=100):
        fn = os.path.join(args.data_folder, item)
        prediction = run(args, fn, image_representer)
        predictions.append(prediction)

    if args.data_folder.find("iq50") != -1:
        labels = [0] * len(predictions)
    elif args.data_folder.find("raven") != -1 or args.data_folder.find("marvel") != -1:
        labels = get_label_from_path(args.label_path)

    print(
        f"Accuracy: {sum([i == j for i, j in zip(predictions, labels)]) / len(predictions)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="../data/iq50/obo")
    parser.add_argument("--label_path", type=str, default="../data/iq50/obo/labels.txt")
    parser.add_argument("--model", type=str, default="pixel_rep")
    args = parser.parse_args()

    # Run experiment
    main(args)
