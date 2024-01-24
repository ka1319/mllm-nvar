import os

import numpy as np
import torch
from torchvision import models, transforms
from transformers import AutoProcessor, CLIPVisionModel


def get_free_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp")
    memory_available = [-int(x.split()[2]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


class ImageRepresenter(object):
    def __init__(self):
        self.device = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.eval()
        self.resnet_model.to(self.device)

        self.vit_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.vit_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.vit_model.eval()
        self.vit_model.to(self.device)

    def pixel_rep(self, img):
        img = img.resize((224, 224))

        img = np.array(img)

        img = img.reshape(-1)

        return img

    def resnet_rep(self, img):
        transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = transformation(img).unsqueeze(0).to(self.device)

        features = self.resnet_model(image).cpu().numpy().reshape(-1)

        return features

    def vit_rep(self, img):
        inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.vit_model(**inputs)

        outputs = outputs.pooler_output.squeeze().cpu().numpy()

        if type(img) == list:
            features = [outputs[i, :] for i in range(outputs.shape[0])]

        features = outputs

        return features
