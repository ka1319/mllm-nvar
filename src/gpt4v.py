import os

import requests

from src.common import get_icl_prompt, get_icl_text, get_prompt, run


def _get_prompt(args, sample):
    if "icl" in args.mode:
        content = [
            {
                "type": "text",
                "text": get_icl_prompt(),
            },
        ]
        for i, icl_example in enumerate(sample["icl_examples"], 1):
            if "asym" in args.mode:
                content += [
                    {
                        "type": "text",
                        "text": f"\n\nDemonstration {i}:\n\n{icl_example['text_description']}",
                    },
                ]
            else:
                content += [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                    for image in icl_example["images"]
                ]
            content += [
                {
                    "type": "text",
                    "text": "\n\n" + get_icl_text(args, icl_example),
                },
            ]
        content += [
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                for image in sample["images"]
            ],
        ]
        if "asym" in args.mode:
            content += [
                {
                    "type": "text",
                    "text": "\n\nLet's solve the puzzle in the image, step by step, similar to the demonstrations.",
                },
            ]
    elif args.mode == "text-description":
        content = [
            {"type": "text", "text": get_prompt(args, sample)},
        ]
    else:
        content = [
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                for image in sample["images"]
            ],
            {"type": "text", "text": get_prompt(args, sample)},
        ]
    prompt = [
        {
            "role": "user",
            "content": content,
        },
    ]
    if args.mode == "hints-corrective":
        prompt += [
            {"role": "assistant", "content": sample["output"]},
            {
                "role": "user",
                "content": f"Hint: {sample['corrective_hint'].capitalize()}.",
            },
        ]
    return prompt


def _generate(args, model, processor, images, prompt, device):
    api_key = os.environ["OPENAI_API_KEY"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": args.model,
        "messages": prompt,
        "max_tokens": args.max_generation_length,
        "top_p": 1.0,
        "temperature": 0.0,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    if response.status_code != 200:
        print("response:", response.json())
        return -1, None

    return -1, response.json()["choices"][0]["message"]["content"]


def gpt4v(args):
    return run(args, None, None, _get_prompt, _generate)
