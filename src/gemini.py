import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

from src.common import get_icl_prompt, get_icl_text, get_prompt, run

BARD_MODEL = None


def _get_prompt(args, sample):
    if "icl" in args.mode:
        prompt = [
            get_icl_prompt(),
        ]
        for i, icl_example in enumerate(sample["icl_examples"], 1):
            if "asym" in args.mode:
                prompt += [
                    f"\n\nDemonstration {i}:\n\n{icl_example['text_description']}",
                ]
            else:
                prompt += [
                    Image.load_from_file(image) for image in icl_example["images"]
                ]
            prompt += [
                "\n\n" + get_icl_text(args, icl_example),
            ]
        prompt += [
            *[Image.load_from_file(image) for image in sample["images"]],
        ]
        if "asym" in args.mode:
            prompt += [
                "\n\nLet's solve the puzzle in the image, step by step, similar to the demonstrations.",
            ]
    elif args.mode == "text-description":
        prompt = [
            get_prompt(args, sample),
        ]
    else:
        prompt = [
            *[Image.load_from_file(image) for image in sample["images"]],
            get_prompt(args, sample),
        ]
        if args.mode == "hints-corrective":
            prompt += [
                (
                    f"Your first answer was '{sample['output']}' which is incorrect. "
                    f"Consider the folowing hint: {sample['corrective_hint'].capitalize()}."
                ),
            ]
    return prompt


def _generate(args, model, processor, images, prompt, device):
    global BARD_MODEL
    if BARD_MODEL is None:
        BARD_MODEL = GenerativeModel("gemini-pro-vision")

    config = {
        "max_output_tokens": args.max_generation_length,
        "top_p": 1.0,
        "temperature": 0.0,
    }
    response = BARD_MODEL.generate_content(contents=prompt, generation_config=config)

    return -1, response.text


def gemini(args):
    vertexai.init(project="mllm-nvar", location="us-west1")
    return run(args, None, None, _get_prompt, _generate)
