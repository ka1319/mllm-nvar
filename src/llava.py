from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.common import (
    LABLES,
    generate,
    generate_scores,
    get_label,
    get_prompt,
    get_torch_dtype,
    run,
)


def _get_model(args, device):
    model_name = f"llava-hf/{args.model}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=get_torch_dtype(args), ignore_mismatched_sizes=True
    ).to(device)
    return model, processor


def _get_prompt(args, sample):
    if args.mode == "text-description":
        return f"USER: {get_prompt(args, sample)}\nASSISTANT:"
    return f"USER: <image>\n{get_prompt(args, sample)}\nASSISTANT:"


def _generate(args, model, processor, images, prompt, device):
    inputs = processor(prompt, images[0], return_tensors="pt").to(device)

    if args.mode == "text-description":
        inputs["pixel_values"] = None

    if args.mode in [
        "zs-cot",
        "icl-cot-asym",
        "consistency-easy",
        "consistency-hard",
        "text-description",
    ]:
        outputs = generate(args, model, inputs)
        return -1, processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        outputs = generate_scores(args, model, inputs)

        _processor = lambda x: processor.tokenizer(
            x, return_tensors="pt", add_special_tokens=False
        ).input_ids[0][-1]
        p, label = get_label(args, outputs, _processor, LABLES)
        return p, label


def llava(args, device):
    return run(args, device, _get_model, _get_prompt, _generate)
