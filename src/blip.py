from transformers import Blip2ForConditionalGeneration, Blip2Processor

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
    model_name = f"Salesforce/{args.model}"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=get_torch_dtype(args)
    ).to(device)
    return model, processor


def _generate(args, model, processor, images, prompt, device):
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
    if args.mode in ["consistency-easy", "consistency-hard"]:
        outputs = generate(args, model, inputs)
        return -1, processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        outputs = generate_scores(args, model, inputs)

        _processor = lambda x: processor.tokenizer(
            x, return_tensors="pt", add_special_tokens=False
        ).input_ids[0][-1]
        return get_label(args, outputs, _processor, LABLES)


def blip(args, device):
    return run(args, device, _get_model, get_prompt, _generate)
