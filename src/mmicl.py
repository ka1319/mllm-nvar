from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from src.common import (
    LABLES,
    generate,
    generate_scores,
    get_label,
    get_prompt,
    get_torch_dtype,
    run,
)

IMAGE_PLACEHOLDER = "图"
REPLACE_TOKEN = "".join(32 * [IMAGE_PLACEHOLDER])
CONFIG = {
    "MMICL-vicuna-7b": "Salesforce/instructblip-vicuna-7b",
    "MMICL-vicuna-13b": "Salesforce/instructblip-vicuna-13b",
    "MMICL-Instructblip-T5-xl": "Salesforce/instructblip-flan-t5-xl",
    "MMICL-Instructblip-T5-xxl": "Salesforce/instructblip-flan-t5-xxl",
}


def _get_model(args, device):
    model_name = f"BleachNick/{args.model}"
    processor = InstructBlipProcessor.from_pretrained(CONFIG[args.model])
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=get_torch_dtype(args)
    ).to(device)

    sp = [IMAGE_PLACEHOLDER] + [f"<image{i}>" for i in range(20)]
    sp = sp + processor.tokenizer.additional_special_tokens[len(sp) :]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": sp})
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(
        processor.qformer_tokenizer
    ):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))

    return model, processor


def _get_prompt(args, sample):
    return f"<image0>{REPLACE_TOKEN}. {get_prompt(args, sample)}"


def _generate(args, model, processor, images, prompt, device):
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)
    if args.mode in ["zs-cot", "icl-cot-asym", "consistency-easy", "consistency-hard"]:
        outputs = generate(args, model, inputs)
        return -1, processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        outputs = generate_scores(args, model, inputs)

        _processor = lambda x: processor.tokenizer(
            x, return_tensors="pt", add_special_tokens=False
        ).input_ids[0][-1]
        p, label = get_label(args, outputs, _processor, LABLES)
        return p, label


def mmicl(args, device):
    return run(args, device, _get_model, _get_prompt, _generate)
