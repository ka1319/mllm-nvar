from transformers import AutoProcessor, IdeficsForVisionText2Text

from src.common import (
    INFLECT_ENGINE,
    LABLES,
    OBO_EQUIV_LABELS,
    generate,
    generate_scores,
    get_icl_prompt,
    get_icl_text,
    get_label,
    get_prompt,
    get_torch_dtype,
    run,
)


def _get_model(args, device):
    model_name = f"HuggingFaceM4/{args.model}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = IdeficsForVisionText2Text.from_pretrained(
        model_name, torch_dtype=get_torch_dtype(args)
    ).to(device)
    return model, processor


def _get_prompt(args, sample, query_images=None, answer_image=None):
    if args.mode == "one-by-one":
        if query_images is None or answer_image is None:
            return None
        num_query_images = INFLECT_ENGINE.number_to_words(len(query_images))
        prompt = [
            f"Here are {num_query_images} images: ",
            *query_images,
            "The following image is: ",
            answer_image,
            "Is it correct?",
        ]
    elif "icl" in args.mode:
        prompt = [
            get_icl_prompt(),
        ]
        for icl_example in sample["icl_examples"]:
            prompt += [
                *icl_example["images"],
                get_icl_text(args, icl_example),
            ]
        prompt += [
            *sample["images"],
        ]
    else:
        prompt = [
            *sample["images"],
            get_prompt(args, sample),
        ]
    return prompt


def _generate(args, model, processor, _, prompt, device):
    inputs = processor(prompt, return_tensors="pt").to(device)
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids
    if args.mode in ["consistency-easy", "consistency-hard"]:
        outputs = generate(args, model, inputs, bad_words_ids=bad_words_ids)
        return -1, processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        outputs = generate_scores(args, model, inputs, bad_words_ids=bad_words_ids)

        _processor = lambda x: processor.tokenizer(
            x, return_tensors="pt", add_special_tokens=False
        ).input_ids[0][-1]
        _labels = OBO_EQUIV_LABELS if args.mode == "one-by-one" else LABLES
        p, label = get_label(args, outputs, _processor, _labels)
        return p, label


def _generate_obo(args, model, processor, images, _, device):
    query_images, answer_images = images[: -len(LABLES)], images[-len(LABLES) :]
    max_p, predicted_label = -1, None
    for answer_image, label in zip(answer_images, LABLES):
        _prompt = _get_prompt(args, None, query_images, answer_image)
        p, _ = _generate(args, model, processor, _, _prompt, device)
        if p > max_p:
            max_p = p
            predicted_label = label
    return max_p, predicted_label


def idefics(args, device):
    func = _generate_obo if args.mode == "one-by-one" else _generate
    return run(args, device, _get_model, _get_prompt, func)
