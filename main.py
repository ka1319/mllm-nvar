import torch

from src.blip import blip
from src.common import update_labels
from src.fuyu import fuyu
from src.gemini import gemini
from src.gpt4v import gpt4v
from src.idefics import idefics
from src.idefics_instruct import idefics_instruct
from src.instructblip import instructblip
from src.llava import llava
from src.mmicl import mmicl
from src.qwen import qwen
from src.utils import get_args


@torch.inference_mode()
def main():
    args = get_args()
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    update_labels(args)

    if args.model.startswith("blip"):
        blip(args, device)
    elif args.model.startswith("fuyu"):
        fuyu(args, device)
    elif args.model.startswith("gemini"):
        gemini(args)
    elif args.model.startswith("gpt-4"):
        gpt4v(args)
    elif args.model.startswith("idefics"):
        if args.model.endswith("instruct"):
            idefics_instruct(args, device)
        else:
            idefics(args, device)
    elif args.model.startswith("instructblip"):
        instructblip(args, device)
    elif args.model.startswith("llava") or args.model.startswith("bakLlava"):
        llava(args, device)
    elif args.model.startswith("MMICL"):
        mmicl(args, device)
    elif args.model.startswith("Qwen"):
        qwen(args, device)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
