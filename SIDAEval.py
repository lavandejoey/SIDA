import sys
import argparse
import os
from functools import partial
from typing import Iterator, List
import random
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import deepspeed
import torch
import transformers
from peft import LoraConfig, get_peft_model
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.SID_Set import collate_fn, FakePartsV2Dataset
from utils.batch_sampler import BalancedBatchSampler
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda
from utils.DataUtils import standardise_predictions
import warnings
import glob
import pandas as pd
import torch.distributed as dist

warnings.filterwarnings("ignore")


def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"],
                        help="precision for inference")
    parser.add_argument("--vision_pretrained", default="ck/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./results", type=str)
    parser.add_argument("--results_root", default=None, type=str,
                        help="Parent folder containing past eval runs. If unset, dirname(log_base_dir) is also scanned.")
    parser.add_argument("--exp_name", default="sida", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


def _load_existing_sample_ids_all(log_base_dir: str, results_root: str | None = None) -> set[str]:

    ids: set[str] = set()
    search_roots: list[str] = []

    if results_root and os.path.isdir(results_root):
        search_roots.append(results_root)

    if log_base_dir:
        parent = os.path.dirname(os.path.abspath(log_base_dir))
        if os.path.isdir(parent):
            search_roots.append(parent)
        if os.path.isdir(log_base_dir):
            search_roots.append(log_base_dir)

    seen_files: set[str] = set()
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not (fn == "predictions.csv" or fn.startswith("predictions.rank")):
                    continue
                path = os.path.join(dirpath, fn)
                if path in seen_files:
                    continue
                seen_files.add(path)
                try:
                    df = pd.read_csv(path, usecols=["sample_id"])
                    ids.update(df["sample_id"].astype(str).tolist())
                except Exception:
                    pass
    print(f"[resume] Found {len(ids)} already-evaluated sample_ids across {len(seen_files)} CSV files.")
    return ids


class ShardByRankBatchSampler(torch.utils.data.Sampler[List[int]]):

    def __init__(self, batch_sampler, world_size: int, rank: int):
        self.batch_sampler = batch_sampler
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)

    def __iter__(self) -> Iterator[List[int]]:
        for i, batch in enumerate(self.batch_sampler):
            if i % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        L = len(self.batch_sampler)
        if L <= 0:
            return 0
        return (L - self.rank + self.world_size - 1) // self.world_size


def main(cli_args):
    args = parse_args(cli_args)
    env_local_rank = int(os.getenv("LOCAL_RANK", str(args.local_rank)))
    args.local_rank = env_local_rank

    deepspeed.init_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CLS]")
    tokenizer.add_tokens("[SEG]")
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = SIDAForCausalLM.from_pretrained(
        args.version, dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model_, lora_target_modules_):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model_.named_modules():
                if (isinstance(module, cls)
                        and all([x not in name for x in [
                            "visual_model", "vision_tower", "mm_projector", "text_hidden_fcs", "cls_head", "sida_fc1",
                            "attention_layer"]])
                        and any([x in name for x in lora_target_modules_])):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
                                 lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False
    for n, p in model.named_parameters():
        if any([x in n for x in
                ["embed_tokens", "mask_decoder", "text_hidden_fcs", "cls_head", "sida_fc1", "attention_layer"]]):
            p.requires_grad = True

    args.distributed = world_size > 1
    print(f"\nInitializing datasets:")

    # Build global skip set (scan parent of log_base_dir + results_root if provided)
    already = _load_existing_sample_ids_all(
        log_base_dir=args.log_base_dir,
        results_root=args.results_root
    )

    # Build dataset with exclusion
    val_dataset = FakePartsV2Dataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split="",
        precision=args.precision,
        image_size=args.image_size,
        binary=True,
        exclude_sample_ids=already,
    )

    if len(val_dataset) == 0:
        if len(already) > 0:
            print(f"[resume] No new samples to evaluate. Existing predictions already cover: {args.dataset_dir}")
        else:
            print("[resume] No samples found to evaluate.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    print(f"Validating with {len(val_dataset)} examples.")
    model_engine = model.to(args.local_rank)


    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    base_sampler = BalancedBatchSampler(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        world_size=1,  
        rank=0,
    )
    val_sampler = ShardByRankBatchSampler(base_sampler, world_size=world_size, rank=rank)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank
        )
    )
    try:
        print(f"[rank{rank}] local batches: {len(val_sampler)}")
    except Exception:
        pass
    # ------------------------------------------------------------------

    df = validate(val_loader, model_engine, args)

    os.makedirs(args.log_base_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    shard_csv = os.path.join(args.log_base_dir, f"predictions.rank{local_rank}.csv")
    df.to_csv(shard_csv, index=False)
    print(f"[rank{local_rank}] wrote shard: {shard_csv}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if local_rank == 0:
        frames = []
        main_csv = os.path.join(args.log_base_dir, "predictions.csv")
        if os.path.exists(main_csv):
            try:
                frames.append(pd.read_csv(main_csv))
            except Exception:
                pass
        for s in sorted(glob.glob(os.path.join(args.log_base_dir, "predictions.rank*.csv"))):
            try:
                frames.append(pd.read_csv(s))
            except Exception:
                pass
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged = merged.drop_duplicates(subset=["sample_id"], keep="last")
            merged.to_csv(main_csv, index=False)
            print(f"[rank0] merged → {main_csv} (rows={len(merged)})")
            for s in glob.glob(os.path.join(args.log_base_dir, "predictions.rank*.csv")):
                try:
                    os.remove(s)
                except Exception:
                    pass

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def validate(val_loader, model_engine, args, sample_ratio=None):
    """
    Validate and emit a standardised predictions table for downstream processing.
    The table follows DataUtils.REQUIRED_COLS.
    """
    model_engine.eval()
    rows = []
    correct = 0
    total = 0

    total_batches = len(val_loader)
    if sample_ratio is not None:
        import random as _r
        num_batches = max(1, int(total_batches * sample_ratio))
        sample_indices = set(_r.sample(range(total_batches), num_batches))
        print(f"Validating on {num_batches}/{total_batches} randomly sampled batches...")
    else:
        sample_indices = None

    dataset = val_loader.dataset
    model_name = f"SIDA-{args.exp_name}"

    import torch.nn.functional as F
    import tqdm

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        dev = getattr(model_engine, 'device', next(model_engine.parameters()).device)
        # Ensure all tensors are on the same device
        for k, v in list(input_dict.items()):
            if torch.is_tensor(v):
                input_dict[k] = v.to(dev, non_blocking=True)
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                input_dict[k] = [t.to(dev, non_blocking=True) for t in v]

        if sample_indices is not None and batch_idx not in sample_indices:
            continue

        torch.cuda.empty_cache()

        # Keep image paths for mapping back to metadata
        batch_paths = input_dict.get("image_paths", None)

        input_dict = dict_to_cuda(input_dict)
        # dtype management
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        input_dict['inference'] = True

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)  # [B,2]
        preds = torch.argmax(probs, dim=1)  # [B]
        cls_labels = input_dict["cls_labels"]  # [B]

        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        # Convert to CPU numpy for iteration
        probs_np = probs.detach().float().cpu().numpy()
        preds_np = preds.detach().int().cpu().numpy()
        labels_np = cls_labels.detach().int().cpu().numpy()

        # Assemble REQUIRED_COLS per sample
        for i in range(len(preds_np)):
            # Map back to dataset meta via absolute path
            abs_path = batch_paths[i] if batch_paths is not None else None
            meta = dataset.meta_from_abs(abs_path) if abs_path is not None else None
            if meta is None:
                meta = {
                    "sample_id": str(abs_path) if abs_path is not None else str(i),
                    "task": "unknown",
                    "method": "unknown",
                    "subset": "unknown",
                    "label": int(labels_np[i]),
                    "mode": "frame",
                }
            score_fake = float(probs_np[i, 1])  # Higher => more likely fake

            rows.append({
                "sample_id": str(meta["sample_id"]),
                "task": str(meta["task"]),
                "method": str(meta["method"]),
                "subset": str(meta["subset"]),
                "label": int(meta["label"]),
                "model": str(model_name),
                "mode": str(meta["mode"]),
                "score": float(score_fake),
                "pred": int(preds_np[i]),
            })

    accuracy = (correct / max(1, total)) * 100.0
    print(f"Classification Accuracy: {accuracy:.4f}%  (N={total})")

    # Standardise and return (saving is handled in main with per-rank shards)
    df = standardise_predictions(rows)
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
