import cv2
import torch
import torch.nn.functional as F
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from pathlib import Path
from pycocotools import mask
from transformers import CLIPImageProcessor
from typing import List, Dict, Any

from .DataUtils import index_dataframe, IMG_EXTS
from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN


def collate_fn(batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True,
               local_rank=-1, cls_token_idx=None):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    cls_labels_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    has_text_description = []

    # Process batch items
    for (image_path, images, images_clip, conversations, masks, label,
         cls_labels, resize, questions, sampled_classes, inference, has_text,) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        cls_labels_list.append(torch.tensor(cls_labels))
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        has_text_description.append(has_text)

    # Handle image tokens
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # Pre-calculate original lengths before padding
    original_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    original_lengths = [len(ids) for ids in original_input_ids]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        original_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Process targets using original lengths
    targets = []
    for i, conversation in enumerate(conversation_list):
        if has_text_description[i]:
            target = input_ids[i].clone()
        else:
            target = torch.full_like(input_ids[i], IGNORE_INDEX)
        targets.append(target)

    targets = torch.stack(targets)
    conv = conversation_lib.default_conversation.copy()

    # Set separator based on conversation type
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    # Process each conversation using original lengths
    for idx, (conversation, target, orig_len) in enumerate(zip(conversation_list, targets, original_lengths)):
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"Warning: Unexpected format in conversation {idx}")
                continue

            parts[0] += sep

            # Calculate lengths
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        # Use original length for verification
        total_len = orig_len

        if cur_len != total_len:
            print(f"Length mismatch in conversation {idx}:")
            print(f"cur_len: {cur_len}, total_len: {total_len}")
            print(f"conversation: {conversation}")

        # Keep the assertion as a safety check
        assert cur_len == total_len, f"Length mismatch: cur_len={cur_len}, total_len={total_len}"

        target[cur_len:] = IGNORE_INDEX

    # Handle truncation for non-inference cases
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "cls_labels": torch.stack(cls_labels_list).view(-1),
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "cls_labels_list": cls_labels_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class FakePartsV2Dataset(torch.utils.data.Dataset):
    """
    Dataset that is *index-driven* via DataUtils.index_dataframe().
    It mirrors utils.SID_Set.CustomDataset's output structure so that collate_fn remains unchanged.
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir: Path, tokenizer, vision_tower: str, split: str = "",
                 precision: str = "fp32", image_size: int = 224, binary: bool = True, ) -> None:
        super().__init__()
        self.base_image_dir = base_image_dir
        self.split = split
        self.tokenizer = tokenizer
        self.precision = precision
        self.image_size = image_size
        self.binary = binary

        # Image processors
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # Build the dataframe index once
        self.df = index_dataframe(root_path=base_image_dir, file_exts=IMG_EXTS)
        # Keep only frames/images; videos could be supported later if needed
        # but we keep both since evaluation may be frame- or video-mode.
        # Labels are provided by DataUtils (0=real, 1=fake)
        if len(self.df) == 0:
            raise RuntimeError(f"No media files found under: {base_image_dir}")

        # Store core arrays for speed
        self.abs_paths: List[str] = self.df["abs_path"].tolist()
        self.cls_labels: List[int] = self.df["label"].astype(int).tolist()

        # Quick dict to retrieve metadata by abs path during validation
        self.meta_by_abs: Dict[str, Dict[str, Any]] = {
            r["abs_path"]: {
                "sample_id": r["rel_path"],  # id we will emit
                "task": r["task"],
                "method": r["method"],
                "subset": r["subset"],
                "label": int(r["label"]),
                "mode": r["mode"],
            }
            for _, r in self.df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.abs_paths)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _generate_response(self, cls_label: int) -> str:
        if self.binary:
            return "[CLS] The image is real" if cls_label == 0 else "[CLS] The image is fake [SEG]"
        else:
            # kept for compatibility with 3-class variants
            if cls_label == 0: return "[CLS] The image is real"
            if cls_label == 1: return "[CLS] The image is full synthetic"
            return "[CLS] The image is tampered [SEG]"

    def __getitem__(self, idx):
        image_path = self.abs_paths[idx]
        cls_label = int(self.cls_labels[idx])

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # CLIP branch
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # backbone branch
        image_resized = self.transform.apply_image(image)
        resize = image_resized.shape[:2]
        image_tensor = self.preprocess(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous())

        # No segmentation supervision in this *binary* evaluator path
        mask = torch.zeros((1, resize[0], resize[1]))

        # Conversation
        conv = conversation_lib.default_conversation.copy()
        q = f"{DEFAULT_IMAGE_TOKEN}\nIs this image real or fake?"
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], self._generate_response(cls_label))
        conversation = conv.get_prompt()
        has_text = False

        labels = torch.full(
            (self.image_size, self.image_size),
            self.ignore_label,
            dtype=torch.long
        )
        return (
            image_path,  # absolute path
            image_tensor,  # images
            image_clip,  # images_clip
            [conversation],  # conversations
            mask,  # masks (unused for binary)
            labels,  # labels (unused for binary)
            cls_label,  # cls_labels
            resize,  # resize
            None, None,  # questions, sampled_classes (unused)
            False,  # inference flag (set later in collate)
            has_text,  # text supervision for seg (false)
        )

    # Helper for validator
    def meta_from_abs(self, abs_path: str) -> Dict[str, Any]:
        return self.meta_by_abs[abs_path]

# class CustomDataset(torch.utils.data.Dataset):
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     img_size = 1024
#     ignore_label = 255
#
#     def __init__(
#             self,
#             base_image_dir,  # Root directory containing real/full_synthetic/tampered
#             tokenizer,
#             vision_tower,
#             split="",
#             precision: str = "fp32",
#             image_size: int = 224,
#             binary=True,
#             real_glob=None,
#             fake_glob=None
#     ):
#         self.base_image_dir = base_image_dir
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision
#         self.split = split
#         # Image processing
#         self.transform = ResizeLongestSide(image_size)
#         self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
#         # Set up paths
#         # split_dir = os.path.join(base_image_dir, split)
#         # required_dirs = ["0_real", "full_synthetic", "tampered"]
#         # for dir_name in required_dirs:
#         #     dir_path = os.path.join(split_dir, dir_name)
#         #     if not os.path.exists(dir_path):
#         #         raise ValueError(f"Required directory {dir_path} does not exist!")
#         split_dir = os.path.join(base_image_dir, split)
#         self.binary = binary
#
#         # Load images and verify
#         self.images = []
#         self.cls_labels = []
#         self.invalid_samples = []  # Track problematic samples
#
#         # Load images and verify counts
#         # real_images = glob.glob(os.path.join(split_dir, "0_real", "*.jpg"))
#         # full_syn_images = glob.glob(os.path.join(split_dir, "full_synthetic", "*.png"))
#         # tampered_images = glob.glob(os.path.join(split_dir, "tampered", "*.png"))
#         def grab_recursive(patterns):
#             out = []
#             for p in patterns:
#                 out.extend(glob.glob(p, recursive=True))
#             return out
#
#         def ext_patterns(root):
#             return [os.path.join(root, "**", f"*.{e}") for e in ("png", "jpg", "jpeg", "webp")]
#
#         images = []
#         labels = []
#         print(f"Traverse under {split_dir} for binary real/fake classification; bi-mode: {binary}")
#         if binary:
#             # REAL
#             real_list = grab_recursive([real_glob] if real_glob else ext_patterns(os.path.join(split_dir, "0_real")))
#             # FAKE
#             fake_list = grab_recursive([fake_glob] if fake_glob else ext_patterns(os.path.join(split_dir, "1_fake")))
#             real_list = sorted(set(real_list))
#             fake_list = sorted(set(fake_list))
#             images.extend(real_list)
#             labels.extend([0] * len(real_list))
#             images.extend(fake_list)
#             labels.extend([1] * len(fake_list))
#             self.has_tampered_masks = False
#         else:
#             real_list = grab_recursive(ext_patterns(os.path.join(split_dir, "0_real")))
#             fs_list = grab_recursive(ext_patterns(os.path.join(split_dir, "full_synthetic")))
#             tam_list = grab_recursive(ext_patterns(os.path.join(split_dir, "tampered")))
#             real_list, fs_list, tam_list = map(lambda x: sorted(set(x)), (real_list, fs_list, tam_list))
#
#             # keep only tampered images that have masks
#             valid_tam = []
#             for p in tam_list:
#                 base = os.path.splitext(os.path.basename(p))[0]
#                 mask = os.path.join(split_dir, "masks", f"{base}_mask.png")
#                 if os.path.exists(mask):
#                     valid_tam.append(p)
#
#             images.extend(real_list)
#             labels.extend([0] * len(real_list))
#             images.extend(fs_list)
#             labels.extend([1] * len(fs_list))
#             images.extend(valid_tam)
#             labels.extend([2] * len(valid_tam))
#             self.has_tampered_masks = True
#
#         self.images = images
#         self.cls_labels = labels
#         assert len(self.images) == len(self.cls_labels), "image/label size mismatch"
#
#         # # Verify tampered images have corresponding masks
#         # valid_tampered_images = []
#         # for img_path in tampered_images:
#         #     # Extract the base filename without extension
#         #     base_name = os.path.splitext(os.path.basename(img_path))[0]
#         #     # Construct the mask path (assuming the mask filename appends '_mask' to the base name)
#         #     mask_name = f"{base_name}_mask.png"
#         #     mask_path = os.path.join(split_dir, "masks", mask_name)
#         #     # Check if the mask exists
#         #     if os.path.exists(mask_path):
#         #         valid_tampered_images.append(img_path)
#         #     else:
#         #         print(f"Mask not found for: {img_path}")
#         #
#         # # Add only valid images to the dataset
#         # self.images.extend(real_images)
#         # self.images.extend(full_syn_images)
#         # self.images.extend(valid_tampered_images)  # Use valid_tampered_images here
#         #
#         # # Assign labels based on the valid counts
#         # self.cls_labels.extend([0] * len(real_images))
#         # self.cls_labels.extend([1] * len(full_syn_images))
#         # self.cls_labels.extend([2] * len(valid_tampered_images))  # Use valid_tampered_images here
#         #
#         # # Print dataset statistics
#         # print(f"\nDataset Statistics for {split} split:")
#         # print(f"Real images: {len(real_images)}")
#         # print(f"Full synthetic images: {len(full_syn_images)}")
#         # print(f"Tampered images: {len(valid_tampered_images)} (Valid) / {len(tampered_images)} (Total)")
#         # if self.invalid_samples:
#         #     print(f"Warning: Found {len(self.invalid_samples)} invalid samples")
#
#     def __len__(self):
#         return len(self.images)
#
#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         x = (x - self.pixel_mean) / self.pixel_std
#         h, w = x.shape[-2:]
#         padh = self.img_size - h
#         padw = self.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x
#
#     # def _generate_response(self, cls_label, image_name):
#     #     """Generate appropriate response based on image type and available description"""
#     #     if cls_label == 0:
#     #         return "[CLS] The image is real"
#     #     elif cls_label == 1:
#     #         return "[CLS] The image is full synthetic"
#     #     else:  # cls_label == 2 (tampered)
#     #         return "[CLS] The image is tampered [SEG]"
#     def _generate_response(self, cls_label, image_name):
#         """Generate response matching binary vs 3-class setup"""
#         if self.binary:
#             return "[CLS] The image is real" if cls_label == 0 else "[CLS] The image is fake [SEG]"
#         else:
#             if cls_label == 0: return "[CLS] The image is real"
#             if cls_label == 1: return "[CLS] The image is full synthetic"
#             return "[CLS] The image is tampered [SEG]"
#
#     def __getitem__(self, idx):
#         image_path = self.images[idx]
#         image_name = os.path.basename(image_path)
#         cls_labels = self.cls_labels[idx]
#         # Load and process image
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Process for CLIP
#         image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
#
#         # Process image for model
#         image = self.transform.apply_image(image)
#         resize = image.shape[:2]
#         image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
#
#         # Initialize mask
#         mask = torch.zeros((1, resize[0], resize[1]))
#
#         # Load mask for tampered
#         if cls_labels == 2:
#             base_name = os.path.splitext(image_name)[0]
#             mask_name = f"{base_name}_mask.png"
#             mask_path = os.path.join(self.base_image_dir, self.split, "masks", mask_name)
#
#             if os.path.exists(mask_path):
#                 mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                 mask_img = self.transform.apply_image(mask_img)
#                 mask_img = mask_img / 255.0
#                 mask = torch.from_numpy(mask_img).unsqueeze(0)
#
#         # Generate conversation
#         conv = conversation_lib.default_conversation.copy()
#         # conv.append_message(conv.roles[0],
#         #                     f"{DEFAULT_IMAGE_TOKEN}\nCan you identify if this image is real, full synthetic, or tampered image? Please mask the tampered regions if it is tampered.")
#         if self.binary:
#             q = f"{DEFAULT_IMAGE_TOKEN}\nIs this image real or fake?"
#         else:
#             q = f"{DEFAULT_IMAGE_TOKEN}\nIs this image real, full synthetic, or tampered? If tampered, please mask the tampered regions."
#         conv.append_message(conv.roles[0], q)
#
#         response = self._generate_response(cls_labels, image_name)
#         conv.append_message(conv.roles[1], response)
#         conversation = conv.get_prompt()
#         has_text = False
#         # labels = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
#         if (not self.binary) and (cls_labels == 2) and self.has_tampered_masks:
#             # gt_mask should be [H, W] with {0,1}
#             labels = torch.where(mask[0] > 0.5, 1, 0).long()
#             has_text = True
#         else:
#             labels = torch.full(
#                 (self.image_size, self.image_size),
#                 self.ignore_label,
#                 dtype=torch.long
#             )
#
#         return image_path, image, image_clip, [
#             conversation], mask, labels, cls_labels, resize, None, None, False, has_text
#
#     def __len__(self):
#         return len(self.images)
