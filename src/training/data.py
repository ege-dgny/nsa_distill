"""LIBERO RLDS dataloader wrapper for NSA-Distill.

Reuses OpenVLA-OFT's `RLDSDataset` + `PaddedCollatorForActionPrediction` pipeline
unchanged. We only need a single `next(loader_iter)` per training step; gradient
accumulation is handled by the train loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402


def make_libero_dataloader(
    data_root_dir: str,
    dataset_name: str,
    teacher_dir: str,
    batch_size: int,
    image_sizes,
    *,
    use_wrist_image: bool = True,
    use_proprio: bool = True,
    shuffle_buffer_size: int = 100_000,
    image_aug: bool = True,
):
    """Build a DataLoader yielding RLDS batches for LIBERO-style training.

    Returns: (dataloader, processor, action_tokenizer, dataset_statistics)
    """
    processor = AutoProcessor.from_pretrained(teacher_dir, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
    )

    dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=tuple(image_sizes),
        shuffle_buffer_size=shuffle_buffer_size,
        image_aug=image_aug,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader, processor, action_tokenizer, dataset.dataset_statistics
