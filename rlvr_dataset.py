import dataclasses
import logging
import traceback
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from .mm_utils import extract_vision_info, fetch_image, fetch_video
from .sharegpt_utils import ShareGPTMessageDataset, build_config

logger = logging.getLogger(__name__)


def _omegaconf_to_dict(config: Any) -> dict[str, Any] | None:
    """Extract sharegpt overrides from a verl DictConfig."""
    if config is None:
        return None
    if isinstance(config, DictConfig):
        plain = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        plain = config
    else:
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    if isinstance(plain, dict) and "sharegpt" in plain:
        return plain["sharegpt"]
    return None


def _process_vision_info_sync(messages, image_patch_size, config=None):
    """Extract and resize images/videos, injecting max_pixels/min_pixels from config."""
    vision_infos = extract_vision_info(messages)

    max_pixels = config.get("max_pixels", None) if config is not None else None
    min_pixels = config.get("min_pixels", None) if config is not None else None

    image_inputs = []
    video_inputs = []
    for ele in vision_infos:
        if max_pixels is not None:
            ele.setdefault("max_pixels", max_pixels)
        if min_pixels is not None:
            ele.setdefault("min_pixels", min_pixels)

        if "image" in ele or "image_url" in ele:
            image_inputs.append(fetch_image(ele, image_patch_size=image_patch_size))
        elif "video" in ele:
            video_input, _ = fetch_video(
                ele,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")

    return image_inputs or None, video_inputs or None


class RLVRShareGPTDataset(RLHFDataset):
    """verl adapter built on top of framework-agnostic ShareGPT parsing."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.sharegpt_config = build_config(_omegaconf_to_dict(config))
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def _download(self, use_origin_parquet=False):
        return None

    def _read_files_and_tokenize(self):
        self.base_dataset = ShareGPTMessageDataset(self.data_files, config=self.sharegpt_config)
        total = len(self.base_dataset)
        print(f"dataset len: {total}")

        self.sample_indices: list[int] | None = None
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                import numpy as np

                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                self.sample_indices = rng.choice(total, size=self.max_samples, replace=False).tolist()
            else:
                self.sample_indices = list(range(self.max_samples))
            print(f"selected {self.max_samples} random samples out of {total}")

        if self.filter_overlong_prompts:
            logger.warning("Ignoring data.filter_overlong_prompts for ShareGPTRawPromptDataset to keep lazy loading.")

        # Kept for compatibility with base RLHFDataset serialization logic.
        self.dataframe = None

    def maybe_filter_out_long_prompts(self, dataframe: list[dict[str, Any]] | None = None):
        if dataframe is None:
            dataframe = self.dataframe
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor_or_tokenizer = self.processor if self.processor is not None else self.tokenizer

        def doc2len(doc) -> int:
            try:
                messages = doc["messages"]
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                raw_prompt = processor_or_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_kwargs,
                )

                if self.processor is not None:
                    images, videos = _process_vision_info_sync(messages, self.image_patch_size, self.config)
                    model_inputs = self.processor(
                        text=[raw_prompt],
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        do_resize=False,
                    )
                    return len(model_inputs["input_ids"][0])

                return len(tokenizer.encode(raw_prompt, add_special_tokens=False))
            except Exception:
                print("Error processing one of the samples, skipping...")
                traceback.print_exc()
                return self.max_prompt_length + 1

        filtered = [doc for doc in dataframe if doc2len(doc) <= self.max_prompt_length]
        print(f"filter dataset len: {len(filtered)}")
        return filtered

    def __len__(self):
        if self.sample_indices is None:
            return len(self.base_dataset)
        return len(self.sample_indices)

    def _resolve_sample_index(self, item: int) -> int:
        if self.sample_indices is None:
            return item
        return self.sample_indices[item]

    def _build_messages(self, example: dict):
        return example["messages"]

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        """Extract and resize images/videos from messages, applying config pixel limits.

        Called by AgentLoop.process_vision_info (verl/experimental/agent_loop/agent_loop.py)
        before apply_chat_template, to produce pixel data for the processor.

        The base RLHFDataset.process_vision_info ignores the config parameter, so
        data.max_pixels / data.min_pixels have no effect. This override injects those
        values into each vision element so fetch_image/fetch_video respect them.
        """
        return _process_vision_info_sync(messages, image_patch_size, config)

    def __getitem__(self, item):
        sample_index = self._resolve_sample_index(item)
        parsed = self.base_dataset[sample_index]
        row_dict: dict[str, Any] = dataclasses.asdict(parsed)

        # Lift agent_name only (not arbitrary pass_through keys) so verl's
        # DataProto exposes it on non_tensor_batch for AgentLoopWorker dispatch.
        # Broad lifting is unsafe: unknown keys can mismatch batch shape across
        # samples or collide with framework fields.
        pass_through = row_dict.pop("pass_through", None) or {}
        if "agent_name" in pass_through:
            row_dict.setdefault("agent_name", pass_through["agent_name"])

        row_dict["raw_prompt"] = row_dict["messages"]
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        reward_model = row_dict.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {}
        reward_model.setdefault("ground_truth", row_dict.get("ground_truth"))
        row_dict["reward_model"] = reward_model

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        index = row_dict.get("extra_info", {}).get("index", sample_index)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict


__all__ = [
    "build_config",
    "RLVRShareGPTDataset",
]
