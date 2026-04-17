"""Agent-style ShareGPT dataset with full media-reference resolution.

:class:`RLVRShareGPTDataset` assumes every media reference lives in the
prompt text: the parser builds a pool from top-level ``images`` / ``videos``
/ ``audios`` and substitutes tokens during conversation parsing. Agent
training breaks that assumption — tool ``create_kwargs`` and other
``extra_info`` subfields need the same resolved paths, and the parser
doesn't walk there by default.

:class:`AgentShareGPTDataset` opts in to that walk. Two behaviors come
with it:

1. **Indexed references resolve everywhere.** A string like ``<@image:0>``
   in ``extra_info.tools_kwargs.*.create_kwargs.image`` (or anywhere else
   under ``extra_info``) gets rewritten to the resolved ``images[0]``
   absolute path. Non-reference strings pass through unchanged.
2. **Prompt-side parsing is unchanged** versus the base class. Both legacy
   ``<image>`` tokens and indexed ``<@image:N>`` references already work
   in prompts via the shared parser; agent datasets just add the
   extra-info walk on top.

Behavioral delta is one config flag
(:attr:`ShareGPTDatasetConfig.resolve_extra_info_references`). Everything
else is inherited from :class:`RLVRShareGPTDataset`.

Use this class when ShareGPT rows carry tool-side or otherwise structured
media paths under ``extra_info``; use :class:`RLVRShareGPTDataset` for
classic prompt-only multimodal data (keeps the hot path free of a
recursive walk it would never need).
"""

from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import PreTrainedTokenizer, ProcessorMixin

from .rlvr_dataset import RLVRShareGPTDataset

logger = logging.getLogger(__name__)


class AgentShareGPTDataset(RLVRShareGPTDataset):
    """RLVRShareGPTDataset + extra_info reference resolution for agent rollouts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Inject the walk flag into the sharegpt sub-config so that
        # RLVRShareGPTDataset.__init__ picks it up when it calls build_config.
        # OmegaConf.merge returns a new config; the caller's object is untouched.
        #
        # Hydra CLI overrides like `+data.sharegpt.pass_through_keys=...` create
        # the sharegpt node in struct mode, so a plain merge would fail with
        # ConfigKeyError on the new `resolve_extra_info_references` key. Open
        # struct for the duration of the merge; the context manager restores
        # struct on `config` afterwards (the returned `merged_config` is a
        # deep-copied new object and keeps struct=False, which is fine since
        # RLVRShareGPTDataset only reads it via OmegaConf.to_container).
        override = OmegaConf.create({"sharegpt": {"resolve_extra_info_references": True}})
        with open_dict(config):
            merged_config = OmegaConf.merge(config, override)
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=merged_config,
            processor=processor,
            max_samples=max_samples,
        )


__all__ = ["AgentShareGPTDataset"]
