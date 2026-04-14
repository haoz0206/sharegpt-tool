from .sharegpt_utils import ShareGPTDatasetConfig, ShareGPTMessageDataset, ShareGPTParser, ShareGPTTagConfig, ParsedSample

try:
    from .rlvr_dataset import RLVRShareGPTDataset, build_config
    from .agent_dataset import AgentShareGPTDataset
except Exception:  # pragma: no cover - optional runtime dependency (verl/ray)
    RLVRShareGPTDataset = None
    AgentShareGPTDataset = None
    build_config = None

__all__ = [
    "ShareGPTTagConfig",
    "ShareGPTDatasetConfig",
    "ShareGPTParser",
    "ShareGPTMessageDataset",
    "ParsedSample",
]

if build_config is not None and RLVRShareGPTDataset is not None:
    __all__.extend(
        [
            "build_config",
            "RLVRShareGPTDataset",
            "AgentShareGPTDataset",
        ]
    )
