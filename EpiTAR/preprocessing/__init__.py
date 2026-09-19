__all__ = [
    "BaselineClipDataset",
    "EVOBRAIN_TUSZ_CHANNELS",
    "MODEL_CHANNEL_POLICIES",
    "MODEL_INPUT_SPECS",
    "PREDICTION_RULES",
    "adapt_model_channels",
    "aggregate_view_logits",
    "prepare_native_model_batch",
    "variable_channel_collate",
]


def __getattr__(name):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
