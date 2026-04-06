
__version__ = "2.5.1"
from .configuration_gpt2 import GPT2Config, GPT2SmallConfig
from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    )
