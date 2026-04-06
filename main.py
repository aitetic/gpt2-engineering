
import torch
import logging
from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, GPT2TokenizerFast, PreTrainedModel, PreTrainedTokenizer

# comment this if you want to load gpt2 class from transformers
from models import GPT2LMHeadModel
from models import GPT2Config, GPT2SmallConfig

# uncomment this if you want to load gpt2 class from transformers
# from transformers import GP2Config, GPT2LMHeadModel

from utils.gpt2_args_parser import ArgsParser

logger = logging.getLogger(__name__)
EOT_TOKEN = "<|endoftext|>"


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast),
    "gpt2-small": (GPT2SmallConfig, GPT2LMHeadModel, GPT2TokenizerFast),
}


def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.model_name_or_path == 'openai-gpt':
        special_tokens = {
            "bos_token": EOT_TOKEN,
            "eos_token": EOT_TOKEN,
            "additional_special_tokens": []
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
    elif args.model_name_or_path == 'gpt2':
        pass

    return model, tokenizer, model_class, args


def main():

    args = ArgsParser().parse()
    model, tokenizer, model_class, args = get_model_tokenizer(args)

if __name__ == "__main__":
    pass
