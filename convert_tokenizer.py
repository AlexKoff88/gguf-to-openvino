from functools import lru_cache
from io import BytesIO
from typing import Any, Iterable, Union

import numpy as np
import openvino as ov
from openvino import Type, opset15 as opset, op
from openvino.utils.node_factory import NodeFactory
from openvino.utils.types import as_node, make_constant_node
from openvino_tokenizers import _ext_path as ov_tokenizers_extension_path


MIN_CACHE_CAPACITY = 20_000
VOCAB_SIZE_CACHE_PROPORTION = 0.2
MAX_LENGTH = 8192  # check the default constant in the original code


split_regex_mapping = {
    # https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/src/llama-vocab.cpp#L336
    "smollm": [
        "\\p{N}",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    ],
}
split_behaviour_mapping = {
    "\\p{N}": "isolate",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)": "isolate",
}

# https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/src/llama-vocab.cpp#L405
DEFAULT_BPE_SPLIT_RE = [
    "[\\p{P}\\$\\+<=>\\^~\\|]+",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    "\\p{N}+",
    "[0-9][0-9][0-9]",
]

# https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/include/llama.h#L120
def is_special(token_type: int) -> bool:
    return token_type == 3 or token_type == 4


def to_bytes(number: int) -> bytes:
    return number.to_bytes(4, "little")


def create_unpacked_string(strings: Iterable[Union[str, bytes]]) -> list[ov.Output]:
    """
    Convert any list of strings to U8/1D numpy array with begins, ends, and chars
    """
    begins = BytesIO()
    ends = BytesIO()
    chars = BytesIO()
    offset = 0

    for string in strings:
        byte_string = string.encode("utf-8") if isinstance(string, str) else string
        length = len(byte_string)

        begins.write(to_bytes(offset))
        offset += length
        ends.write(to_bytes(offset))
        chars.write(byte_string)

    begins = np.frombuffer(begins.getvalue(), np.int32)
    ends = np.frombuffer(ends.getvalue(), np.int32)
    chars = np.frombuffer(chars.getvalue(), np.uint8)

    return [op.Constant(ov.Tensor(x)).output(0) for x in [begins, ends, chars]]


def create_string_constant(strings: Union[str, bytes, Iterable[Union[str, bytes]]]) -> list[ov.Output]:
    if isinstance(strings, str):
        return op.Constant(np.frombuffer(strings.encode("utf-8"), dtype=np.uint8)).outputs()

    if isinstance(strings, bytes):
        return op.Constant(np.frombuffer(strings, dtype=np.uint8)).outputs()

    return create_unpacked_string(strings)


# from transformers.models.gpt2.tokenization_gpt2
@lru_cache()
def unicode_to_bytes() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = (chr(n) for n in cs)
    return dict(zip(cs, bs))


def apply_unicode_to_bytes(token: str) -> bytes:
    bytes_encoder = unicode_to_bytes()
    try:
        return bytes(bytes_encoder[char] for char in token)
    except KeyError:
        # tokens that was not bytes-to-chars encoded, remove them from the dictionary
        return token.encode("utf-8")


def parse_bbpe_vocab(tokens: list[str]) -> list[bytes]:
    return [apply_unicode_to_bytes(token) for token in tokens]


def parse_bbpe_config(config: dict[str, Any], inputs: list[ov.Output], node_factory: NodeFactory) -> list[ov.Output]:
    """Parse Byte-level BPE tokenizer configuration"""

    # undo bytes-to-chars encoding
    # todo: share with detokenizer vocab
    vocab = parse_bbpe_vocab(config["tokens"])
    inputs += create_string_constant(vocab)

    merges = [tuple(map(apply_unicode_to_bytes, merge.split(" "))) for merge in config["merges"]]
    left_merges, right_merges = zip(*merges)
    inputs += create_string_constant(left_merges)
    inputs += create_string_constant(right_merges)

    special_tokens = [
        (token, idx)
        for idx, (token, token_type) in enumerate(zip(vocab, config["token_type"]))
        if is_special(token_type)
    ]
    special_tokens, special_tokens_idx = zip(*special_tokens)
    inputs += create_string_constant(special_tokens)
    inputs += op.Constant(np.array(special_tokens_idx, dtype=np.int32)).outputs()

    return (
            node_factory
            .create(
                "BPETokenizer",
                inputs,
                {
                    "unk_token": vocab[config["unknown_token_id"].item()],
                    "fuse_unk": True,
                    "suffix_indicator": "",
                    "end_suffix": "",
                    "byte_fallback": True,
                    "cache_capacity": max(int(len(vocab) * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY),
                },
            )
            .outputs()
        )


tokenizer_node_parser_mapping = {
    "gpt2": parse_bbpe_config,
}

vocab_parser_mapping = {
    "gpt2": parse_bbpe_vocab,
}


def add_ragged_dimension(input_node: list[ov.Output]) -> list[ov.Output]:
    shape = opset.shape_of(input_node[0])
    batch_size = opset.gather(shape, as_node(0), as_node(0))
    ragged_begins = opset.range(as_node(0), batch_size, as_node(1), output_type="i32").outputs()
    ragged_ends = opset.range(
        as_node(1), opset.add(batch_size, make_constant_node(1, Type.i64)), as_node(1), output_type="i32"
    ).outputs()
    return ragged_begins + ragged_ends + input_node


def create_tokenizer_from_config(tokenizer_config: dict[str, Any]) -> tuple[ov.Model, ov.Model]:
    """
    Create OpenVINO tokenizer and detokenizer models from tokenizer configuration
    """
    tokenizer_config = {key.split(".")[-1]: value for key, value in tokenizer_config.items()}
    node_factory = NodeFactory()
    node_factory.add_extension(ov_tokenizers_extension_path)

    # 1 string tensor
    tokenizer_inputs = [ov.op.Parameter(Type.string, ov.PartialShape(["?"]))]

    # 3 tensors: begins[i32], ends[i32], chars[u8]
    outputs = opset.string_tensor_unpack(tokenizer_inputs[0]).outputs()
    # 5 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8]
    outputs = add_ragged_dimension(outputs)

    special_tokens = [
        token
        for token, token_type in zip(tokenizer_config["tokens"], tokenizer_config["token_type"])
        if is_special(token_type)
    ]
    special_tokens_re = "|".join(special_tokens)

    # 6 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8], skips[bool]
    outputs = node_factory.create(
        "SpecialTokensSplit", outputs + create_string_constant(special_tokens_re)
    ).outputs()

    ###############################
    ##### normalization steps #####
    ###############################

    # no normalization steps

    ###############################
    ######### split steps #########
    ###############################

    split_res = split_regex_mapping.get(tokenizer_config["pre"], DEFAULT_BPE_SPLIT_RE)
    for split_re in split_res:
        # 6 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8], skips[bool]
        outputs += create_string_constant(split_re)
        outputs = node_factory.create(
            "RegexSplit",
            outputs,
            {
                "behaviour": split_behaviour_mapping.get(split_re, "isolate"),
                "invert": False,
                "max_splits": -1,
            }
        ).outputs()

    ###############################
    ##### tokenization step ######
    ###############################

    # 3 tensors: begins[i32], ends[i32], token_ids[int32]
    outputs = tokenizer_node_parser_mapping[tokenizer_config["model"]](tokenizer_config, outputs[:5], node_factory)

    ###############################
    #### posttokenization step ####
    ###############################

    # left truncation
    max_length = opset.minimum(
        opset.subtract(outputs[1], outputs[0]),
        make_constant_node(MAX_LENGTH, Type.i32),
    )
    # change begins for the left truncation
    outputs[0] = opset.subtract(outputs[1], max_length).output(0)

    # todo: check for CombineSegments step

    # left padding
    # max length is max length of the batch
    max_length = opset.reduce_max(
        opset.subtract(outputs[1], outputs[0]),
        make_constant_node(0, Type.i32),
    )
    # 2 tensors: input_ids[i32], attention_mask[i32]
    outputs = node_factory.create(
        "RaggedToDense",
        outputs
        + max_length.outputs()
        + make_constant_node(0, Type.i32).outputs(),  # pad_value
        {
            "pad_right": False,
            "pad_max_length": False,
        },
    ).outputs()

    for idx, name in enumerate(["input_ids", "attention_mask"]):
        outputs[idx] = opset.convert(outputs[idx], Type.i64).output(0)
        outputs[idx].tensor.add_names({name})

    tokenizer = ov.Model(outputs, tokenizer_inputs, "tokenizer")

    ###############################
    #### detokenization model #####
    ###############################

    detokenizer_input = op.Parameter(Type.i64, ov.PartialShape(["?", "?"]))

    # vocab decoder
    vocab = vocab_parser_mapping[tokenizer_config["model"]](tokenizer_config["tokens"])
    outputs = detokenizer_input.outputs() + create_string_constant(vocab)

    # special tokens skip
    special_token_ids = make_constant_node(
        np.array([idx for idx, token_type in enumerate(tokenizer_config["token_type"]) if is_special(token_type)]),
        Type.i32,
    )
    stop_const = make_constant_node(np.array([np.iinfo(np.int32).max]), Type.i32)  # make 0 to not skip special tokens
    zero_const = make_constant_node(np.array([0]), Type.i32)
    one_const = make_constant_node(np.array([1]), Type.i32)
    sliced_skips = opset.slice(special_token_ids, zero_const, stop_const, one_const).outputs()
    outputs += sliced_skips

    # 2 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8]
    outputs = node_factory.create("VocabDecoder", outputs).outputs()

    # fuse ragged dimension
    outputs = node_factory.create("FuzeRagged", outputs[:-1], {}).outputs() + outputs[-1:]

    # todo: add clean_up_tokenization_spaces step
    # todo: add utf-8 check

    outputs = opset.string_tensor_pack(*outputs).outputs()

    detokenizer = ov.Model(outputs, [detokenizer_input], "detokenizer")
    detokenizer.output().tensor.add_names({"string_output"})

    # todo: add more relevant rt_info
    for key, value in tokenizer_config.items():
        if key in ("token_type", "tokens", "merges"):
            continue

        try:
            value = value.item()
        except AttributeError:
            pass

        tokenizer.set_rt_info(value, key)
        detokenizer.set_rt_info(value, key)

    return tokenizer, detokenizer