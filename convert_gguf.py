import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np
import openvino as ov
import torch
from openvino.runtime import Model, Type
from openvino.runtime import opset15 as opset
from openvino.runtime import serialize
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from convert_tokenizer import create_tokenizer_from_config

OV_XML_FILE_NAME = "openvino_model.xml"
OV_TOKENIZER_FILE_NAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_FILE_NAME = "openvino_detokenizer.xml"


class QType(Enum):
    FP16 = 1
    INT8 = 2
    INT4 = 3

GGML_QUANTIZATION_GROUP_SIZE = 32


def show_model(m):
    print("inputs of the model:")
    for port, _input in enumerate(m.inputs):
        print("	[{}] {}".format(port, _input))
    print("outputs of the model:")
    for port, _output in enumerate(m.outputs):
        print("	[{}] {}".format(port, _output))


def create_causal_mask(attention_mask, keys, hidden_dim, input_shape):
    # Extract shape of attention mask
    t130 = opset.shape_of(attention_mask, output_type=Type.i64)
    t131 = opset.constant(1, dtype=np.int64)
    t132 = opset.constant(0, dtype=np.int64)
    t133 = opset.gather(t130, t131, t132, batch_dims=0)
    
    # Reshape and construct new shapes
    t134 = opset.constant([1], dtype=np.int64)
    t135 = opset.reshape(t133, t134, special_zero=False)
    t40 = input_shape
    t127 = opset.gather(t40, opset.constant(1, dtype=np.int64), axis=0)
    t129 = opset.reshape(t127, opset.constant([1], dtype=np.int64), special_zero=False) 
    t136 = opset.concat([t129, t135], axis=0)
    min_shape_val = opset.constant([1, 1], dtype=np.int64)
    t137 = opset.maximum(min_shape_val, t136, auto_broadcast="numpy")
    t138 = opset.broadcast(opset.constant(-65504, dtype=np.float32), t137, broadcast_spec="NUMPY")
    
    # Create upper triangular mask for causal masking
    t139 = opset.shape_of(t138, output_type=Type.i32)
    t140 = opset.constant(1, dtype=np.int64)
    t141 = opset.constant(0, dtype=np.int64)
    t142 = opset.gather(t139, t140, t141, batch_dims=0)
    t143 = opset.constant(1, dtype=np.int32)
    
    # Define ranges for the causal mask
    zero_const = opset.constant(0, dtype=np.int32)
    t144 = opset.range(zero_const, t142, t143, output_type=Type.i32)
    t145 = opset.unsqueeze(t144, axes=zero_const)
    t146 = opset.constant(1, dtype=np.int32)
    t147 = opset.constant(0, dtype=np.int64)
    t148 = opset.constant(0, dtype=np.int64)
    
    # Broadcast causal mask
    t149 = opset.gather(t139, t147, t148, batch_dims=0)
    t150 = opset.add(t149, t146, auto_broadcast="numpy")
    t151 = opset.range(t146, t150, t143, output_type=Type.i32)
    t152 = opset.unsqueeze(t151, axes=t143)
    t153 = opset.greater_equal(t145, t152, auto_broadcast="numpy")
    
    # Create a causal mask using a selective operation
    t154 = opset.constant(0.0, dtype=np.float32)
    t155 = opset.select(t153, t138, t154, auto_broadcast="numpy")
    
    # Next branch
    t156 = opset.constant(0, dtype=np.int32)
    t157 = opset.constant(1, dtype=np.int32)
    t158 = opset.range(t156, t133, t157, output_type=Type.f32)
    t159 = opset.convert(t158, destination_type=np.int64)
    t160 = opset.convert(t159, destination_type=np.float32)
    t161 = opset.shape_of(keys, output_type=Type.i64)
    t162 = opset.constant(2, dtype=np.int64)
    t163 = opset.constant(0, dtype=np.int64)
    t164 = opset.gather(t161, t162, t163, batch_dims=0)
    t165 = opset.add(t164, t127, auto_broadcast='numpy')
    t166 = opset.constant(1, dtype=np.int32)
    t167 = opset.range(t164, t165, t166, output_type=Type.f32)
    t168 = opset.constant([-1, 1], dtype=np.int64)
    t169 = opset.reshape(t167, t168, special_zero=False)
    t170 = opset.greater(t160, t169, auto_broadcast='numpy')
    t171 = opset.convert(t170, destination_type=np.float32)
    
    t172 = opset.multiply(t155, t171, auto_broadcast='numpy')
    t173 = opset.constant(0, dtype=np.int64)
    t174 = opset.unsqueeze(t172, t173)
    t48 = opset.constant(1, dtype=np.int64)
    t175 = opset.unsqueeze(t174, t48)
    t41 = opset.constant([0], dtype=np.int64)
    t42 = opset.constant(0, dtype=np.int64)
    t43 = opset.gather(t40, t41, t42, batch_dims=0)
    t176 = opset.constant([1], dtype=np.int64)
    t177 = opset.constant([1], dtype=np.int64)
    t178 = opset.constant([1], dtype=np.int64)
    t179 = opset.concat([t43, t176, t177, t178], axis=0)
    t180 = opset.broadcast(t175, t179, broadcast_spec="bidirectional")
    t181 = opset.constant([-1], dtype=np.int64)
    t182 = opset.reshape(t180, t181, special_zero=False)
    t183 = opset.constant(0, dtype=np.int64)
    t184 = opset.shape_of(t180, output_type=Type.i64)
    t185 = opset.reduce_prod(t184, t183, keep_dims=False)
    t186 = opset.constant(1, dtype=np.int64)
    t187 = opset.range(t183, t185, t186, output_type=Type.i64)
    t188 = opset.reshape(t187, t184, special_zero=False)
    t189 = opset.constant([0], dtype=np.int64)
    t190 = opset.constant([1], dtype=np.int64)
    t191 = opset.slice(t188, t189, t135, t190, hidden_dim)
    t192 = opset.constant([-1, 1], dtype=np.int64)
    t193 = opset.reshape(t191, t192, special_zero=False)
    t194 = opset.constant([0], dtype=np.int64)
    t195 = opset.constant([1], dtype=np.int64)
    t196 = opset.slice(t180, t194, t135, t195, hidden_dim)

    t197 = opset.unsqueeze(attention_mask, t48)
    t198 = opset.constant(2, dtype=np.int64)
    t199 = opset.unsqueeze(t197, t198)
    t200 = opset.convert(t199, destination_type=np.float32)
    t201 = opset.add(t196, t200, auto_broadcast='numpy')
    t202 = opset.constant([[[[0.0]]]], dtype=np.float32)
    t203 = opset.equal(t201, t202, auto_broadcast='numpy')
    t204 = opset.constant(-65504.0, dtype=np.float32)
    t205 = opset.select(t203, t204, t196, auto_broadcast='numpy')
    t206 = opset.shape_of(t196, output_type=Type.i64)
    t207 = opset.broadcast(t205, t206, broadcast_spec="NUMPY")
    t208 = opset.constant([-1], dtype=np.int64)
    t209 = opset.reshape(t207, t208, special_zero=False)
    t210 = opset.scatter_nd_update(t182, t193, t209)
    t211 = opset.reshape(t210, t184, special_zero=False)
    t212 = opset.constant([0], dtype=np.int64)
    t213 = opset.constant([1], dtype=np.int64)
    t214 = opset.reshape(t164, t213, special_zero=False)
    t215 = opset.add(t214, t129, auto_broadcast='numpy')
    t216 = opset.constant([1], dtype=np.int64)
    t217 = opset.slice(t211, t212, t215, t216, hidden_dim)
    return t217


def rotate_half(x, head_size, axis):
    """Rotates half the hidden dimensions of the input tensor."""
    t58 = opset.constant([head_size // 2], dtype=np.int64)
    t59 = opset.constant([9223372036854775807], dtype=np.int64)
    t60 = opset.constant([1], dtype=np.int64)
    t62 = opset.slice(x, t58, t59, t60, axis)
    t63 = opset.constant([[[[-1.0]]]], dtype=np.float32)
    t64 = opset.multiply(t62, t63)
    t65 = opset.constant([0], dtype=np.int64)
    t66 = opset.constant([head_size // 2], dtype=np.int64)
    t67 = opset.constant([1], dtype=np.int64)
    t68 = opset.slice(x, t65, t66, t67, axis)
    rotated = opset.concat([t64, t68], axis=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin, head_size, hidden_dim, cos_sin_cached, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors using OpenVINO API.

    Args:
        q: Query tensor (OpenVINO node).
        k: Key tensor (OpenVINO node).
        cos: Cosine part of rotary embedding (OpenVINO node).
        sin: Sine part of rotary embedding (OpenVINO node).
        unsqueeze_dim: Dimension to unsqueeze cos and sin for broadcasting.
    Returns:
        Tuple of (q_rotated, k_rotated) tensors.
    """
    # Unsqueeze cos and sin along the specified dimension
    if cos_sin_cached is None:
        cos_unsqueezed = opset.unsqueeze(cos, np.int64(unsqueeze_dim))
        sin_unsqueezed = opset.unsqueeze(sin, np.int64(unsqueeze_dim))
    else:
        cos_unsqueezed = cos_sin_cached[0]
        sin_unsqueezed = cos_sin_cached[1]

    # Apply Rotary Positional Embedding
    q_rotated = opset.add(opset.multiply(q, cos_unsqueezed), opset.multiply(rotate_half(q, head_size, hidden_dim), sin_unsqueezed))
    k_rotated = opset.add(opset.multiply(k, cos_unsqueezed), opset.multiply(rotate_half(k, head_size, hidden_dim), sin_unsqueezed))

    return q_rotated, k_rotated, (cos_unsqueezed, sin_unsqueezed)


def rope_emb(x, rope_const, position_ids, batch_dim):
    """
    Generates Rotary Position Embedding (RoPE) cosine and sine components using OpenVINO.

    Args:
        x: The input tensor to determine the device and dtype (OpenVINO node).
        rope_const: Tensor containing the rotary embedding constants (OpenVINO node).
        position_ids: Tensor with position IDs (OpenVINO node).
        batch_dim: batch dimension

    Returns:
        cos: Cosine component of the rotary embedding.
        sin: Sine component of the rotary embedding.
    """
    # Expand dimensions for broadcasting
    position_ids_expanded = opset.convert(opset.unsqueeze(position_ids, 1), Type.f32)  # Add head_dim axis: [batch_size, 1, seq_len]

    # Broadcast RoPE to batch dimension
    inv_freq_expanded = opset.broadcast(rope_const,
                                        target_shape=opset.concat([batch_dim,
                                                      np.int64([1]),
                                                      np.int64([1])], axis=0),
                                        broadcast_spec="BIDIRECTIONAL")
    # Compute frequencies
    freqs = opset.matmul(inv_freq_expanded, position_ids_expanded, transpose_a=False, transpose_b=False)  # Shape: [batch_size, seq_len, head_dim]
    freqs_transposed = opset.transpose(freqs, np.int32([0, 2, 1]))  # Transpose to shape: [batch_size, head_dim, seq_len]

    # Concatenate frequencies along the last dimension
    emb = opset.concat([freqs_transposed, freqs_transposed], axis=-1)  # Shape: [batch_size, head_dim, seq_len * 2]

    # Compute cosine and sine
    cos = opset.cos(emb)
    sin = opset.sin(emb)

    return cos, sin


def multi_head_attention(query, key, value,
                        configs,
                        batch_dim,
                        layer_idx,
                        hidden_dim,
                        input_shape,
                        output_shape,
                        attention_mask,
                        mask=None,
                        position_ids=None,
                        rope_const=None,
                        beam_idx=None,
                        cos_sin_cached=None):
    """
    Implements multi-head self-attention using OpenVINO operations
    
    Args:
        query: Query tensor of shape [batch_size, seq_len, hidden_dim]
        key: Key tensor of shape [batch_size, seq_len, hidden_dim]
        value: Value tensor of shape [batch_size, seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        mask: Optional attention mask
        sin: Optional sine component for rotary embeddings
        cos: Optional cosine component for rotary embeddings
        opset: OpenVINO operation set to use
    """
    num_heads = configs["head_num"]
    head_dim = configs["head_size"]
    num_heads_kv = configs["head_num_kv"]
    
    # 1. Reshape Q, K, V to split heads
    def split_heads(x, num_heads, head_dim):
        # Reshape from [batch, seq_len, hidden_dim] to [batch, seq_len, num_heads, head_dim]
        x = opset.reshape(x, [0, 0, num_heads, head_dim], special_zero=True)
        # Transpose to [batch, num_heads, seq_len, head_dim]
        x = opset.transpose(x, [0, 2, 1, 3])
        return x
    
    q = split_heads(query, num_heads, head_dim)
    k = split_heads(key, num_heads_kv, head_dim)
    v = split_heads(value, num_heads_kv, head_dim)

    cos = None
    sin = None
    if cos_sin_cached is None:
        cos, sin = rope_emb(v, rope_const, position_ids, batch_dim)
    
    # 2. Apply rotary embeddings
    q, k, cos_sin_cached = apply_rotary_pos_emb(q, k, cos, sin, head_dim, hidden_dim, cos_sin_cached)

    default_key = opset.broadcast(opset.constant(0.0, dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads_kv]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    k_cache_val = opset.read_value(default_key, 
                                   variable_shape=ov.PartialShape([-1,num_heads_kv,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    k_cache = opset.gather(k_cache_val, beam_idx, opset.constant(0, dtype=np.int64), batch_dims=0)
    default_value = opset.broadcast(opset.constant(0.0, dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads_kv]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    v_cache_val = opset.read_value(default_value, 
                                   variable_shape=ov.PartialShape([-1,num_heads_kv,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")
    v_cache = opset.gather(v_cache_val, beam_idx, opset.constant(0, dtype=np.int64), batch_dims=0)

    k_combined = opset.concat([k_cache, k], axis=2)
    v_combined = opset.concat([v_cache, v], axis=2)

    k_assigned = opset.assign(k_combined, f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    v_assigned = opset.assign(v_combined, f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")

    if num_heads == num_heads_kv:
        k_reshaped = k_combined
        v_reshaped = v_combined
    else: # Group Query Attention branch
        kv_per_head = num_heads // num_heads_kv
        k_combined_unsq = opset.unsqueeze(k_combined, opset.constant(2, dtype=np.int64))
        k_combined_broad = opset.broadcast(k_combined_unsq,
                                           opset.concat([batch_dim,
                                                np.int64([num_heads_kv]),
                                                np.int64([kv_per_head]),
                                                np.int64([0]),
                                                np.int64([head_dim])],
                                                axis=0),
                                            broadcast_spec="BIDIRECTIONAL")
        k_reshaped = opset.reshape(k_combined_broad, [0, num_heads, -1, head_dim], special_zero=True)

        v_combined_unsq = opset.unsqueeze(v_combined, opset.constant(2, dtype=np.int64))
        v_combined_broad = opset.broadcast(v_combined_unsq,
                                           opset.concat([batch_dim,
                                                np.int64([num_heads_kv]),
                                                np.int64([kv_per_head]),
                                                np.int64([0]),
                                                np.int64([head_dim])],
                                                axis=0),
                                            broadcast_spec="BIDIRECTIONAL")
        v_reshaped = opset.reshape(v_combined_broad, [0, num_heads, -1, head_dim], special_zero=True)

    if mask is None:
        mask = create_causal_mask(attention_mask, k_cache, hidden_dim, input_shape)

    # 3. Calculate attention
    #scale = opset.constant(np.float32(1.0 / np.sqrt(head_dim)))
    
    attention_output = opset.scaled_dot_product_attention(q, k_reshaped, v_reshaped, mask)
    
    # 4. Reshape output
    # Transpose back to [batch, seq_len, num_heads, head_dim]
    context_transposed = opset.transpose(attention_output, [0, 2, 1, 3])
    
    # Combine heads: [batch, seq_len, hidden_dim]
    output = opset.reshape(context_transposed,
                          output_shape,
                          special_zero=False)
    
    return output, [k_assigned, v_assigned], cos_sin_cached, mask


# Reorder weight rows from 0,2,1,3 -> 0,1,2,3 for each head
def reorder_interleaved_format(weights, head_size):
    shape = weights.shape
    num_heads = shape[0] // head_size
    weights = weights.reshape((num_heads, head_size, -1))
    weights = np.moveaxis(weights, 0, 1)
    new_weight = np.empty_like(weights)
    new_weight[0:head_size // 2] = weights[0::2]
    new_weight[(head_size//2):head_size] = weights[1::2]
    new_weight = np.moveaxis(new_weight, 0, 1)
    return new_weight.reshape(shape)


def make_fp16_weights(key, consts, reorder, head_size):
    weight = consts[f"{key}.weight"]
    if reorder:
        weight = reorder_interleaved_format(weight, head_size)

    weights = opset.constant(weight, dtype=np.float16)
    weights.set_friendly_name(name=f"{key}.weight")
    w_f32 = opset.convert(weights, Type.f32)
    return w_f32


def make_int8_weights(key, consts, reorder, head_size):#weight = ov.Tensor(weight, weight.shape, const_dtype)
    weight = consts[f"{key}.weight"]
    weight = weight.view(np.uint8)
    orig_weight_shape = list(weight.shape)
    weight = weight.reshape(orig_weight_shape[0], -1, GGML_QUANTIZATION_GROUP_SIZE)
    scale = np.expand_dims(consts[f"{key}.scales"], axis=2)
    bias = np.expand_dims(consts[f"{key}.biases"], axis=2)
    if reorder:
        weight = reorder_interleaved_format(weight, head_size)
        scale = reorder_interleaved_format(scale, head_size)
        bias = reorder_interleaved_format(bias, head_size)

    weights = opset.constant(weight, dtype=np.uint8)
    weights.set_friendly_name(name=f"{key}.weight")
    weights_f16 = opset.convert(weights, Type.f16)

    zero_point = (-bias / scale).astype(np.uint8)
    zero_points = opset.constant(zero_point, dtype=np.uint8)
    zero_points_f16 = opset.convert(zero_points, Type.f16)

    scales = opset.constant(scale, dtype=np.float16)

    w_zp = opset.subtract(weights_f16, zero_points_f16, auto_broadcast="numpy")
    w_zp_s = opset.multiply(w_zp, scales, auto_broadcast="numpy")

    w_zp_s_r = opset.reshape(w_zp_s, opset.constant(orig_weight_shape, dtype=np.int64), special_zero=False)
    w_zp_s_f32 = opset.convert(w_zp_s_r, Type.f32)
    return w_zp_s_f32


def make_int4_weights(key, consts, reorder, head_size):#
    weight = consts[f"{key}.weight"]
    weight = weight.view(np.uint8)
    orig_weight_shape = list(weight.shape)
    orig_weight_shape[1] = orig_weight_shape[1] * 2 # double number of columns as it is 4-bit tensor

    weight = weight.reshape(orig_weight_shape[0], -1, GGML_QUANTIZATION_GROUP_SIZE//2)
    scale = np.expand_dims(consts[f"{key}.scales"], axis=2)
    bias = np.expand_dims(consts[f"{key}.biases"], axis=2)

    if reorder:
        weight = reorder_interleaved_format(weight, head_size)
        scale = reorder_interleaved_format(scale, head_size)
        bias = reorder_interleaved_format(bias, head_size)

    shape = (orig_weight_shape[0], orig_weight_shape[1]//GGML_QUANTIZATION_GROUP_SIZE, GGML_QUANTIZATION_GROUP_SIZE)
    weight_tensor = ov.Tensor(weight.reshape(-1), shape, Type.u4)
    weights = opset.constant(weight_tensor, name=f"{key}.weight", shared_memory=False) # Don't use shared_memory=True
    weights_f16 = opset.convert(weights, Type.f16)

    zero_point = (-bias / scale).astype(np.uint8)
    zero_point_shape = list(zero_point.shape)
    zero_point = zero_point.reshape(-1)
    # Pack zero points: two subsequent values into one
    mask = np.array(0b00001111, dtype=np.uint8)
    zero_point_packed = (zero_point[1::2] << 4) | (zero_point[0::2] & mask)
    zero_point_tensor = ov.Tensor(zero_point_packed, tuple(zero_point_shape), Type.u4)
    zero_points = opset.constant(zero_point_tensor, shared_memory=False) # Don't use shared_memory=True
    zero_points_f16 = opset.convert(zero_points, Type.f16)

    scales = opset.constant(scale, dtype=np.float16)

    w_zp = opset.subtract(weights_f16, zero_points_f16, auto_broadcast="numpy")
    w_zp_s = opset.multiply(w_zp, scales, auto_broadcast="numpy")

    w_zp_s_r = opset.reshape(w_zp_s, opset.constant(orig_weight_shape, dtype=np.int64), special_zero=False)
    w_zp_s_f32 = opset.convert(w_zp_s_r, Type.f32)
    return w_zp_s_f32


def make_weights_subgraph(key, consts, qtype, reorder, head_size):
    if qtype == QType.FP16:
        final_node = make_fp16_weights(key, consts, reorder, head_size)
    elif qtype == QType.INT8:
        final_node = make_int8_weights(key, consts, reorder, head_size)
    elif qtype == QType.INT4:
        final_node = make_int4_weights(key, consts, reorder, head_size)
    else:
        raise ValueError("Unsupported quantization type:")
    
    return final_node


def make_fc(key, input, consts, qtype, reorder=False, head_size=-1):
    # weight const f32 NxK
    w_f32 = make_weights_subgraph(key, consts, qtype, reorder, head_size)
    matmul = opset.matmul(input, w_f32, transpose_a=False, transpose_b=True)
    return matmul


def make_lm_head(key, input, consts, embeddings_node, qtype):
    if consts.get(f"{key}.weight", None) is not None:
        if consts.get(f"{key}.scales", None) is not None:
            lm_head_qtype = qtype
        else:
            lm_head_qtype = QType.FP16
        w_f32 = make_weights_subgraph(key, consts, lm_head_qtype, False, -1)
    else:
        w_f32 = embeddings_node # shared weights with embeddings
    return opset.matmul(input, w_f32, transpose_a=False, transpose_b=True)


def make_mvn(key, input, consts, configs, name_suffix=""):
    mvn = opset.mvn(input, axes=[-1], normalize_variance=True, eps=configs["layer_norm_eps"], eps_mode="inside_sqrt", name=f"{key}.mvn{name_suffix}")
    if consts[f"{key}.weight"] is not None:
        weights = opset.constant(consts[f"{key}.weight"], Type.f16, name=f"{key}.weight{name_suffix}")
        mvn = opset.multiply(mvn, weights, auto_broadcast="numpy", name=f"{key}.mul{name_suffix}")
    if consts[f"{key}.bias"] is not None:
        bias = opset.constant(consts[f"{key}.bias"], Type.f16, name=f"{key}.bias{name_suffix}")
        mvn = opset.add(mvn, bias, auto_broadcast="numpy", name=f"{key}.add{name_suffix}")
    return mvn


def make_rms_norm(key, input, consts, epsilon):
    epsilon_c = opset.constant(np.float16([[[epsilon]]]))
    pow = opset.power(input, opset.convert(np.array([[[2]]], np.float16), Type.f32))
    variance = opset.reduce_mean(pow, opset.constant([-1], dtype=np.int64), keep_dims=True)
    add = opset.add(variance, opset.convert(epsilon_c, Type.f32))
    sqrt = opset.sqrt(add)
    div = opset.divide(opset.convert(opset.constant([[[1]]], dtype=np.float16), Type.f32), sqrt)
    mul = opset.multiply(div, input, auto_broadcast="numpy")
    if not np.all(consts[f"{key}.weight"] == 1.0):
        weights = opset.convert(opset.constant(consts[f"{key}.weight"].reshape((1, 1, -1)), np.float16), Type.f32)
        mul = opset.multiply(mul, weights, auto_broadcast="numpy")

    return mul


def make_embedding(key, input, consts, qtype):
    if consts.get(f"{key}.scales", None) is not None:
        embedding_type = qtype
    else:
        embedding_type = QType.FP16
    embed_f32 = make_weights_subgraph(key, consts, embedding_type, False, -1)
    input_int32 = opset.convert(input, Type.i32)
    inputs_embeds = opset.gather(embed_f32, indices=input_int32, axis=0)
    return inputs_embeds, embed_f32


def save_original_tokenizer(orig_model_path, ov_model_path):
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    tokenizer.save_pretrained(ov_model_path)


def layer(configs, consts, layer_idx, hidden_states, attn_mask, causal_mask, position_ids, rope_const, beam_idx, batch_dim, hidden_dim, cos_sin_cached, output_shape):
    name_suffix = f".layer{layer_idx}"
    name_prefix = "model.layers.self_attn"
    # layerNorm operation
    input_layernorm = make_rms_norm("model.layers.input_layernorm", hidden_states, consts["layers"][layer_idx], configs["rms_norm_eps"])

    q = make_fc("model.layers.self_attn.q_proj", input_layernorm, consts["layers"][layer_idx], configs["qtype"], True, configs["head_size"])
    k = make_fc("model.layers.self_attn.k_proj", input_layernorm, consts["layers"][layer_idx], configs["qtype"], True, configs["head_size"])
    v = make_fc("model.layers.self_attn.v_proj", input_layernorm, consts["layers"][layer_idx], configs["qtype"])

    input_shape = opset.shape_of(input_layernorm)
    if output_shape is None:
        output_shape = opset.concat([opset.gather(input_shape,
                                                 opset.constant([0, 1], dtype=np.int64),
                                                 axis=0,
                                                 batch_dims=0),
                                    opset.constant([-1], dtype=np.int64)],
                                    axis=0)


    attn_output, sinks, cos_sin_cached, causal_mask = multi_head_attention(q, k, v,
                        configs,
                        batch_dim=batch_dim,
                        layer_idx=layer_idx,
                        hidden_dim=hidden_dim,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        attention_mask=attn_mask,
                        mask=causal_mask,
                        position_ids=position_ids,
                        rope_const=rope_const,
                        beam_idx=beam_idx,
                        cos_sin_cached=cos_sin_cached)

    attn_output = make_fc("model.layers.self_attn.o_proj", attn_output, consts["layers"][layer_idx], configs["qtype"])

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast="numpy", name=f"{name_prefix}.add0{name_suffix}")
    post_attention_layernorm = make_rms_norm("model.layers.post_attention_layernorm", attn_output, consts["layers"][layer_idx], configs["rms_norm_eps"])

    # mlp
    def mlp(states):
        gate_proj = make_fc("model.layers.mlp.gate_proj", states, consts["layers"][layer_idx], configs["qtype"])
        silu = opset.swish(gate_proj)
        up_proj = make_fc("model.layers.mlp.up_proj", states, consts["layers"][layer_idx], configs["qtype"])
        mul = opset.multiply(silu, up_proj, auto_broadcast="numpy", name=f"{name_prefix}.mlp.mul{name_suffix}")
        down_proj = make_fc("model.layers.mlp.down_proj", mul, consts["layers"][layer_idx], configs["qtype"])
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f"{name_prefix}.add1{name_suffix}")
    return output, sinks, causal_mask, cos_sin_cached, output_shape


def init_rope(head_dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
    inv_freq = inv_freq[None, :, None]
    # For BC we register cos and sin cached
    rope_const = opset.constant(inv_freq.numpy(), Type.f32)
    return rope_const


def create_model(configs, consts):
    print("start generate ov model...")
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    # [batch, query_len+past_len]
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    # [batch, query_len+past_len]
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    # [batch, max_kv_len]
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    inputs_embeds, embeddings = make_embedding("model.embed_tokens", input_ids, consts, configs["qtype"])
    hidden_states = inputs_embeds

    rope_const = init_rope(configs["head_size"], configs["max_position_embeddings"], configs["rope_freq_base"])

    input_shape = opset.shape_of(input_ids)
    batch_size = opset.gather(input_shape,
                         opset.constant([0], dtype=np.int64),
                         opset.constant([0], dtype=np.int64))
    hidden_dim = opset.constant([3], dtype=np.int64)

    sinks = []
    # Shared tensors across all the Transformer blocks
    causal_mask = None
    cos_sin_cached = None
    output_shape = None
    for i in tqdm(range(configs["layer_num"])):
        hidden_states, layer_sinks, causal_mask, cos_sin_cached, output_shape = layer(configs, consts, i, hidden_states, attention_mask, causal_mask, position_ids, rope_const, beam_idx, batch_size, hidden_dim, cos_sin_cached, output_shape)
        sinks = sinks + layer_sinks
    # final_layernorm
    final_layernorm = make_rms_norm("model.norm", hidden_states, consts, configs["rms_norm_eps"])
    # embed_out
    embed_out = make_lm_head("lm_head", final_layernorm, consts, embeddings, configs["qtype"])

    logits = opset.result(embed_out, name="logits")
    logits.set_friendly_name("logits")
    cost = time.time() - beg
    print(f"generate ov model done, cost {cost:.2f} seconds.")
    model = Model([logits], sinks,
                 [input_ids, attention_mask, position_ids, beam_idx])
    model.outputs[0].get_tensor().set_names({"logits"})

    # set runtime options
    model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
    model.set_rt_info("8.0", ["runtime_options", "ACTIVATIONS_SCALE_FACTOR"])

    return model


def get_quantizaiton_type(gguf_type):
    if gguf_type == 0 or gguf_type == 1:
        qtype = QType.FP16
        print("Working with FP16 model")
    elif gguf_type == 2 or gguf_type == 3:
        # MOSTLY_Q4_0 or MOSTLY_Q4_1
        qtype = QType.INT4
        # print bits value
        print("Working with INT4 quantized model")
    elif gguf_type == 7:
        # MOSTLY_Q8_0 = 7
        qtype = QType.INT8
        print("Working with INT8 quantized model")
    else:
        qtype = None
        raise ValueError("Using unsupported GGUF quantization")
    return qtype


def load_gguf_model(model_path: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Extract configurations and weights from GGUF model"""
    print(f"extracting from GGUF model '{model_path}'...")
    beg = time.time()

    # Load GGUF model
    weights, metadata = mx.load(model_path, return_metadata=True)

    print("Metadata:\n", metadata.keys())

    try:
        url_parts = metadata["general.source.url"].split("/")
        model_id = f"{url_parts[-2]}/{url_parts[-1]}"
    except Exception:
        print("Cannot get model_id to get the config.json and tokenizer")
        model_id = None

    config = {
        "layer_num": metadata["llama.block_count"].item(),
        "head_num": metadata["llama.attention.head_count"].item(),
        "head_size": metadata["llama.embedding_length"].item() // metadata["llama.attention.head_count"].item(),
        "head_num_kv": metadata.get("llama.attention.head_count_kv", metadata["llama.attention.head_count"]).item(),
        "hidden_size": metadata["llama.embedding_length"].item(),
        "max_position_embeddings": metadata.get("llama.context_length", np.int32([2048])).item(),
        "rotary_dims": metadata["llama.rope.dimension_count"].item(),
        "rms_norm_eps": metadata["llama.attention.layer_norm_rms_epsilon"].item(),
        "rope_freq_base": metadata.get("llama.rope.freq_base", np.float32(10000)).item(),
        "qtype": get_quantizaiton_type(int(metadata["general.file_type"])),
        "model_id": model_id,        
    }

    print("Config:\n", config)

    # Extract weights and biases
    print("Extract weights and biases")
    print("============= Weight keys ============")
    print(list(weights.keys()))
    consts = {
        "model.embed_tokens.weight": np.array(weights["token_embd.weight"]),
        "model.norm.weight": np.array(weights["output_norm.weight"]),
        "lm_head.weight": np.array(weights["output.weight"]) if weights.get("output.weight", None) is not None else None,
        "layers": []
    }
    if weights.get("token_embd.scales", None) is not None:
        consts["model.embed_tokens.scales"] = np.array(weights["token_embd.scales"])
        consts["model.embed_tokens.biases"] = np.array(weights["token_embd.biases"])

    if weights.get("output.scales", None) is not None:
        consts["lm_head.scales"] = np.array(weights["output.scales"])
        consts["lm_head.biases"] = np.array(weights["output.biases"])


    # w = np.array(weights["blk.15.attn_q.weight"])
    # s = np.array(weights["blk.15.attn_q.scales"])
    # b = np.array(weights["blk.15.attn_q.biases"])

    # print("blk.15.attn_q.weight:", w.shape)
    # print("blk.15.attn_q.scales:", s.shape)
    # print("blk.15.attn_q.biases:", b.shape)
    # print("blk.15.attn_q.weight:", w.dtype)
    # print("blk.15.attn_q.scales:", s.dtype)
    # print("blk.15.attn_q.biases:", b.dtype)

    # print("blk.15.attn_q.scales: ", s)
    # print("blk.15.attn_q.biases: ", b)
    # print("ratio: ", b/s)
    
    # Extract layer weights
    print("Extract layer weights")
    for i in range(config["layer_num"]):
        layer_weights = {
            "model.layers.input_layernorm.weight": np.array(weights[f"blk.{i}.attn_norm.weight"]),
            "model.layers.post_attention_layernorm.weight": np.array(weights[f"blk.{i}.ffn_norm.weight"]),
            "model.layers.self_attn.q_proj.bias": None,
            "model.layers.self_attn.q_proj.weight": np.array(weights[f"blk.{i}.attn_q.weight"]),
            "model.layers.self_attn.k_proj.bias": None,
            "model.layers.self_attn.k_proj.weight": np.array(weights[f"blk.{i}.attn_k.weight"]),
            "model.layers.self_attn.v_proj.bias": None,
            "model.layers.self_attn.v_proj.weight": np.array(weights[f"blk.{i}.attn_v.weight"]),
            "model.layers.self_attn.o_proj.bias": None,
            "model.layers.self_attn.o_proj.weight": np.array(weights[f"blk.{i}.attn_output.weight"]),
            "model.layers.mlp.gate_proj.bias": None,
            "model.layers.mlp.gate_proj.weight": np.array(weights[f"blk.{i}.ffn_gate.weight"]),
            "model.layers.mlp.up_proj.bias": None,
            "model.layers.mlp.up_proj.weight": np.array(weights[f"blk.{i}.ffn_up.weight"]),
            "model.layers.mlp.down_proj.bias": None,
            "model.layers.mlp.down_proj.weight": np.array(weights[f"blk.{i}.ffn_down.weight"])
        }
        if config["qtype"] != QType.FP16:
            q_weights = {
                "model.layers.self_attn.q_proj.scales": np.array(weights[f"blk.{i}.attn_q.scales"]),
                "model.layers.self_attn.k_proj.scales": np.array(weights[f"blk.{i}.attn_k.scales"]),
                "model.layers.self_attn.v_proj.scales": np.array(weights[f"blk.{i}.attn_v.scales"]),
                "model.layers.self_attn.o_proj.scales": np.array(weights[f"blk.{i}.attn_output.scales"]),
                "model.layers.mlp.gate_proj.scales": np.array(weights[f"blk.{i}.ffn_gate.scales"]),
                "model.layers.mlp.up_proj.scales": np.array(weights[f"blk.{i}.ffn_up.scales"]),
                "model.layers.mlp.down_proj.scales": np.array(weights[f"blk.{i}.ffn_down.scales"]),

                "model.layers.self_attn.q_proj.biases": np.array(weights[f"blk.{i}.attn_q.biases"]),
                "model.layers.self_attn.k_proj.biases": np.array(weights[f"blk.{i}.attn_k.biases"]),
                "model.layers.self_attn.v_proj.biases": np.array(weights[f"blk.{i}.attn_v.biases"]),
                "model.layers.self_attn.o_proj.biases": np.array(weights[f"blk.{i}.attn_output.biases"]),
                "model.layers.mlp.gate_proj.biases": np.array(weights[f"blk.{i}.ffn_gate.biases"]),
                "model.layers.mlp.up_proj.biases": np.array(weights[f"blk.{i}.ffn_up.biases"]),
                "model.layers.mlp.down_proj.biases": np.array(weights[f"blk.{i}.ffn_down.biases"])
            }
            layer_weights = {**layer_weights, **q_weights}

        consts["layers"].append(layer_weights)
    
    cost = time.time() - beg

    tokenizer_config = {key.split(".")[-1]: value for key, value in metadata.items() if key.startswith("tokenizer")}

    print(f"extracting done, cost {cost:.2f} seconds.\nmodel configs:")
    for k, v in config.items():
        print(f"{k}: {v}")

    return config, consts, tokenizer_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--org_model_path", type=str, default="Model ID (can be a Hugginface Hub id, or a local directory)")
    parser.add_argument("--ov_model_path", type=str, nargs="?", default="./gen/llama-2-7b-chat/")
    parser.add_argument("--model_id", type=str, nargs="?", default=None)
    args = parser.parse_args()

    os.makedirs(args.ov_model_path, exist_ok=True)

    config, consts, tokenizer_config = load_gguf_model(args.org_model_path)
    model = create_model(config, consts)
    show_model(model)

    print(f"serialize ov model to '{args.ov_model_path}'...")
    beg = time.time()
    serialize(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME))
    cost = time.time() - beg
    print(f"serialize done, cost {cost:.2f} seconds.")

    print("create tokenizer and detokenizer...")
    beg = time.time()
    tokenizer, detokenizer = create_tokenizer_from_config(tokenizer_config)
    # todo: check chat_template
    cost = time.time() - beg
    print(f"create tokenizer and detokenizer done, cost {cost:.2f} seconds.")
    print("tokenizer:")
    show_model(tokenizer)
    print("detokenizer:")
    show_model(detokenizer)

    print(f"serialize ov tokenizer and detokenizer to '{args.ov_model_path}'...")
    beg = time.time()
    serialize(tokenizer, os.path.join(args.ov_model_path, OV_TOKENIZER_FILE_NAME))
    serialize(detokenizer, os.path.join(args.ov_model_path, OV_DETOKENIZER_FILE_NAME))
    cost = time.time() - beg
    print(f"serialize done, cost {cost:.2f} seconds.")

    # save tokenizer and config to load with GenAI and Optimum
    model_id = args.model_id or config["model_id"]  # "HuggingFaceTB/SmolLM2-135M" #"meta-llama/Llama-2-7b-chat-hf"
    if model_id:
        print(f"save original tokenzier to '{args.ov_model_path}' ...")
        save_original_tokenizer(model_id, args.ov_model_path)
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(args.ov_model_path)
    else:
        print("[WARNING]: config.json were not saved because model_id was not found or provided as an option.")