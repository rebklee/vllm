from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils import is_navi
import aiter

if HAS_TRITON:
    from vllm.attention.ops.prefix_prefill import context_attention_fwd

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512 if not current_platform.is_rocm() or is_navi() else 1024


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        k_scale,k_scale_per_tensor=k_scale
        v_scale,v_scale_per_tensor=v_scale
        if key_cache.dtype.itemsize == 1:
            # if "fp8" in kv_cache_dtype:
            #     key_cache = key_cache.view(torch.float8_e4m3fnuz)
            #     value_cache = value_cache.view(torch.float8_e4m3fnuz)
            # else:
            key_cache = key_cache.view(torch.int8)
            value_cache = value_cache.view(torch.int8)
            dtype=key.dtype
            aiter.reshape_and_cache_with_pertoken_quant(
                key,
                value,
                key_cache,
                value_cache,
                k_scale,
                v_scale,
                slot_mapping.flatten(),
                True
            )
        else:
            aiter.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping.flatten(),
                kv_cache_dtype,
                k_scale_per_tensor,
                v_scale_per_tensor,
                True
            )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
        out=None
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.

        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        if kv_cache_dtype not in ["int8", "fp8", "fp8_e4m3"]:
            k_scale, v_scale = (None, None)
            query = query.contiguous()
        elif "fp8" in kv_cache_dtype:
            # key_cache = key_cache.view(torch.float8_e4m3fnuz)
            # value_cache = value_cache.view(torch.float8_e4m3fnuz)

            # num_kv_heads=value_cache.shape[1]
            # k_scale_ = torch.empty((num_kv_heads, max_num_blocks_per_seq * block_size), 
            #                 dtype=torch.float32, device=key_cache.device)
            # v_scale_ = torch.empty((num_kv_heads, max_num_blocks_per_seq * block_size), 
            #                           dtype=torch.float32, device=key_cache.device)
            # k_scale_.fill_(k_scale.item())
            # v_scale_.fill_(v_scale.item())
            # k_scale=k_scale_
            # v_scale=v_scale_
            k_scale,_=k_scale
            v_scale,_=v_scale
            key_cache = key_cache.view(torch.int8)
            value_cache = value_cache.view(torch.int8)
        # def asm_V_shuffle(VC):
        #     # [num_blocks, num_kv_heads, head_size, block_size]
        #     x = 16//VC.element_size()
        #     num_blocks, num_kv_heads, head_size, block_size = VC.shape
        #     VC = VC.view(num_blocks, num_kv_heads, head_size, block_size//x, x)
        #     # [num_blocks, num_kv_heads, block_size/X, head_size, X]
        #     VC = VC.permute(0, 1, 3, 2, 4).contiguous()
        #     return VC
        aiter.pa_fwd_asm(query, key_cache, value_cache, block_tables, seq_lens,
                         max_num_blocks_per_seq, k_scale, v_scale, out)
        return out

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
