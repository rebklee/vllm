# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_fwd
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


def is_aiter_mla_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER \
        and envs.VLLM_ROCM_USE_AITER_MLA


class AiterMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "AITER_MLA"

    @staticmethod
    def get_impl_cls() -> Type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AiterMLAMetadata"]:
        return AiterMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["AiterMLAState"]:
        return AiterMLAState


@dataclass
class AiterMLAMetadata(MLACommonMetadata):
    # The following 4 tensors are for current version of AITER MLA
    block_table_bound: Optional[torch.Tensor] = None
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_lens: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self):
        prefill_metadata = super().prefill_metadata
        self._cached_prefill_metadata = prefill_metadata

        if prefill_metadata is not None:
            prefill_metadata.paged_kv_indptr = self.paged_kv_indptr
            prefill_metadata.paged_kv_indices = self.paged_kv_indices
            prefill_metadata\
                .paged_kv_last_page_lens = self.paged_kv_last_page_lens
            prefill_metadata.block_table_bound = self.block_table_bound

            self._cached_prefill_metadata = self.__class__(
                **prefill_metadata.__dict__)

        return self._cached_prefill_metadata

    @property
    def decode_metadata(self):
        decode_metadata = super().decode_metadata

        self._cached_decode_metadata = decode_metadata

        if decode_metadata is not None:
            decode_metadata.paged_kv_indptr = self.paged_kv_indptr
            decode_metadata.paged_kv_indices = self.paged_kv_indices
            decode_metadata\
                .paged_kv_last_page_lens = self.paged_kv_last_page_lens
            decode_metadata.block_table_bound = self.block_table_bound

            self._cached_decode_metadata = self.__class__(
                **decode_metadata.__dict__)

        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        self.advance_step_assertions(
            num_seqs=num_seqs,
            num_queries=num_queries,
            turn_prefills_into_decodes=turn_prefills_into_decodes)

        ops.advance_step_flashinfer(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_last_page_lens=self.paged_kv_last_page_lens,
            block_table_bound=self.block_table_bound)


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        super().__init__(input_builder)
        assert self.block_size == 1, "AITER MLA requires only block size 1."

    def prepare(self):
        super().prepare()
        self.paged_kv_indices: list[int] = []
        self.paged_kv_indptr: list[int] = [0]
        self.paged_kv_last_page_lens: list[int] = []
        self.total_blocks = 0

    def _add_seq_group(self, inter_data, chunked_prefill_enabled: bool,
                       prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block, input_positions) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks,
                 inter_data.input_positions):
            self.input_positions.extend(input_positions)
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)
            if is_profile_run:
                return

            # Update paged_kv_* tensors only for non-profile run
            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: list[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = seq_len // self.block_size + 1 \
            if seq_len % self.block_size != 0 \
            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_lens.append(last_page_len)

    def get_block_tables_with_captured_graph(self, cuda_graph_pad_size: int,
                                             num_seqs: int) -> torch.Tensor:
        self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
        self.block_tables.extend([[]] * cuda_graph_pad_size)
        block_tables = self._get_graph_runner_block_tables(
            num_seqs, self.block_tables)

        return block_tables

    def build(self, seq_lens: list[int], query_lens: list[int],
              cuda_graph_pad_size: int, batch_size: int) -> AiterMLAMetadata:
        metadata = super().build(seq_lens, query_lens, cuda_graph_pad_size,
                                 batch_size)
        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        if use_captured_graph:
            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_lens.extend([0] * cuda_graph_pad_size)

        # For current version of AITER MLA
        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            self.paged_kv_indices.extend(
                [0] * (self.total_blocks - len(self.paged_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device=device,
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device=device,
                                                  dtype=torch.int)
            paged_kv_last_page_lens_tensor = torch.tensor(
                self.paged_kv_last_page_lens, device=device, dtype=torch.int)
            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device=device,
                                                   dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_lens_tensor = None
            block_table_bound_tensor = None

        metadata.paged_kv_indptr = paged_kv_indptr_tensor
        metadata.paged_kv_indices = paged_kv_indices_tensor
        metadata.paged_kv_last_page_lens = paged_kv_last_page_lens_tensor
        metadata.block_table_bound = block_table_bound_tensor

        return metadata


class AiterMLAState(MLACommonState[AiterMLAMetadata]):

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._paged_kv_indices_tensor = torch.from_numpy(
            self.runner.paged_kv_indices).to(device=self.runner.device)
        self._paged_kv_indptr_tensor = torch.zeros(max_batch_size + 1,
                                                   dtype=torch.int32,
                                                   device=self.runner.device)
        self._paged_kv_last_page_lens_tensor = torch.full(
            (max_batch_size, ), self.runner.block_size, dtype=torch.int32)
        with super().graph_capture(max_batch_size):
            yield

        del self._paged_kv_indices_tensor
        del self._paged_kv_indptr_tensor
        del self._paged_kv_last_page_lens_tensor

    def graph_capture_get_metadata_for_batch(
            self,
            batch_size: int,
            is_encoder_decoder_model: bool = False) -> AiterMLAMetadata:

        metadata = super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)

        paged_kv_indptr = self._paged_kv_indptr_tensor[:batch_size + 1]
        paged_kv_indices = self._paged_kv_indices_tensor
        paged_kv_last_page_lens = self._paged_kv_last_page_lens_tensor[:
                                                                       batch_size]

        metadata.paged_kv_indptr = paged_kv_indptr
        metadata.paged_kv_indices = paged_kv_indices
        metadata.paged_kv_last_page_lens = paged_kv_last_page_lens

        return metadata

    def get_graph_input_buffers(self,
                                attn_metadata: AiterMLAMetadata,
                                is_encoder_decoder_model: bool = False):
        input_buffers = super().get_graph_input_buffers(
            attn_metadata, is_encoder_decoder_model)
        input_buffers[
            'paged_kv_indptr'] = attn_metadata.decode_metadata.paged_kv_indptr
        input_buffers[
            "paged_kv_indices"] = attn_metadata.\
            decode_metadata.paged_kv_indices
        input_buffers[
            "paged_kv_last_page_lens"] = attn_metadata.\
            decode_metadata.paged_kv_last_page_lens

        return input_buffers

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata: AiterMLAMetadata,
                                    is_encoder_decoder_model: bool = False):
        super().prepare_graph_input_buffers(input_buffers, attn_metadata,
                                            is_encoder_decoder_model)

        num_total_blocks = attn_metadata.decode_metadata.paged_kv_indices.shape[
            0]
        input_buffers["paged_kv_indptr"].copy_(
            attn_metadata.decode_metadata.paged_kv_indptr, non_blocking=True)
        input_buffers["paged_kv_indices"][:num_total_blocks].copy_(
            attn_metadata.decode_metadata.paged_kv_indices, non_blocking=True)
        input_buffers["paged_kv_last_page_lens"].copy_(
            attn_metadata.decode_metadata.paged_kv_last_page_lens,
            non_blocking=True)


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)
        assert is_aiter_mla_enabled(
        ), "Aiter MLA is initialized without being enabled properly."

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
    ):
        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None
        assert prefill_metadata.context_chunk_seq_tot is not None
        assert prefill_metadata.context_chunk_cu_seq_lens is not None
        assert prefill_metadata.context_chunk_starts is not None
        assert prefill_metadata.context_chunk_max_seq_lens is not None
        assert prefill_metadata.context_lens_tensor is not None

        output = None
        iters = len(prefill_metadata.context_chunk_seq_tot)

        # Fetch from attn_metadata directly, since it late bound by
        # MLAAttentionState, grabbing it directly `attn_metadata` can avoid
        # any weirdness around prefill_metadata caching
        assert attn_metadata.context_chunk_workspace is not None
        workspace = attn_metadata.context_chunk_workspace

        for i in range(iters):
            toks = prefill_metadata.context_chunk_seq_tot[i]

            ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_tables,
                cu_seq_lens=prefill_metadata.context_chunk_cu_seq_lens[i],
                batch_size=prefill_metadata.num_prefills,
                seq_starts=prefill_metadata.context_chunk_starts[i],
            )

            kv_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank]
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

            # For MLA the v head dim is smaller than qk head dim so we pad
            # out v with 0s to match the qk head dim
            v_padded = torch.nn.functional.pad(v,
                                               [0, q.shape[-1] - v.shape[-1]],
                                               value=0)
            attn_output, attn_softmax_lse = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i],
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.context_chunk_max_seq_lens[i],
                softmax_scale=self.scale,
                causal=False,  # Context is unmasked
                return_lse=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:

        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None

        has_context = prefill_metadata.context_lens_tensor is not None \
            and prefill_metadata.context_lens_tensor.max() > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=prefill_metadata.query_start_loc,
            cu_seqlens_k=prefill_metadata.query_start_loc,
            max_seqlen_q=prefill_metadata.max_prefill_seq_len,
            max_seqlen_k=prefill_metadata.max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
            return_lse=has_context,
        )

        if not has_context:
            output = output[0]

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # slice by `:v.shape[-1]` in order to remove v headdim padding
        output = output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(output)[0]

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        num_kv_splits = 4  # TODO: heuristic

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)

        aiter_mla_decode_fwd(q, kv_c_and_k_pe_cache, o, attn_logits,
                             num_kv_splits, self.scale,
                             attn_metadata.paged_kv_indptr,
                             attn_metadata.paged_kv_indices,
                             attn_metadata.paged_kv_last_page_lens)

        return self._v_up_proj_and_o_proj(o)
