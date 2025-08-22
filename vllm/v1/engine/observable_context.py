from dataclasses import dataclass
from typing import Optional

import itertools

from vllm.transformers_utils.detokenizer_utils import convert_ids_list_to_tokens
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput
from vllm.v1.outputs import IterStats

NONES = itertools.repeat(None)

@dataclass
class ObservableContext:

    iter_batch_size: list[int]
    iter_total_tokens_count: list[int]
    scheduled_time: list[float]
    token_time: list[float]
    candidate_token_ids: Optional[list[list[int]]]
    candidate_decoded_tokens: Optional[list[list[str]]]
    candidate_token_probs: Optional[list[list[float]]]
    tokenizer: Optional[AnyTokenizer]
    not_empty: bool = False

    @classmethod
    def from_new_request(cls, tokenizer: Optional[AnyTokenizer]):
            return cls(
                iter_batch_size=[],
                iter_total_tokens_count=[],
                scheduled_time=[],
                token_time=[],
                candidate_token_ids=[],
                candidate_decoded_tokens=[],
                candidate_token_probs=[],
                tokenizer=tokenizer
            )

    def _update_iter_stats(self, iter_stats: IterStats, new_token_ids: list[int]) -> None:
        self.not_empty = True
        self.iter_total_tokens_count.append(iter_stats.iter_total_tokens_count)
        self.scheduled_time.append(iter_stats.token_scheduled_time)
        self.token_time.append(iter_stats.token_output_time)
        self.iter_batch_size.append(iter_stats.iter_batch_size)
        if iter_stats.logprobs_tensors_for_trace:
            token_ids_lst, logprobs_lst, ranks_lst = iter_stats.logprobs_tensors_for_trace
            for _, logprobs, token_ids in zip(ranks_lst, logprobs_lst, token_ids_lst):
                # Detokenize (non-incrementally).
                decoded_tokens = NONES if self.tokenizer is None else (
                    convert_ids_list_to_tokens(self.tokenizer, token_ids, True))
                # Update with the Logprob dictionary for this pos.
                self.candidate_token_ids.append(token_ids)
                self.candidate_decoded_tokens.append(decoded_tokens)
                self.candidate_token_probs.append(logprobs)
        else:
            self.candidate_token_ids.extend([[id,id] for id in new_token_ids])
            decoded_tokens = NONES if self.tokenizer is None else (
                    convert_ids_list_to_tokens(self.tokenizer, new_token_ids, True))
            self.candidate_decoded_tokens.extend([[token,token] for token in decoded_tokens])



    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.iter_stats is not None:
            self._update_iter_stats(output.iter_stats, output.new_token_ids)