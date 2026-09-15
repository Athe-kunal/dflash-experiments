"""vLLM general plugin: entropy-vs-acceptance-length instrumentation.

Registered under the `vllm.general_plugins` entry point (see this project's
pyproject.toml), so vLLM loads and calls `register()` once in every process
(API server, engine core, and each GPU worker) -- see vllm/plugins/__init__.py.
`register()` wraps `RejectionSampler.__call__` to log, for every drafted
position of every request verified this step: the exact Shannon entropy of
the target model's full-vocab next-token distribution, and whether that
position's draft token was accepted.

Inert unless the `ENTROPY_LOG_PATH` env var is set, so it has zero effect on
any other use of this vLLM install.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_LOG_PATH = os.environ.get("ENTROPY_LOG_PATH")
_fh = None


def _get_fh():
    global _fh
    if _fh is None:
        _fh = open(_LOG_PATH, "a", buffering=1)
    return _fh


_LOG_TOKENS = os.environ.get("ENTROPY_LOG_TOKENS", "0").strip().lower() in ("1", "true")


def _log_entropy(logits, input_batch, num_sampled, sampled_token_ids) -> None:
    import numpy as np
    import torch

    try:
        req_ids = input_batch.req_ids
        cu = input_batch.cu_num_logits_np
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        entropy_np = entropy.cpu().numpy()
        num_sampled_np = (
            num_sampled.cpu().numpy() if torch.is_tensor(num_sampled) else np.asarray(num_sampled)
        )
        token_ids_np = None
        if _LOG_TOKENS and sampled_token_ids is not None:
            token_ids_np = sampled_token_ids.cpu().numpy()
        now = time.time()
        fh = _get_fh()
        for i in range(num_reqs):
            lo, hi = int(cu[i]), int(cu[i + 1])
            n_positions = hi - lo
            if n_positions <= 0:
                continue
            accepted_len = int(num_sampled_np[i])
            for pos in range(n_positions):
                rec = {
                    "t": now,
                    "req_id": req_ids[i],
                    "pos": pos,
                    "entropy": float(entropy_np[lo + pos]),
                    "accepted": pos < accepted_len,
                    "accepted_len": accepted_len,
                    "block_size": n_positions,
                }
                if token_ids_np is not None and pos < token_ids_np.shape[1]:
                    rec["token_id"] = int(token_ids_np[i, pos])
                fh.write(json.dumps(rec) + "\n")
    except Exception:
        logger.warning("entropy_plugin: failed to log a step", exc_info=True)


def register() -> None:
    if not _LOG_PATH:
        return

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    if getattr(RejectionSampler.__call__, "_entropy_instrumented", False):
        return  # already patched (plugins can load more than once per process)

    original_call = RejectionSampler.__call__

    def patched_call(self, logits, input_batch, draft_logits=None):
        output = original_call(self, logits, input_batch, draft_logits=draft_logits)
        _log_entropy(logits, input_batch, output.num_sampled)
        return output

    patched_call._entropy_instrumented = True
    RejectionSampler.__call__ = patched_call
    logger.info("entropy_plugin: RejectionSampler patched, logging to %s", _LOG_PATH)
