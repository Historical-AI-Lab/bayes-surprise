"""model_query.py — model-access primitives for the Bayesian-surprise pipeline.

Two public functions:

    generate_answer(prompt, model_id, ...)  -> dict
        Calls an instruction-tuned model via vLLM's OpenAI-compatible
        chat-completions endpoint.

    score_continuations(prompt, continuations, model_id, ...)  -> list[dict]
        Measures the sum log-likelihood of each continuation after the
        prompt, using a base (non-instruction-tuned) model.

Both functions default to a vLLM server assumed to be running locally.
Configure the server address and API key via environment variables:

    BAYES_VLLM_BASE_URL   (default: http://localhost:8000/v1)
    BAYES_VLLM_API_KEY    (default: EMPTY)

For validation/cross-checking, pass backend="transformers" to use an
in-process HuggingFace Transformers forward pass instead.  That path is
intentionally slower and for spot-checks only; it does not require a
running server.

Model identifiers recognised by the project (pass any of these as model_id):

    MODELS dict keys: "mixtral-it", "mixtral-base", "gemma-it", "gemma-base"
    or a full HuggingFace model string for either backend.
"""

import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Model catalogue (spec §0)
# ---------------------------------------------------------------------------

MODELS = {
    "mixtral-it":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-base":  "mistralai/Mixtral-8x7B-v0.1",
    "gemma-it":      "google/gemma-4-31B-it",
    "gemma-base":    "google/gemma-4-31B",
}


def _resolve_model(model_id: str) -> str:
    """Expand a short alias to the full HuggingFace model string."""
    return MODELS.get(model_id, model_id)


# ---------------------------------------------------------------------------
# vLLM client (shared across calls)
# ---------------------------------------------------------------------------

def _get_client():
    """Return a cached openai.OpenAI client pointed at the vLLM server."""
    global _vllm_client
    try:
        return _vllm_client
    except NameError:
        pass
    import openai
    base_url = os.environ.get("BAYES_VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key  = os.environ.get("BAYES_VLLM_API_KEY",  "EMPTY")
    _vllm_client = openai.OpenAI(base_url=base_url, api_key=api_key)
    return _vllm_client


# ---------------------------------------------------------------------------
# Function A: generation
# ---------------------------------------------------------------------------

def generate_answer(
    prompt: str,
    model_id: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    top_p: float | None = None,
    reasoning: str = "off",   # "off" | "on"
    seed: int | None = None,
    backend: str = "vllm",
) -> dict:
    """Call an instruction-tuned model and return its reply.

    Parameters
    ----------
    prompt        : the user-turn text.
    model_id      : full HF model string or short alias from MODELS.
    system_prompt : optional system-turn text.
    temperature   : sampling temperature (0.0 = greedy).
    max_new_tokens: maximum tokens to generate.
    top_p         : nucleus sampling probability; None = omit parameter.
    reasoning     : "on" enables Gemma-4 thinking mode; default "off".
                    For Mixtral this flag is ignored.
    seed          : optional integer seed for reproducibility.
    backend       : "vllm" (default) | "transformers" (in-process, slow).

    Returns
    -------
    dict with keys:
        text          - generated text (thinking stripped if applicable).
        reasoning     - reasoning trace if the model exposed it, else None.
        raw_response  - the raw API response object (for debugging).
        model_id      - the resolved model string.
        finish_reason - e.g. "stop", "length".
    """
    model_id = _resolve_model(model_id)

    if backend == "transformers":
        return _generate_transformers(
            prompt, model_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if backend != "vllm":
        raise ValueError(f"Unknown backend {backend!r}; use 'vllm' or 'transformers'.")

    client = _get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    extra_body: dict = {}
    # Gemma-4 thinking mode — only when explicitly requested.
    # vLLM exposes the thinking trace in message.reasoning after stripping
    # the thought-channel label.  Never enable for likelihood scoring.
    if model_id.startswith("google/gemma-4") and reasoning != "off":
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    kwargs: dict = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        extra_body=extra_body or None,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message

    text = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None)

    return {
        "text":         text,
        "reasoning":    reasoning_text,
        "raw_response": response,
        "model_id":     model_id,
        "finish_reason": choice.finish_reason,
    }


def _generate_transformers(
    prompt: str,
    model_id: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> dict:
    """In-process generation via HF Transformers.  Validation use only."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as e:
        raise ImportError("transformers/torch required for backend='transformers'") from e

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )

    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")
    parts.append(prompt)
    text_in = "".join(parts)

    inputs = tokenizer(text_in, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with __import__("torch").no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text_out = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "text":         text_out,
        "reasoning":    None,
        "raw_response": None,
        "model_id":     model_id,
        "finish_reason": "stop",
    }


# ---------------------------------------------------------------------------
# Function B: likelihood scoring
# ---------------------------------------------------------------------------

def score_continuations(
    prompt: str,
    continuations: list[str],
    model_id: str,
    *,
    normalize: str = "none",       # "none" | "per_token"
    add_bos: bool | None = None,
    backend: str = "vllm",
    use_prefix_cache: bool = True,
    batch_size: int | None = None,
    return_token_details: bool = False,
) -> list[dict]:
    """Score the log-likelihood of each continuation after prompt.

    Each continuation should include the leading whitespace/newline exactly
    as it should be scored.  This function does NOT silently prepend a space.

    Parameters
    ----------
    prompt        : the shared prefix.
    continuations : list of continuation strings (each including leading
                    whitespace if needed).
    model_id      : full HF model string or short alias from MODELS.
                    Should be a base (non-IT) model for exact likelihoods.
    normalize     : "none" returns the raw sum log-likelihood;
                    "per_token" divides by the number of continuation tokens.
    add_bos       : whether to prepend a BOS token.  None lets the tokenizer
                    decide.  For most models, pass False for continuations
                    after a non-empty prompt.
    backend       : "vllm" (default) | "transformers".
    use_prefix_cache : hint to vLLM to reuse the prompt's KV cache
                    (always true when prefix caching is enabled server-side,
                    but batching all continuations together reinforces it).
    batch_size    : if set, score this many continuations per API call
                    rather than all at once.  None = send all together.
    return_token_details : if True, include per_token list in each result.

    Returns
    -------
    List of dicts (same order as continuations), each with:
        continuation  - the original string.
        logprob       - sum log-likelihood over continuation tokens.
        avg_logprob   - logprob / n_tokens.
        n_tokens      - number of continuation tokens.
        per_token     - list of (token_str, logprob) or None.
    """
    model_id = _resolve_model(model_id)

    if backend == "transformers":
        return _score_transformers(
            prompt, continuations, model_id,
            normalize=normalize,
            add_bos=add_bos,
            return_token_details=return_token_details,
        )

    if backend != "vllm":
        raise ValueError(f"Unknown backend {backend!r}; use 'vllm' or 'transformers'.")

    return _score_vllm(
        prompt, continuations, model_id,
        normalize=normalize,
        add_bos=add_bos,
        batch_size=batch_size,
        return_token_details=return_token_details,
    )


def _score_vllm(
    prompt: str,
    continuations: list[str],
    model_id: str,
    *,
    normalize: str,
    add_bos: bool | None,
    batch_size: int | None,
    return_token_details: bool,
) -> list[dict]:
    """vLLM scoring via completions + prompt_logprobs.

    Sends prompt+continuation as a single string and requests
    prompt_logprobs=1.  The continuation span is identified by comparing
    the tokenisation of (prompt) versus (prompt+continuation) — not by
    tokenising the continuation alone, which is incorrect when tokenisation
    is non-compositional at the boundary.
    """
    from transformers import AutoTokenizer

    client   = _get_client()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenise the prompt once to find its boundary.
    # add_special_tokens=True matches vLLM's default (BOS prepended).
    add_st = (add_bos if add_bos is not None else True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_st)
    n_prompt = len(prompt_ids)

    def _score_one(cont: str) -> dict:
        full_text  = prompt + cont
        full_ids   = tokenizer.encode(full_text, add_special_tokens=add_st)
        n_cont_toks = len(full_ids) - n_prompt
        if n_cont_toks <= 0:
            warnings.warn(
                f"Continuation {cont!r:.40} tokenises to 0 tokens after prompt; "
                "returning logprob=0."
            )
            return {
                "continuation": cont,
                "logprob":      0.0,
                "avg_logprob":  0.0,
                "n_tokens":     0,
                "per_token":    [] if return_token_details else None,
            }

        response = client.completions.create(
            model=model_id,
            prompt=full_text,
            max_tokens=0,
            temperature=0,
            extra_body={
                "prompt_logprobs": 1,
                "add_special_tokens": add_st,
            },
        )

        # vLLM returns prompt_logprobs as a list parallel to the prompt
        # tokens (index 0 is None for the very first token, then log-probs
        # for each subsequent token).  We want only the continuation span.
        raw_lp = response.choices[0].prompt_logprobs  # list[None | dict]

        # Extract the continuation slice: tokens at positions
        # n_prompt .. len(full_ids)-1 in the prompt_logprobs array.
        logprob_sum = 0.0
        per_token   = [] if return_token_details else None

        for pos in range(n_prompt, len(full_ids)):
            entry = raw_lp[pos] if raw_lp and pos < len(raw_lp) else None
            if entry is None:
                continue
            # entry is a dict {token_id: logprob_value, ...}; pick top token.
            tok_id  = full_ids[pos]
            lp_val  = entry.get(tok_id)
            if lp_val is None:
                # Fall back to the highest-probability token in the dict.
                lp_val = max(entry.values())
            logprob_sum += lp_val
            if return_token_details:
                tok_str = tokenizer.decode([tok_id])
                per_token.append((tok_str, lp_val))

        avg = logprob_sum / n_cont_toks if n_cont_toks else 0.0
        if normalize == "per_token":
            logprob_sum = avg

        return {
            "continuation": cont,
            "logprob":      logprob_sum,
            "avg_logprob":  avg,
            "n_tokens":     n_cont_toks,
            "per_token":    per_token,
        }

    # Score in batches if requested (mostly useful for memory budgets;
    # vLLM prefix caching already handles the shared-prefix optimisation
    # as long as all continuations are sent close together in time).
    results = []
    batch = batch_size or len(continuations)
    for start in range(0, len(continuations), batch):
        chunk = continuations[start : start + batch]
        for c in chunk:
            results.append(_score_one(c))
    return results


def _score_transformers(
    prompt: str,
    continuations: list[str],
    model_id: str,
    *,
    normalize: str,
    add_bos: bool | None,
    return_token_details: bool,
) -> list[dict]:
    """In-process reference scorer via HF Transformers.

    A single forward pass per continuation; no batching across prompts
    (intentional — this path is for validation spot-checks only, not
    for production throughput).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("transformers/torch required for backend='transformers'") from e

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    add_st = (add_bos if add_bos is not None else True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_st)
    n_prompt   = len(prompt_ids)

    results = []
    for cont in continuations:
        full_text = prompt + cont
        full_ids  = tokenizer.encode(full_text, add_special_tokens=add_st)
        n_cont    = len(full_ids) - n_prompt

        if n_cont <= 0:
            results.append({
                "continuation": cont,
                "logprob":      0.0,
                "avg_logprob":  0.0,
                "n_tokens":     0,
                "per_token":    [] if return_token_details else None,
            })
            continue

        input_ids = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # (seq, vocab)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        logprob_sum = 0.0
        per_token   = [] if return_token_details else None
        for i, tok_id in enumerate(full_ids[n_prompt:], start=n_prompt):
            lp = float(log_probs[i - 1, tok_id])
            logprob_sum += lp
            if return_token_details:
                per_token.append((tokenizer.decode([tok_id]), lp))

        avg = logprob_sum / n_cont if n_cont else 0.0
        if normalize == "per_token":
            logprob_sum = avg

        results.append({
            "continuation": cont,
            "logprob":      logprob_sum,
            "avg_logprob":  avg,
            "n_tokens":     n_cont,
            "per_token":    per_token,
        })

    return results
