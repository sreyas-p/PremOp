"""MEMIT-style weight editing and LoRA consolidation on Apple Silicon, via MLX.

Requires bf16 weights. A 4-bit model cannot be edited: `down_proj` is a
`QuantizedLinear` holding packed uint32, and a MEMIT delta is a small fp update
to the unpacked matrix — a dequantize/edit/requantize round trip loses more
precision than the edit carries. Load `...-bf16`, not `...-4bit`.

The edit follows MEMIT's batched form on a single MLP down-projection:

    ΔW = R Kᵀ (C + K Kᵀ)⁻¹

with K the stacked keys (down_proj inputs at each subject's last token), R the
stacked residuals (v* − W k), and C the second-moment statistic E[k kᵀ]
estimated from a corpus. Solving the batch jointly is the point — applying
edits one at a time gives materially worse results, which is the difference
between MEMIT and repeated ROME.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from ..models import Fact

#: Layers to edit, ascending. Mid-stack MLPs carry factual association.
#: Layer 13 alone is measured working on Llama-3.2-3B (see README); widening
#: the range spreads each edit thinner, which needs the sequential distribution
#: in `apply_edits` to compose correctly.
DEFAULT_EDIT_LAYERS = (13,)

_DRIFT_CORPUS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level pressure.",
    "The mitochondrion is often described as the powerhouse of the cell.",
    "In 1969 Apollo 11 landed the first humans on the surface of the Moon.",
    "A prime number has exactly two distinct positive divisors.",
]


class _Capture(nn.Module):
    """Wraps a Linear to record what was fed into it on the last forward pass."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.last_input: mx.array | None = None

    def __call__(self, x: mx.array) -> mx.array:
        self.last_input = x
        return self.inner(x)


class MLXBackend:
    """A real model behind the `MemoryBackend` protocol."""

    def __init__(
        self,
        model_id: str = "mlx-community/Llama-3.2-3B-Instruct-bf16",
        *,
        edit_layers: tuple[int, ...] = DEFAULT_EDIT_LAYERS,
        drift_corpus: list[str] | None = None,
        snapshot_dir: Path | str = "./snapshots",
        covariance_samples: int = 256,
        value_steps: int = 25,
        value_lr: float = 0.5,
        lora_layers: int = 8,
    ) -> None:
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.edit_layers = edit_layers
        self._drift_corpus = drift_corpus or _DRIFT_CORPUS
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._covariance_samples = covariance_samples
        self._value_steps = value_steps
        self._value_lr = value_lr
        self._lora_layers = lora_layers

        self._guard_against_quantization()

        #: Pristine down_proj weights, so edits are always solved from the base
        #: matrix rather than compounding on top of the previous edit.
        self._base: dict[int, mx.array] = {
            layer: mx.array(self._down(layer).weight) for layer in edit_layers
        }
        self._covariance: dict[int, mx.array] = {}
        self._lora_fused = 0

    @property
    def name(self) -> str:
        return f"mlx:{self.model_id}@layers{list(self.edit_layers)}"

    # ── model plumbing ──────────────────────────────────────────────────

    def _layer(self, index: int):
        return self.model.model.layers[index]

    def _down(self, index: int) -> nn.Module:
        return self._layer(index).mlp.down_proj

    def _guard_against_quantization(self) -> None:
        offender = type(self._down(self.edit_layers[0])).__name__
        if offender != "Linear":
            raise TypeError(
                f"down_proj is {offender}, not Linear — this model is quantized "
                "and cannot be weight-edited. Load a bf16 checkpoint "
                "(e.g. mlx-community/Llama-3.2-3B-Instruct-bf16)."
            )

    def _encode(self, text: str) -> mx.array:
        return mx.array(self.tokenizer.encode(text))[None]

    # ── keys, values, covariance ────────────────────────────────────────

    def _key(self, layer: int, prompt: str, subject: str) -> mx.array:
        """The down_proj input at the subject's final token — MEMIT's k."""
        layer_module = self._layer(layer)
        original = layer_module.mlp.down_proj
        capture = _Capture(original)
        layer_module.mlp.down_proj = capture
        try:
            ids = self._encode(prompt)
            self.model(ids)
            mx.eval(capture.last_input)
            activations = capture.last_input
            index = self._subject_end_index(prompt, subject)
            return activations[0, index]
        finally:
            layer_module.mlp.down_proj = original

    def _subject_end_index(self, prompt: str, subject: str) -> int:
        """Token index of the subject's last token, falling back to the end."""
        prompt_ids = self.tokenizer.encode(prompt)
        position = prompt.rfind(subject)
        if position < 0:
            return len(prompt_ids) - 1
        prefix_ids = self.tokenizer.encode(prompt[: position + len(subject)])
        return max(0, min(len(prefix_ids), len(prompt_ids)) - 1)

    def _value(self, layer: int, prompt: str, subject: str, target: str,
               key: mx.array, weight: mx.array | None = None) -> mx.array:
        """Optimize v* so the model emits `target` when down_proj outputs it.

        Gradient descent on the output vector alone, not on any weights — the
        weights are solved for afterwards, in closed form.

        Two details that decide whether this works at all: the patch goes at the
        *subject's* last token (the same position the key was read from, or the
        association is written against the wrong key), and the loss covers every
        target token rather than only the first — the first token of "a quantum
        florist" is " a", which carries essentially no signal.
        """
        weight = self._base[layer] if weight is None else weight
        v = weight @ key  # start from what the model already produces

        prompt_ids = self.tokenizer.encode(prompt)
        full_ids = self.tokenizer.encode(f"{prompt} {target}".strip())
        if len(full_ids) <= len(prompt_ids):
            return v
        ids = mx.array(full_ids)[None]
        first_target = len(prompt_ids)
        targets = mx.array(full_ids[first_target:])

        layer_module = self._layer(layer)
        original = layer_module.mlp.down_proj
        index = self._subject_end_index(prompt, subject)

        def loss_fn(candidate: mx.array) -> mx.array:
            layer_module.mlp.down_proj = _ValuePatch(original, index, candidate)
            try:
                logits = self.model(ids)
            finally:
                layer_module.mlp.down_proj = original
            predictions = logits[0, first_target - 1 : len(full_ids) - 1]
            return nn.losses.cross_entropy(predictions, targets, reduction="mean")

        loss_and_grad = mx.value_and_grad(loss_fn)
        for _ in range(self._value_steps):
            loss, grad = loss_and_grad(v)
            # Normalized step: raw gradient scale varies by orders of magnitude
            # across layers, which makes a single fixed lr useless.
            norm = mx.linalg.norm(grad)
            v = v - self._value_lr * grad / mx.maximum(norm, 1e-6)
            mx.eval(v, loss)

        # ROME's constraint: keep the new value near what the layer already
        # produces, or the edit bleeds into unrelated prompts.
        residual = v - weight @ key
        max_norm = 4.0 * mx.linalg.norm(weight @ key)
        residual_norm = mx.linalg.norm(residual)
        if float(residual_norm.item()) > float(max_norm.item()):
            v = weight @ key + residual * (max_norm / residual_norm)
        return v

    def _estimate_covariance(self, layer: int) -> mx.array:
        """C = E[k kᵀ] over the drift corpus, cached per layer.

        MEMIT estimates this over ~100k Wikipedia samples. This is a much
        smaller sample and will be correspondingly noisier — raise
        `covariance_samples` and widen the corpus before trusting edit quality.
        """
        if layer in self._covariance:
            return self._covariance[layer]

        layer_module = self._layer(layer)
        original = layer_module.mlp.down_proj
        capture = _Capture(original)
        layer_module.mlp.down_proj = capture

        dimension = self._base[layer].shape[1]
        accumulator = mx.zeros((dimension, dimension), dtype=mx.float32)
        count = 0
        try:
            for text in self._drift_corpus:
                self.model(self._encode(text))
                mx.eval(capture.last_input)
                keys = capture.last_input[0].astype(mx.float32)
                accumulator = accumulator + keys.T @ keys
                count += keys.shape[0]
        finally:
            layer_module.mlp.down_proj = original

        covariance = accumulator / max(count, 1)
        self._covariance[layer] = covariance
        return covariance

    # ── MemoryBackend surface ───────────────────────────────────────────

    def apply_edits(self, facts: list[Fact]) -> None:
        """Solve and write the full active edit set, from pristine weights.

        Always re-solved from `self._base` rather than layered on top of the
        previous edit, so scaling a fact down actually shrinks its delta instead
        of leaving the old one buried underneath.

        Across multiple layers the residual is distributed *sequentially* — each
        layer is edited, then the next layer's key and residual are recomputed
        against the already-updated model. Splitting one independently-solved
        residual across layers in parallel does not compose through the
        nonlinearity: measured on Llama-3.2-3B it fails to land the edit at all,
        while the same fact on a single layer succeeds.
        """
        for layer in self.edit_layers:
            self._down(layer).weight = mx.array(self._base[layer])
        mx.eval(self.model.parameters())

        live = [f for f in facts if f.memit_scale > 0]
        if not live:
            return

        remaining = len(self.edit_layers)
        for layer in self.edit_layers:
            current = self._down(layer).weight.astype(mx.float32)

            keys, residuals = [], []
            for fact in live:
                k = self._key(layer, fact.prompt, fact.subject).astype(mx.float32)
                v = self._value(
                    layer, fact.prompt, fact.subject, fact.target, k, weight=current
                ).astype(mx.float32)
                # Take this layer's share of what is still missing.
                share = fact.memit_scale / remaining
                keys.append(k)
                residuals.append((v - current @ k) * share)

            K = mx.stack(keys, axis=1)        # (d_mlp, n)
            R = mx.stack(residuals, axis=1)   # (d_model, n)
            C = self._estimate_covariance(layer)

            gram = C + K @ K.T
            gram = gram + mx.eye(gram.shape[0]) * 1e-4  # keep the solve stable
            delta = R @ K.T @ mx.linalg.inv(gram, stream=mx.cpu)
            self._down(layer).weight = (current + delta).astype(
                self._base[layer].dtype
            )
            mx.eval(self._down(layer).weight)
            remaining -= 1

    def probe(self, facts: list[Fact]) -> dict[str, bool]:
        from mlx_lm import generate

        results: dict[str, bool] = {}
        for fact in facts:
            text = generate(
                self.model, self.tokenizer, prompt=fact.prompt, max_tokens=24
            )
            results[fact.id] = fact.target.lower() in text.lower()
        return results

    def ask(self, question: str, *, max_tokens: int = 96) -> str:
        from mlx_lm import generate

        return generate(
            self.model, self.tokenizer, prompt=question, max_tokens=max_tokens
        ).strip()

    def consolidate(self, facts: list[Fact], *, rank: int, alpha: int,
                    epochs: int, lr: float) -> None:
        """Train a LoRA adapter on `facts`, then fuse it into the base weights.

        Loss is masked to the target tokens: training on the prompt as well
        teaches the model to produce the question, which is not the point.

        After fusing, `self._base` is refreshed from the new weights. Without
        that, the next `apply_edits` would reset down_proj back to pre-LoRA
        values and silently throw the consolidation away.
        """
        import mlx.optimizers as optim
        from mlx_lm.tuner import linear_to_lora_layers
        from mlx_lm.tuner.lora import LoRALinear

        examples = []
        for fact in facts:
            prompt_ids = self.tokenizer.encode(fact.prompt)
            full_ids = self.tokenizer.encode(f"{fact.prompt} {fact.target}".strip())
            if len(full_ids) > len(prompt_ids):
                examples.append((full_ids, len(prompt_ids)))
        if not examples:
            return

        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            num_layers=self._lora_layers,
            config={"rank": rank, "scale": alpha / max(rank, 1), "dropout": 0.0},
        )
        self.model.train()

        def loss_fn(model, ids: mx.array, mask: mx.array) -> mx.array:
            logits = model(ids[:, :-1]).astype(mx.float32)
            targets = ids[:, 1:]
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            weights = mask[:, 1:].reshape(-1)
            return (losses * weights).sum() / mx.maximum(weights.sum(), 1.0)

        optimizer = optim.Adam(learning_rate=lr)
        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        for _ in range(epochs):
            for token_ids, prompt_length in examples:
                ids = mx.array(token_ids)[None]
                mask = mx.concatenate(
                    [
                        mx.zeros((1, prompt_length)),
                        mx.ones((1, len(token_ids) - prompt_length)),
                    ],
                    axis=1,
                )
                _, grads = loss_and_grad(self.model, ids, mask)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)

        self.model.eval()
        for _, module in self.model.named_modules():
            for child_name, child in list(module.children().items()):
                if isinstance(child, LoRALinear):
                    setattr(module, child_name, child.fuse())
        self.model.unfreeze()
        mx.eval(self.model.parameters())

        # The consolidated weights are the new floor for future edits.
        self._base = {
            layer: mx.array(self._down(layer).weight) for layer in self.edit_layers
        }
        self._covariance.clear()
        self._lora_fused += 1

    def perplexity(self) -> float:
        total_nll = 0.0
        total_tokens = 0
        for text in self._drift_corpus:
            ids = self._encode(text)
            if ids.shape[1] < 2:
                continue
            logits = self.model(ids[:, :-1])
            targets = ids[:, 1:]
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none",
            )
            total_nll += float(losses.sum().item())
            total_tokens += targets.size
        return math.exp(total_nll / max(total_tokens, 1))

    def snapshot(self) -> str:
        handle = f"snap_{uuid.uuid4().hex[:12]}"
        payload = {
            str(layer): self._down(layer).weight for layer in self.edit_layers
        }
        mx.save_safetensors(str(self._snapshot_dir / f"{handle}.safetensors"), payload)
        (self._snapshot_dir / f"{handle}.json").write_text(
            json.dumps({"model": self.model_id, "layers": list(self.edit_layers),
                        "lora_fused": self._lora_fused})
        )
        return handle

    def restore(self, handle: str) -> None:
        path = self._snapshot_dir / f"{handle}.safetensors"
        if not path.exists():
            raise KeyError(f"Unknown snapshot {handle!r}")
        weights = mx.load(str(path))
        for layer in self.edit_layers:
            self._down(layer).weight = weights[str(layer)]
        mx.eval(self.model.parameters())


class _ValuePatch(nn.Module):
    """Replaces one position's down_proj output with a candidate vector."""

    def __init__(self, inner: nn.Module, index: int, value: mx.array) -> None:
        super().__init__()
        self.inner = inner
        self.index = index
        self.value = value

    def __call__(self, x: mx.array) -> mx.array:
        out = self.inner(x)
        index = min(self.index, out.shape[1] - 1)
        mask = mx.arange(out.shape[1])[None, :, None] == index
        return mx.where(mask, self.value.astype(out.dtype), out)
