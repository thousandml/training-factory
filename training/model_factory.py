"""Construct a text-only HF CausalLM for CQ training without Llama-specific assumptions.

Inference can serve multimodal and expert architectures that the current training
kernels cannot train. Construction does not imply numerical backend support.
"""
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
import torch


def causal_lm_skeleton(root, *, trust_remote_code=False, text_only=False):
    config = AutoConfig.from_pretrained(root, trust_remote_code=trust_remote_code)
    if hasattr(config, 'quantization_config'):
        del config.quantization_config
    attempts = [config]
    if text_only:
        for attr in ('text_config', 'language_config', 'llm_config'):
            child = getattr(config, attr, None)
            if child is not None and child is not config and child not in attempts:
                attempts.insert(0, child)
    errors = []
    for candidate in attempts:
        if hasattr(candidate, 'quantization_config'):
            del candidate.quantization_config
        constructors = [AutoModelForCausalLM] if text_only else ([AutoModelForImageTextToText] if AutoModelForImageTextToText else []) + [AutoModelForCausalLM]
        for constructor in constructors:
            try:
                with init_empty_weights():
                    model = constructor.from_config(candidate, trust_remote_code=trust_remote_code, dtype=torch.float16)
                model.tie_weights()
                if not text_only and not _has_multimodal_tower(model) and _has_multimodal_config(config):
                    raise ValueError('Model skeleton omits multimodal towers; refusing silent text-only fallback')
                return model, candidate
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f'{constructor.__name__}/{getattr(candidate, "model_type", type(candidate).__name__)}: {exc}')
    raise ValueError('No supported text CausalLM skeleton for this checkpoint. '
                     'Try --text-only on a multimodal model, or add an architecture bridge. '
                     + '; '.join(errors))


def _has_multimodal_config(config):
    return any(getattr(config, key, None) is not None for key in ('vision_config', 'audio_config', 'visual_config'))

def _has_multimodal_tower(model):
    return any(any(token in name.lower().split('.') for token in ('visual', 'vision', 'vision_model', 'audio', 'audio_tower', 'audio_model')) for name, _ in model.named_modules())
