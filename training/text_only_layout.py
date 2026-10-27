"""Filter CQ records for towers intentionally absent from text-only skeletons.

Use explicit path segments: the compiled namespace() helper may classify a
Gemma 4 `model.vision_tower.*` record as backbone.
"""

_AUX_SEGMENTS = {
    "vision": {"vision_tower", "vision_model", "vision_encoder", "visual", "vision", "embed_vision", "vision_projector", "multi_modal_projector"},
    "audio": {"audio_tower", "audio_model", "audio_encoder", "audio"},
    "mtp": {"mtp", "mtp_head", "multi_token_prediction"},
}


def _aux_kind(name):
    segments = set(name.lower().split("."))
    for kind, tokens in _AUX_SEGMENTS.items():
        if segments.intersection(tokens):
            return kind
    return None


def text_only_index(index, linears, *, text_only=False):
    if not text_only:
        return index
    modeled_kinds = {_aux_kind(name) for name in linears}
    ignored_kinds = set(_AUX_SEGMENTS) - modeled_kinds
    kept = {
        name: record for name, record in index['layers'].items()
        if _aux_kind(name) not in ignored_kinds
    }
    if not kept:
        raise ValueError('Text-only filtering removed all CQ records; check checkpoint namespaces')
    filtered = dict(index)
    filtered['layers'] = kept
    return filtered
