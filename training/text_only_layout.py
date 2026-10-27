"""Strict text-only filtering of optional CQ towers before layout planning."""
from checkpoint_layout import namespace


def text_only_index(index, linears, *, text_only=False):
    if not text_only:
        return index
    # Only discard entire auxiliary namespaces absent from the skeleton.
    # Never suppress mapping errors for backbone or partially modeled towers.
    modeled = {namespace(name) for name in linears}
    ignored = {'vision', 'audio', 'mtp'} - modeled
    filtered = dict(index)
    filtered['layers'] = {
        name: record for name, record in index['layers'].items()
        if namespace(name) not in ignored
    }
    if not filtered['layers']:
        raise ValueError('Text-only filtering removed all CQ records; check checkpoint namespaces')
    return filtered
