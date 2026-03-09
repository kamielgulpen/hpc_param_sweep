"""
Stub module for deserializing .pkl files that contain asnu objects.

The pkl network files were created with objects from the 'asnu' package.
This stub lets pickle reconstruct those objects without needing the real package.
The objects only need to expose a .graph (nx.Graph) attribute, which is preserved
through __setstate__ from the original pickle data.
"""


class _Stub:
    """Generic stub for any asnu class. Preserves all original attributes via pickle state."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)


def __getattr__(name):
    """Dynamically create a stub class for any asnu class referenced in pkl files."""
    return type(name, (_Stub,), {})
