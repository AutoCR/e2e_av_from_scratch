from uniad.core.tensor_utils import ConfigDict, TORCH_VERSION, deprecated_api_warning, digit_version, to_2tuple


class _ExtLoader:
    def load_ext(self, *args, **kwargs):
        class _MissingExt:
            def __getattr__(self, name):
                raise RuntimeError("Standalone UniAD was built without MMCV CUDA extensions.")

        return _MissingExt()


ext_loader = _ExtLoader()

__all__ = [
    "ConfigDict",
    "TORCH_VERSION",
    "deprecated_api_warning",
    "digit_version",
    "ext_loader",
    "to_2tuple",
]
