import torch


TORCH_VERSION = torch.__version__


def digit_version(version_str):
    version = version_str.split("+")[0]
    parts = []
    for item in version.replace("rc", ".").split("."):
        if item.isdigit():
            parts.append(int(item))
        else:
            number = "".join(ch for ch in item if ch.isdigit())
            if number:
                parts.append(int(number))
    return tuple(parts)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_apply(func, *args, **kwargs):
    pfunc = lambda *items: func(*items, **kwargs) if kwargs else func(*items)
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    return tensor


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


def deprecated_api_warning(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value
