def _identity_decorator(*dargs, **dkwargs):
    if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
        return dargs[0]

    def wrap(func):
        return func

    return wrap


auto_fp16 = _identity_decorator
force_fp32 = _identity_decorator
