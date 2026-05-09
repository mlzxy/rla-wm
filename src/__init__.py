import importlib

__submodules = {
    "datasets",
    "models",
    "modules",
    "pipelines",
    "renderers",
    "representations",
    "trainers",
    "utils",
}

__all__ = sorted(__submodules)


def __getattr__(name):
    if name not in __submodules:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
