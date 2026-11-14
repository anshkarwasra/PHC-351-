"""Small shared wrapper for optional-looking, consistent tqdm progress bars."""
from tqdm.auto import tqdm


def progress(iterable=None, **kwargs):
    """Wrap an iterable or return a manually-updated tqdm bar."""
    return tqdm(iterable, **kwargs)
