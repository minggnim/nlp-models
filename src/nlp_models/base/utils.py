import importlib
import numpy as np
from torch import Tensor, device


def fullname(o):
    '''
    Give a full name (package_name.class_name) for a class / object in Python.
    Used to load the correct classes from JSON files
    '''
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + '.' + o.__class__.__name__


def import_from_string(dotted_path):
    '''
    Import a dotted module path and return the attribute/class designed by
    the last name in the path. Raise ImportError if the import failed.
    '''
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = f'{dotted_path} doesn\'t look like a module path'
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module {module_path} does not define a {class_name} attribute/class'
        raise ImportError(msg)


def batch_to_device(batch, target_device: device):
    '''
    Send a pytorch batch to a device (CPU/GPU)
    '''
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].squeeze(1).to(target_device)
    return batch


def get_embedding_group(label, cat_lookup, corpus):
    idx_grp = cat_lookup[label]
    corpus_embeddings, question_corpus, answer_corpus = corpus
    corpus_embeddings = corpus_embeddings[idx_grp]
    question_corpus = np.array(question_corpus)[idx_grp].tolist()
    answer_corpus = np.array(answer_corpus)[idx_grp].tolist()
    return corpus_embeddings, question_corpus, answer_corpus
