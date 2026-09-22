__all__ = ["PointNet2SemSeg"]


def __getattr__(name):
    if name == "PointNetSegmentor":
        from .inference import PointNetSegmentor
        return PointNetSegmentor
    raise AttributeError(name)
