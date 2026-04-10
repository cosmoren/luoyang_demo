"""Defer import of ``loader_test`` so ``train.py`` only imports dataset symbols from ``dataloader.luoyang``."""


def run_pv_loader_test(**kwargs):
    from dataloader.luoyang import loader_test

    return loader_test(**kwargs)
