:github_url: https://github.com/bootphon/fastabx

================================
Welcome to fastabx documentation
================================

fastabx is a Python package for efficient computation of ABX discriminability.

The ABX discriminability is measures how well categories of interest are separated in the representation space
by determining whether tokens from the same category are closer to each other than to those from a different category.
While ABX has been mostly used to evaluate speech representations,
it is a generic framework that can be applied to other domains of representation learning.

This package provides a simple interface that can be adapted to any ABX conditions, and to any input modality.

Citation
========

A preprint is available on arXiv: https://arxiv.org/abs/2505.02692

If you use fastabx in your work, please cite it:

.. code-block:: bibtex

   @misc{fastabx,
     title={fastabx: A library for efficient computation of ABX discriminability},
     author={Maxime Poli and Emmanuel Chemla and Emmanuel Dupoux},
     year={2025},
     eprint={2505.02692},
     archivePrefix={arXiv},
     primaryClass={cs.CL},
     url={https://arxiv.org/abs/2505.02692},
   }

Contents
=========

.. toctree::
   :titlesonly:
   :maxdepth: 1

   install
   abx
   guide
   api
   examples/index
   advanced/index
