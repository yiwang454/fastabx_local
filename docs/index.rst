:github_url: https://github.com/bootphon/fastabx

================================
Welcome to fastabx documentation
================================

fastabx is a Python package to perform the ABX discrimination test, and do it fast.

.. image:: ./static/abx_light.svg
   :width: 70%
   :align: center
   :class: only-light

.. image:: ./static/abx_dark.svg
   :width: 70%
   :align: center
   :class: only-dark

.. list-table:: Example of valid triples for various ABX tasks.
   :widths: 70 10 10 10
   :header-rows: 1

   * - Task
     - :math:`a`
     - :math:`b`
     - :math:`x`
   * - ON fruit
     - 🍎
     - 🍋
     - 🍏
   * - ON color
     - 🍋‍🟩
     - 🍎
     - 🍐
   * - ON fruit, BY color
     - 🍎
     - 🍓
     - 🍎
   * - ON fruit, BY color, ACROSS size
     - 🍏
     - 🍐
     - 🍏

Contents
=========

.. toctree::
   :titlesonly:
   :maxdepth: 1

   install
   guide
   examples/index
   api
   slicing
