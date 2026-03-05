ITS parity and reproducibility
==============================

The ``xyz`` KSG implementation has been cross-checked against ITS toolbox
nearest-neighbor estimators (via Octave + TSTOOL MEX functions).

Compiled functions
------------------

- ``nn_prepare``
- ``nn_search``
- ``range_search``

These are used internally by ITS functions such as ``its_Eknn``, ``its_BTEknn``,
``its_PTEknn`` and ``its_SEknn``.

Reference benchmark
-------------------

Dataset: ``tests/r.csv``, ``k=3``, Chebyshev/max metric.

.. list-table::
   :header-rows: 1

   * - Measure
     - Octave ITS
     - Python ``xyz``
   * - ``Eknn_Hy``
     - ``3.9808891418417671``
     - ``3.9808891418417671``
   * - ``BTE_TE``
     - ``0.032597986659973044``
     - ``0.032597986659972822``
   * - ``PTE_TE``
     - ``-0.054432883408784827``
     - ``-0.054432883408783939``
   * - ``SE_Sy``
     - ``0.02011865351899278``
     - ``0.02011865351899067``

Agreement is within floating-point precision.

Why parity mattered
-------------------

The critical correction was excluding self-neighbors in projected range counts
for KSG TE/PTE/SE, matching ITS ``range_search(..., past=0)`` semantics.
