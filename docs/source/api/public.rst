Public API
==========

Top-level exports from ``xyz``:

**Entropy and mutual information**
- ``MVNEntropy``, ``MVNMutualInformation``
- ``KSGEntropy``, ``KSGMutualInformation``
- ``GaussianCopulaMutualInformation``, ``GaussianCopulaConditionalMutualInformation``
- ``MVKSGCondEntropy``, ``MVKSGCondMutualInformation``, ``DirectKSGConditionalMutualInformation``
- ``MVKSGPartialInformationDecomposition``

**Transfer entropy and self-entropy**
- ``GaussianTransferEntropy``, ``GaussianPartialTransferEntropy``, ``GaussianSelfEntropy``
- ``GaussianCopulaTransferEntropy``
- ``KSGTransferEntropy``, ``KSGPartialTransferEntropy``, ``KSGSelfEntropy``
- ``KernelTransferEntropy``, ``KernelPartialTransferEntropy``, ``KernelSelfEntropy``
- ``DiscreteTransferEntropy``, ``DiscretePartialTransferEntropy``, ``DiscreteSelfEntropy``
- ``MVKSGTransferEntropy``

**Model selection and workflows**
- ``RagwitzEmbeddingSearchCV``, ``InteractionDelaySearchCV``
- ``GreedySourceSelectionTransferEntropy``
- ``EnsembleTransferEntropy``, ``GroupTEAnalysis``
- ``SurrogatePermutationTest``, ``BootstrapEstimate``
- ``generate_surrogates``, ``fdr_bh``, ``bonferroni``

Import pattern:

.. code-block:: python

   from xyz import (
       GaussianTransferEntropy,
       GaussianCopulaMutualInformation,
       BootstrapEstimate,
       InteractionDelaySearchCV,
       KSGEntropy,
       KSGMutualInformation,
       MVNEntropy,
   )
