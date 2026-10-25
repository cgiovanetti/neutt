neutt
=====

``neutt`` is the differentiable JAX/GPU port of the neutrino-transport sector of BURST
(Grohs, Fuller, Kishimoto, Paris & Vlasenko 2016): :math:`N_{\rm eff}` and the neutrino spectral
distortions through weak decoupling, with finite-temperature QED and the flavour-mixing (QKE)
refinement, and, coupled to LINX, the light-element abundances.

These pages are the API reference, generated from the docstrings.  The physics, the numerical
design and every option are documented in the reference manual, ``docs/burst_manual.pdf`` in the
repository.

Install with ``pip install neutt`` (CPU) or ``pip install "neutt[cuda12]"`` (NVIDIA GPU).

.. toctree::
   :maxdepth: 2

   api
