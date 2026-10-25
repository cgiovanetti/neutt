"""neutt — the JAX/GPU port of BURST's neutrino transport: N_eff and the neutrino spectra through
weak decoupling, with finite-T QED and the flavour-mixing (QKE) refinement.

float64 is mandatory (the collision operator is a forward-minus-reverse near-cancellation and
float32 fails the conservation sum rules), so it is enabled here, at import: import neutt before
creating any jax array.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

_NEWT = r"""
                          .-''-
           _.-''```'-.---'   0  \
        .-'                     /
     .-'               _.----''
   /             \  /'    \\
 /    _\ \'''---' \\      //\     neutt is compiling...
|   /   ` \ _      `. _
 \  \___  /\        /\
   `-..__``-.._
         ``-..__`-.     .'
                ``-...'
"""


def compiling() -> None:
    """Print the banner (a newt).  Every jitted ``solve`` calls this first; it runs only while JAX traces, so
    it shows on each (re)compile and never on a cached call."""
    print(_NEWT, flush=True)


from . import constants, grid, interp, thermo, transport  # noqa: E402
