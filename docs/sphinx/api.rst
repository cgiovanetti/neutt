API reference
=============

Solvers
-------

Each solver is a factory ``make_*_solver(...) -> (solve, grid)`` that builds the kernels once and
returns a jitted ``solve``.

.. autosummary::
   :toctree: generated

   neutt.decoupling
   neutt.hybrid
   neutt.transport.qke.full_coupled

Background and grid
-------------------

.. autosummary::
   :toctree: generated

   neutt.constants
   neutt.grid
   neutt.interp
   neutt.thermo

Collision kernels
-----------------

.. autosummary::
   :toctree: generated

   neutt.transport.nunu
   neutt.transport.nunubar
   neutt.transport.nuer1
   neutt.transport.nuer2
   neutt.transport.nuepma
   neutt.transport.assemble

Flavour mixing (QKE)
--------------------

.. autosummary::
   :toctree: generated

   neutt.transport.qke.coherent
   neutt.transport.qke.nonadiabatic
   neutt.transport.qke.collide_matrix
   neutt.transport.qke.collide
   neutt.transport.qke.collide_nue
   neutt.transport.qke.collide_nunu
   neutt.transport.qke.solve
