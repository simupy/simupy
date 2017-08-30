---
title: 'SimuPy: A python framework for modeling and simulating dynamical systems'
tags:
  - Python
  - simulation
  - block diagram
authors:
 - name: Benjamin W. L. Margolis
   orcid: 0000-0001-5602-1888
   affiliation: 1
affiliations:
 - name: University of California, Davis
   index: 1
date: 29 August 2017
bibliography: paper.bib
---

# Summary

Numerical simulation is an important part of the design and analysis of dynamical systems, and has become fundamental to the education, practice, and research of many science and engineering disciplines. Especially in systems and controls engineering, model- and systems-based design and simulation have become a part of the dominant workflow as exemplified by tools like Simulink [@Simulink].

SimuPy is a framework for simulating interconnected dynamical system models and provides an open source, python-based tool that can be used in the model-based design and simulation workflow. Using SimuPy, it is easy to implement software representations of dynamical systems from numeric functions or from symbolic expressions using SymPy. SimuPy provides an API to connect these models in block diagrams. The aggregate dynamics are automatically combined and can be simulated using the ordinary differential equation solvers. SimuPy can also coarsely handle event-based discontinuous dynamics.

The author has used SimuPy in numerical studies of nonlinear tracking control [@margolis2016aerompc, @margolis2017threetracking]. The examples and tests show that the block diagram algebra and timing formulations agree with the mathematical definitions.


# References
