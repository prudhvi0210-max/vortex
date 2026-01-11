****This project builds on the Quantum Brush project by adding a new Quantum Vortex effect. The full Quantum Brush codebase is included because the Vortex effect reuses existing utilities, files, and dependencies, all of which remain unchanged. The Vortex effect is integrated alongside the existing effects within the project structure.****

# Quantum Vortex

A hybrid **quantum--classical image distortion effect** that applies
localized vortex swirls along a stroke path, with quantum-generated
angles driving the rotation of each segment.

------------------------------------------------------------------------

## Description

**Quantum Vortex** warps image regions around user-drawn strokes using
polar-coordinate remapping.\
Each stroke segment becomes an independent vortex, with its rotation
angle derived from a quantum circuit.

During early experimentation, the initial *vortex* implementation
produced primarily **linear translation effects** along stroke paths.
This behavior was caused by global scaling dilution and under-amplified
rotational factors, which resulted in subtle pixel displacements rather
than true circular motion.

In the later **vortex extension**, refinements such as localized radial
normalization and boosted angular multipliers transformed the effect
into a coherent swirling distortion. Entangled, quantum-generated
rotations now dynamically twist and blend colors around stroke centers,
enabling visually distinct, controllable vortices with user-tuned
intensity.

------------------------------------------------------------------------

## How It Works

1.  **Stroke Segmentation**\
    User click points split the stroke path into multiple segments. Each
    segment is processed independently.

2.  **Quantum Angle Generation**\
    A quantum circuit prepares qubits in superposition and entangles
    them via a shared ancilla.\
    

3.  **Polar Swirl Mapping**\
    Pixels near each stroke segment are transformed into polar
    coordinates around a computed center point.\
    Their angular component is rotated proportionally to distance from
    the center and mapped back using bilinear interpolation.

4.  **Localized Application**\
    Only pixels within a configurable radius around the stroke are
    modified, preserving surrounding image structure.

------------------------------------------------------------------------

## Parameters

  -----------------------------------------------------------------------
  Name            :   Description
  ----------------- -----------------------------------------------------
  **Radius**      :  Controls the size of the vortex region around each
                    stroke segment

  **Strength**     : Scales both the quantum entanglement and swirl
                    intensity

  **Invert         : Reverses the direction of the swirl
  Luminosity**      
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Input

-   RGBA image (`image_rgba`)
-   Stroke path as pixel coordinates
-   Click points defining segment boundaries


------------------------------------------------------------------------

## Output

-   Modified RGBA image with localized vortex distortions applied along
    the stroke path

------------------------------------------------------------------------

## Dependencies

-   NumPy
-   SciPy
-   Qiskit
-   Custom `utils.py` (interpolation, region selection, estimator
    execution)

------------------------------------------------------------------------

## Intended Use

This effect is designed for: - Generative and creative image
processing - Experimental visual tools - Quantum-inspired artistic
workflows



------------------------------------------------------------------------

## Notes

-   The original vortex implementation demonstrates translation-dominant
    behavior
-   The vortex extension introduces true rotational motion and visible
    twirling effects



