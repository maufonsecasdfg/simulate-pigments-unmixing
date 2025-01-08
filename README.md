# Pigment Simulation and Unmixing Project

## Project Overview
This project aims to simulate pigments by generating absorption and scattering profiles using the **Kubelka-Munk theory**. The workflow involves:

1. **Pigment Generation:** Create synthetic absorbance and scattering profiles for individual pigments.
2. **Pigment Mixtures:** Simulate pigment mixtures using the Kubelka-Munk theory to calculate resulting absorption, scattering, and reflectance profiles.
3. **Pigment Visualization:** Use colour-science to convert the reflectance curves into human-visible pigment colors.
4. **Unmixing Experiments:** Explore methods for unmixing pigments from simplified synthetic multispectral experimental data, given an initial list of possible pigments and their spectral profiles.

## Goals
- Generate simple, but fairly realistic synthetic absorbance and scattering profiles for pigments.
- Simulate pigment mixtures using Kubelka-Munk theory.
- Test a naive, greedy and bayesian model for finding the components of a mixture of pigments and their weights.

