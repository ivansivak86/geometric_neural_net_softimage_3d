# Geometric Neural Network - Softimage|3D 4.0 (Native Plugin)

This started as one of those slightly absurd ideas: could the original **SOFTIMAGE|3D 4.0 SDK**, running inside an emulated Windows 2000/Celeron machine, be used to build and animate a geometric neural network? 

![NeuralGraph running inside SOFTIMAGE|3D 4.0](https://github.com/ivansivak86/geometric_neural_net_softimage_3d/blob/main/screen1.png)

NeuralGraph is a small native C++ plug-in that creates a graph directly in the Softimage scene, learns scalar couplings between connected nodes, and bakes the evolution into native geometry, materials, F-curves and shape animation. A second phase freezes the learned edge magnitudes as conductivities and visualizes a PDE-style diffusion process using animated node states, segmented edge gradients and moving flow pulses.

![SI3D|Geometric Neural Animation](https://github.com/ivansivak86/geometric_neural_net_softimage_3d/blob/main/si3d_geo_neural_net.gif)

▶ **[Watch the animation](https://bin.ivansivak.com/media/videos/si3d_geometric_neural_net/si3d_geo_neural_net.mp4)**

This is not a mock-up or an imported animation. Everything remains ordinary, editable SI3D scene data.

The learner is intentionally small, deterministic and graph-specific. It uses derivative-free coordinate descent rather than a modern EGNN/e3nn stack, so this is an educational visualization experiment rather than a production machine-learning framework.

One part I enjoyed just as much as the neural-network side was revisiting the old Softimage engineering: a `.cus` file declares the menu, dialog and native entry point; one C++ file implements the behavior; and the host provides geometry, materials, animation, persistence and rendering. It is a good reminder that older systems often had to do much more with much less.

## Requirements

- SOFTIMAGE|3D 4.0 and SDK 4.0
- Microsoft Visual C++ 6.0
- Windows NT/2000; tested on Windows 2000 under 86Box

## Build

Copy the files to:

```text
C:\Softimage\SDK_4.0\GDK\examples\src\NeuralGraph
```

Close Softimage and run:

```text
build_and_install_neural_graph.bat
```

Restart Softimage, then open:

```text
Model → Effect → NeuralGraph +
```

The repository contains the C++ implementation, `.cus` dialog/entry-point declaration, VC6 `mkfile.nt`, and build/install helpers.

## Note

Dense bakes can create thousands of native scene objects and animation keys. On emulated period hardware, high node, segment and epoch counts may take several hours.

