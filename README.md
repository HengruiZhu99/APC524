# Class Project For APC524

## Project Title: Solving the Navier Stokes Equations using the Finite Differencing and Finite Volume Methods

### Overview

This repository contains a fully integrated codebase for solving the one-dimensional Navier-Stokes equations simplified under the assumptions of constant pressure, focusing on the balance between inertial, viscous, and external forces. The simplified equations for incompressible, constant-density flow are:

```math

\rho \frac{\partial u}{\partial t} + \rho u \frac{\partial u}{\partial x} = \mu \frac{\partial^2 u}{\partial x^2} + f

```
where $$u(x,t)$$ is the fluid velocity, $$\rho$$ is the constant fluid density, $$\mu$$ is the dynamic viscosity, and $$f$$ is the external body force per unit volume (if applicable).

This project ocusing on using two common methods, the Finite Difference Method and the Finite Volume Method to solve this equation after discretizing the spatial and temporal derivatives. 

### Finite Difference Method

The Finite Difference Method approximates derivatives by replacing them with differential equations at discrete grid points. In the FDM, the domain is divided into evenly spaced grid points, and the governing equation is solved using explicit or implicit time-stepping schemes. The central difference is often used for spatial derivatives, while forward or backward differences are employed for time integration. 

To illustrate this, the first-order spatial derivative can be approximated using a central difference: 

```math
\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}
```

and the second derivative:

```math
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}
```

This method can be simple to implement but faces drawbacks in numerical instability if the time step or the Reynolds number is too large. 

### Finite Difference Method

The Finite Volume Method divides the domain into control volumes, integrating the governing equations over each volume to ensure conservation of mass, momentum, and energy. The fluxes at the control volume boundaries are computed using numerical approximations, ensuring a conservative formulation. For our problem, the conservation form becomes:

```math
\frac{\partial}{\partial t} \int_{\Delta x} \rho u \, dx + \int_{\Delta x} \frac{\partial (\rho u^2)}{\partial x} \, dx = \int_{\Delta x} \frac{\partial \tau}{\partial x} \, dx + \int_{\Delta x} f \, dx
```

where $$\tau = \mu \frac{\partial u}{\partial x}$$ represents the viscous stress. Approximating the fluxes at the control volume interfaces yields a set of algebraic equations that can be solved iteratively.

### File Overview

The /src/navier_stokes/ contain the files fdm_fvm_functions.py and plotting.py. fdm_fvm_functions.py houses the main functions needed to complete the finite difference and finite volume calculations, and plotting.py contains code to plot the results. FDM_FVM.ipynb is a jupyter notebook file that contains a worked example of the code and its output. The /tests/ folder contains testing functions to check that the code is outputting logical results. 



