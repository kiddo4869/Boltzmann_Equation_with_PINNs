## Boltzmann_Equation_with_PINNs

Objectives: To solve Boltzmann’s equation in phase space using Physics-Informed Neural Networks (PINNs) in different situation.

Current progress: we have built a toy model to predict the probability density $f$ of non-interacting particles on phase space ($q$, $p$) at certain time $t$ (after release from a 1D harmonic trap)

### Boltzmann Equation

$$\begin{aligned}
\frac{df}{dt} 
&=
\Big[ -\frac{\hbar\textbf{k}}{m}\nabla_{\textbf{r}} + \frac{\nabla_{\textbf{r}}U_{ext}\cdot\nabla_{\textbf{k}}}{\hbar}\Big]f + \frac{I_{inel}[f]}{V} + \frac{I_{el}[f]}{V^3} \\\\
&=
-\frac{\hbar\textbf{k}}{m}\nabla_{\textbf{r}}f \\\\
\end{aligned}$$

### PINNs

Input of the model: a set of variables ($q$, $p$, $t$)
Output of the model: probability density $f$

PDE loss: Boltzmann’s equations

### Data Collection

To train the model, we collected the data points in the dimension of (no. of samples, no. of input variables)
* The blue dots are the initial points ($t = 0s$) for the model to compare the exact solution and its prediction
* The red dots are the collocation points for the model to gain information on derivatives based on its prediction (no exact solution needed)

#### Situation I: 1D Harmonic Trap (Base)

$$\frac{df}{dt} = -\frac{p}{m}\frac{\partial f}{\partial q}$$

#### Situation II: Shifted Particle Cloud

#### Situation III: Changing Trapping Potential

#### Situation IV: Include Collisions (Adding more terms)

$$\frac{df}{dt} = \Big[ -\frac{\hbar\textbf{k}}{m}\nabla_{\textbf{r}} + \frac{\nabla_{\textbf{r}}U_{ext}\cdot\nabla_{\textbf{k}}}{\hbar}\Big]f + \frac{I_{inel}[f]}{V} + \frac{I_{el}[f]}{V^3}$$

#### Situation V: High Dimensional Case (2D and 3D)

$$\frac{df}{dt} = -\frac{1}{m}\Big(p_x\frac{\partial f}{\partial x} + p_y\frac{\partial f}{\partial y} + p_z\frac{\partial f}{\partial z}\Big)$$

### Acknowledgement

* Raissi, M., & Perdikaris, P. (2024). A hands-on introduction to Physics-Informed Neural
Networks for solving partial differential equations with benchmark tests taken from
astrophysics and plasma physics. arXiv preprint arXiv:2403.00599. Retrieved
from https://arxiv.org/html/2403.00599v1#S3
* Raissi, M. (n.d.). PINNs. GitHub. Retrieved from https://github.com/maziarraissi/PINNs
* Omniscientoctopus. (n.d.). Physics-Informed-Neural-Networks. GitHub. Retrieved
from https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks
* Moseley, B. (n.d.). harmonic-oscillator-pinn. GitHub. Retrieved from
https://github.com/benmoseley/harmonic-oscillator-pinn
* Teaching Neural Network to Solve Navier-Stokes Equations. Retrieved from https://github.com/ComputationalDomain/PINNs/blob/main/Cylinder-Wake/NS_PINNS.py
* https://johaupt.github.io/blog/pytorch_lbfgs.html