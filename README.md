## Boltzmann_Equation_with_PINNs

Objectives: Solving some of Boltzmann’s equations in phase space using Physics-Informed Neural Networks (PINNs)

Current progress: we have built a toy model to predict the probability density $f$ of non-interacting particles on phase space ($q$, $p$) at certain time $t$ (after release from a 1D harmonic trap)

### PINNs

Input of the model: a set of variables ($q$, $p$, $t$)
Output of the model: probability density $f$
PDE loss: Boltzmann’s equations

### Data Collection

To train the model, we collected the data points in the dimension of (no. of samples, no. of input variables)
* The blue dots are the initial points ($t = 0s$) for the model to compare the exact solution and its prediction
* The red dots are the collocation points for the model to gain information on derivatives based on its prediction (no exact solution needed)

#### Situation I: Shifted Particle Cloud

#### Situation II: Changing Trapping Potential

#### Situation III: Collision (Adding more terms)

#### Situation IV: High Dimensional Case (2D and 3D)


