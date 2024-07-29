#! bash

# Adam Optimizer
python boltzmann_equation.py --name adam --phase test --optimizer adam --log_loss

# L-BFGS Optimizer
#python boltzmann_equation.py --name l-bfgs --phase test --optimizer l-bfgs --log_loss