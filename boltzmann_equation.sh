#! bash

# Debugging
#python boltzmann_equation.py --name debug --debug --phase test #--dynamic_scaling # --fermi_scaling

# Adam Optimizer
#python boltzmann_equation.py --name adam --phase test --optimizer adam --log_loss

# L-BFGS Optimizer
python boltzmann_equation.py --name l-bfgs --phase test --optimizer l-bfgs --log_loss --dynamic_scaling