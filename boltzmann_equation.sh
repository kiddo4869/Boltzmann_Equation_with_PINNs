#! bash

# Debugging
#python boltzmann_equation.py --name debug --debug --phase train --hamiltonian # --dynamic_scaling # --fermi_scaling

# Adam Optimizer
python boltzmann_equation.py --name adam_h --phase train --optimizer adam --log_loss --hamiltonian #--dynamic_scaling #--fermi_scaling

# L-BFGS Optimizer
#python boltzmann_equation.py --name l-bfgs --phase test --optimizer l-bfgs --log_loss #--dynamic_scaling