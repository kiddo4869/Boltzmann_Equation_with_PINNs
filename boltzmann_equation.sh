#! bash

# Debugging
 python boltzmann_equation.py --name debug --debug --phase train --optimizer l-bfgs #--hamiltonian  # --dynamic_scaling # --fermi_scaling

# Adam Optimizer
#python boltzmann_equation.py --name adam_h --phase test --optimizer adam --log_loss \
#                             --N_initial 1000 --N_collocation 3000 --PDE_weight 0.15 --learning_rate 0.01

# L-BFGS Optimizer
#python boltzmann_equation.py --name l-bfgs_h --phase test --optimizer l-bfgs --log_loss --hamiltonian \
#                             --N_initial 300 --N_collocation 900 --PDE_weight 0.15 --learning_rate 0.01