#! bash

# Debugging
#python boltzmann_equation.py --name debug --debug --phase train --optimizer l-bfgs --hamiltonian # --dynamic_scaling # --fermi_scaling

# Adam Optimizer
python boltzmann_equation.py --name adam_h --phase train --optimizer adam --log_loss \
                             --N_initial 150 --N_boundary 150 --N_collocation 600 \
                             --N_initial_val 150 --N_boundary_val 150 --N_collocation_val 600 \
                             --IC_weight 0.1 --BC_weight 0.1 --PDE_weight 0.25 --learning_rate 0.01 --epochs 5000

# L-BFGS Optimizer
#python boltzmann_equation.py --name l-bfgs_h --phase test --optimizer l-bfgs --log_loss --hamiltonian \
#                             --N_initial 300 --N_collocation 900 --PDE_weight 0.15 --learning_rate 0.01