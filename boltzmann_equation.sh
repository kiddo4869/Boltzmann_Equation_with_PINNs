#! bash

# Debugging
#python boltzmann_equation.py --name debug --debug --phase test --optimizer l-bfgs --hamiltonian input \
#                             --N_initial 200 --N_boundary 200 --N_collocation 100 \
#                             --N_initial_val 200 --N_boundary_val 200 --N_collocation_val 100 \
#                             --IC_weight 1.0 --BC_weight 1.0 --PDE_weight 0.2 --learning_rate 1.0 --epochs 1000

# Adam Optimizer
#python boltzmann_equation.py --name adam_II --phase train --optimizer adam --log_loss --hamiltonian --case_idx 2 \
#                             --N_initial 5000 --N_boundary 5000 --N_collocation 5000 \
#                             --N_initial_val 150 --N_boundary_val 150 --N_collocation_val 3000 \
#                             --IC_weight 0.45 --BC_weight 0.35 --PDE_weight 0.1 --learning_rate 0.01 --epochs 1000

# L-BFGS Optimizer
python boltzmann_equation.py --name l-bfgs_new_II_2 --phase test --optimizer l-bfgs --log_loss --hamiltonian input --case_idx 1 \
                             --N_initial 1500 --N_boundary 100 --N_collocation 50000 \
                             --N_initial_val 650 --N_boundary_val 650 --N_collocation_val 650 \
                             --IC_weight 0.5 --BC_weight 0.0 --PDE_weight 0.5 --learning_rate 0.01 --epochs 10000 \
                             --history_size 100 --tolerance_grad 1e-8 --tolerance_change 1e-8