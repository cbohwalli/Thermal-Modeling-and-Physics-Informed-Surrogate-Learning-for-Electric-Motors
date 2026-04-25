import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lptn_system(t, temperatures, capacitances, resistances, p_stator, t_coolant, t_ambient):
    """
    Computes the derivatives for the LPTN system
    temperatures: [t_stator, t_rotor_1, t_rotor_2, t_housing]
    """

    # Unapack Temperatures
    t_stator, t_rotor_1, t_rotor_2, t_housing = temperatures
    
    # Unpack Capacitances
    c_stator, c_rotor_1, c_rotor_2, c_housing = capacitances
    
    # Unpack Resistances
    r_stator_rotor1, r_stator_rotor2, r_stator_housing, r_stator_coolant, r_rotor1_housing, r_rotor2_housing, r_housing_ambient = resistances
    
    # Calculate derivatives
    d_t_stator_dt = (1/c_stator) * (p_stator + 
                                    (t_rotor_1 - t_stator)/r_stator_rotor1 + 
                                    (t_rotor_2 - t_stator)/r_stator_rotor2 + 
                                    (t_housing - t_stator)/r_stator_housing + 
                                    (t_coolant - t_stator)/r_stator_coolant)
    
    d_t_rotor_1_dt = (1/c_rotor_1) * ((t_stator - t_rotor_1)/r_stator_rotor1 + 
                                      (t_housing - t_rotor_1)/r_rotor1_housing)
    
    d_t_rotor_2_dt = (1/c_rotor_2) * ((t_stator - t_rotor_2)/r_stator_rotor2 + 
                                      (t_housing - t_rotor_2)/r_rotor2_housing)
    
    d_t_housing_dt = (1/c_housing) * ((t_stator - t_housing)/r_stator_housing + 
                                      (t_rotor_1 - t_housing)/r_rotor1_housing + 
                                      (t_rotor_2 - t_housing)/r_rotor2_housing + 
                                      (t_ambient - t_housing)/r_housing_ambient)
    
    return [d_t_stator_dt, d_t_rotor_1_dt, d_t_rotor_2_dt, d_t_housing_dt]

# --- Parameters Configuration ---

# Capacitances (J/K)
# Order: [Stator, Rotor_1, Rotor_2, Housing]
capacitances = [800, 200, 200, 1500]

# Resistances (K/W)
# Order: [R_stator_rotor1, R_stator_rotor2, R_stator_housing, R_stator_coolant, 
#         R_rotor1_housing, R_rotor2_housing, R_housing_ambient]
resistances = [3.0, 3.0, 0.3, 0.05, 0.8, 0.8, 0.6] 

# Inputs
p_stator = 200    # Power input to Stator (W)
t_coolant = 25    # Coolant temp (C)
t_ambient = 20    # Ambient temp (C)

# Initial Temperatures [t_stator, t_rotor_1, t_rotor_2, t_housing]
t_initial = [25, 25, 25, 25]
t_span = (0, 3600) 

# --- Execution ---
solution = solve_ivp(lptn_system, t_span, t_initial, 
                     args=(capacitances, resistances, p_stator, t_coolant, t_ambient), 
                     method='RK45')

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='Stator')
plt.plot(solution.t, solution.y[1], label='Rotor 1')
plt.plot(solution.t, solution.y[2], label='Rotor 2')
plt.plot(solution.t, solution.y[3], label='Housing')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('LPTN Thermal Response')
plt.legend()
plt.grid(True)
plt.savefig('thermal_simulation_results.png', dpi=300)
print("Simulation complete. Plot saved as 'thermal_simulation_results.png'")