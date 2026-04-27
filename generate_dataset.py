import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import lptn

def simulate_random_drive_cycles(number_of_cycles, duration, max_loss, t_initial, capacitances, resistances, t_coolant, t_ambient):
    all_data = []

    for cycle in range(1, number_of_cycles + 1):
        # 1. Create Random Walk Load
        time_points = np.arange(0, duration + 1)
        loads = [0.5] # Start at 50% load

        for _ in range(duration):
            # Small step: -0.01, 0, or 0.01
            step = np.random.choice([-0.01, 0.0, 0.01])
            new_load = np.clip(loads[-1] + step, 0, 1) # Keep between 0 and 1
            loads.append(new_load)
        
        # 2. Create Power Interpolation function
        power_vals = np.array(loads) * max_loss
        power_func = interp1d(time_points, power_vals, kind='linear')

        # 3. Simulate
        solution = solve_ivp(lptn.lptn_system, (0, duration), t_initial, 
                             args=(capacitances, resistances, power_func, t_coolant, t_ambient), 
                             method='RK45', t_eval=time_points)

        # 4. Collect Data
        for i in range(len(solution.t)):
            all_data.append({
                'timestamp': solution.t[i],
                'drive_cycle_number': cycle,
                'load': loads[i],
                't_stator': solution.y[0][i],
                't_rotor_1': solution.y[1][i],
                't_rotor_2': solution.y[2][i],
                't_housing': solution.y[3][i]
            })
            
    # 5. Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv('drive_cycle_dataset.csv', index=False)
    print("Simulation complete. Data saved to 'drive_cycle_dataset.csv'")
    return df

# --- Parameters Configuration ---

# Capacitances (J/K)
# Order: [Stator, Rotor_1, Rotor_2, Housing]
capacitances = [800, 200, 200, 1500]

# Resistances (K/W)
# Order: [R_stator_rotor1, R_stator_rotor2, R_stator_housing, R_stator_coolant, 
#         R_rotor1_housing, R_rotor2_housing, R_housing_ambient]
resistances = [3.0, 3.0, 0.3, 0.05, 0.8, 0.8, 0.6] 

# Initial Temperatures [t_stator, t_rotor_1, t_rotor_2, t_housing]
t_initial = [25, 25, 25, 25]

df = simulate_random_drive_cycles(
    number_of_cycles=5, 
    duration=3600, 
    max_loss=1500, 
    t_initial=t_initial, 
    capacitances=capacitances, 
    resistances=resistances, 
    t_coolant=25, 
    t_ambient=20
)