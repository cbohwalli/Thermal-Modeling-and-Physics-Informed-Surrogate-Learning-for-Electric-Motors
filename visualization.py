import matplotlib.pyplot as plt

#todo - add visualization for a drive cycle

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