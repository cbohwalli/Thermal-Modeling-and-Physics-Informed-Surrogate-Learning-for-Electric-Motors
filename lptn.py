def lptn_system(t, temperatures, capacitances, resistances, power_func, t_coolant, t_ambient):
    """
    Computes the derivatives for the LPTN system
    temperatures: [t_stator, t_rotor_1, t_rotor_2, t_housing]
    """

    p_stator = power_func(t)

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