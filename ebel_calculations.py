import numpy as np
import xraylib 

# --- Physical Constants from Ebel 1999 Paper ---
CONST_CONTINUUM_EBEL = 1.35e9  # photons / (s * sr * mA * keV)
CONST_K_ALPHA_EBEL = 5e13      # photons / (s * sr * mA)
CONST_K_BETA_EBEL = 5e13       # Paper states same as K-alpha
CONST_L_ALPHA_EBEL = 6.9e13    # photons / (s * sr * mA)
CONST_L_OTHER_EBEL = 6.9e13    # Approximation for other L lines
CONST_M_EBEL = 3e13            # General approximation for M lines

# Initialize xraylib - crucial for it to find its data files
try:
    xraylib.XRayInit()
    print("xraylib initialized successfully.")
except AttributeError:
    print("Warning: xraylib.XRayInit() not found. This might be an older version of xraylib or xraylib is not properly installed/configured.")
except Exception as e:
    print(f"Error during xraylib.XRayInit(): {e}")


def get_material_data(material_symbol_or_Z):
    """
    Fetches Z, Atomic Weight (A), and Density (rho) for a material.
    Input can be atomic number (int) or element symbol (str).
    Returns a dictionary {"Z": Z, "A": A, "rho": rho} or None if not found or if xraylib data is missing.
    """
    try:
        Z_val = 0
        if isinstance(material_symbol_or_Z, str):
            Z_val = xraylib.SymbolToAtomicNumber(material_symbol_or_Z)
            if Z_val == 0: 
                print(f"Warning: Element symbol '{material_symbol_or_Z}' not recognized by xraylib.SymbolToAtomicNumber.")
                return None
        elif isinstance(material_symbol_or_Z, int):
            Z_val = material_symbol_or_Z
            if not (1 <= Z_val <= xraylib.GetMaxAtomicNumber()):
                 print(f"Warning: Atomic number '{Z_val}' out of xraylib's valid range.")
                 return None
        else:
            print(f"Error: Invalid type for material_symbol_or_Z: {type(material_symbol_or_Z)}")
            return None

        A_val = xraylib.AtomicWeight(Z_val)
        rho_val = xraylib.ElementDensity(Z_val)

        if A_val == 0.0: 
            print(f"Warning: xraylib.AtomicWeight({Z_val}) returned 0.0. Check xraylib installation and data files.")
            return None
        if rho_val == 0.0 and Z_val not in [2, 10, 18, 36, 54, 86]: 
             print(f"Warning: xraylib.ElementDensity({Z_val}) returned 0.0 for a non-noble gas element.")

        return {"Z": Z_val, "A": A_val, "rho": rho_val}

    except Exception as e:
        print(f"Error in get_material_data for '{material_symbol_or_Z}': {e}")
        return None

def gaussian_kernel_1d(sigma_bins, radius_factor=3):
    if sigma_bins <= 0:
        return np.array([1.0]) 
    radius_bins = int(np.ceil(radius_factor * sigma_bins))
    kernel_size = 2 * radius_bins + 1
    x = np.arange(kernel_size) - radius_bins
    kernel = np.exp(-x**2 / (2 * sigma_bins**2))
    return kernel / np.sum(kernel) 

def convolve_1d_manual(signal, kernel):
    if len(kernel) == 1 and kernel[0] == 1.0:
        return signal 
    signal_len = len(signal)
    kernel_len = len(kernel)
    kernel_half = kernel_len // 2
    padded_signal = np.pad(signal, (kernel_half, kernel_half), mode='reflect')
    convolved_signal = np.zeros_like(signal, dtype=float)
    for i in range(signal_len):
        convolved_signal[i] = np.sum(padded_signal[i : i + kernel_len] * kernel[::-1]) 
    return convolved_signal

def apply_detector_response(energies_keV, intensities, dE_keV, fwhm_ref_keV, energy_ref_keV):
    if fwhm_ref_keV <= 0 or energy_ref_keV <= 0 or dE_keV <= 0:
        return intensities 
    effective_fwhm_keV = fwhm_ref_keV 
    if effective_fwhm_keV <= 0:
        return intensities
    sigma_keV = effective_fwhm_keV / (2.0 * np.sqrt(2.0 * np.log(2.0))) 
    sigma_bins = sigma_keV / dE_keV
    if sigma_bins < 0.1 : 
        return intensities
    kernel = gaussian_kernel_1d(sigma_bins, radius_factor=3)
    if len(kernel) <= 1 : 
        return intensities
    return convolve_1d_manual(intensities, kernel)

def calculate_ebel_spectrum_data(params):
    """
    Calculates the X-ray spectrum based on the Ebel model.
    Params now includes "anode_A" and "anode_rho_g_cm3".
    """
    anode_Z = params["anode_Z"]
    A_anode = params["anode_A"] 
    # rho_anode_g_cm3 = params["anode_rho_g_cm3"] 

    E0_keV = params["voltage_kv"]
    i_mA = params["current_ma"]
    psi_deg = params["electron_angle_deg"]
    epsilon_deg = params["xray_angle_deg"]
    be_thickness_cm = params["be_window_thickness_mm"] / 10.0
    filter_sym = params.get("filter_material_symbol")
    filter_thickness_cm = params.get("filter_thickness_mm", 0.0) / 10.0
    omega_sr = params["solid_angle_sr"]
    E_min_keV = params["energy_min_keV"]
    dE_keV = params["energy_step_keV"]

    if E0_keV <= E_min_keV: return np.array([E_min_keV, E0_keV]), np.array([0.0, 0.0])
    psi_rad, epsilon_rad = np.deg2rad(psi_deg), np.deg2rad(epsilon_deg)
    if np.sin(psi_rad) <= 1e-6 or np.sin(epsilon_rad) <= 1e-6:
        print("Warning: Electron or X-ray angle results in near-zero sine value.")
        return np.array([E_min_keV, E0_keV]), np.array([0.0, 0.0])

    energies_keV = np.arange(E_min_keV, E0_keV + dE_keV, dE_keV)
    calc_energies_keV = energies_keV[energies_keV < E0_keV]
    if len(calc_energies_keV) == 0: return np.array([E_min_keV, E0_keV]), np.array([0.0, 0.0])
    
    final_intensity_density = np.zeros_like(energies_keV)

    # --- Continuum (Bremsstrahlung) ---
    x_exponent = 1.109 - 0.00435 * anode_Z + 0.00175 * E0_keV
    J_keV = 0.0135 * anode_Z
    J_eV = J_keV * 1000.0
    rho_z_bar_g_cm2 = (A_anode / anode_Z) * \
                      (0.787e-5 * np.sqrt(J_eV) * (E0_keV**1.5) + 0.735e-6 * (E0_keV**2))
    if rho_z_bar_g_cm2 <= 0: rho_z_bar_g_cm2 = 1e-9 

    E_cont = calc_energies_keV
    term_E0_div_E_minus_1 = (E0_keV / E_cont) - 1
    valid_E_mask = term_E0_div_E_minus_1 > 1e-9 
    
    continuum_no_abs = np.zeros_like(E_cont)
    if np.any(valid_E_mask):
        continuum_no_abs[valid_E_mask] = \
            omega_sr * i_mA * CONST_CONTINUUM_EBEL * anode_Z * \
            (term_E0_div_E_minus_1[valid_E_mask] ** x_exponent)

    tau_E_anode_cm2_g = np.array([xraylib.CS_Total(anode_Z, E) for E in E_cont])
    X_cont_denom = tau_E_anode_cm2_g * 2 * rho_z_bar_g_cm2 * np.sin(psi_rad) / np.sin(epsilon_rad)
    
    absorption_factor_cont = np.ones_like(E_cont)
    non_zero_X_mask = np.abs(X_cont_denom) > 1e-9
    absorption_factor_cont[non_zero_X_mask] = \
        (1.0 - np.exp(-X_cont_denom[non_zero_X_mask])) / X_cont_denom[non_zero_X_mask]
    
    continuum_after_abs_calc = continuum_no_abs * absorption_factor_cont
    
    for i, e_calc in enumerate(calc_energies_keV): 
        idx = np.argmin(np.abs(energies_keV - e_calc))
        final_intensity_density[idx] += continuum_after_abs_calc[i]

    # --- Characteristic Lines ---
    shells_and_lines = {
        xraylib.K_SHELL: [xraylib.KA1_LINE, xraylib.KA2_LINE, xraylib.KB1_LINE, xraylib.KB3_LINE],
        xraylib.L1_SHELL: [xraylib.LB3_LINE, xraylib.LB4_LINE, xraylib.LG2_LINE, xraylib.LG3_LINE], 
        xraylib.L2_SHELL: [xraylib.LB1_LINE, xraylib.LG1_LINE, xraylib.L2M1_LINE], # Corrected LETA_LINE to L2M1_LINE
        xraylib.L3_SHELL: [xraylib.LA1_LINE, xraylib.LA2_LINE, xraylib.LB2_LINE, xraylib.LL_LINE], 
    }

    for shell_idx, specific_lines in shells_and_lines.items():
        try:
            E_edge_jk_keV = xraylib.EdgeEnergy(anode_Z, shell_idx)
        except xraylib.XRayLibError: continue 
        if E_edge_jk_keV == 0 or E0_keV <= E_edge_jk_keV: continue
        
        U0_overvoltage = E0_keV / E_edge_jk_keV
        if U0_overvoltage <= 1.0: continue

        if shell_idx == xraylib.K_SHELL: z_shell, b_shell = 2.0, 0.35
        elif shell_idx in [xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL]: z_shell, b_shell = 8.0, 0.25
        else: continue 

        term_U0lnU0 = U0_overvoltage * np.log(U0_overvoltage) if U0_overvoltage > 0 else 0
        term_U0lnU0_plus_1_minus_U0 = term_U0lnU0 - U0_overvoltage + 1.0
        
        inv_S_jk = 0
        if abs(term_U0lnU0_plus_1_minus_U0) > 1e-9 :
            sqrt_U0 = np.sqrt(U0_overvoltage)
            numerator_S_jk_bracket = sqrt_U0 * np.log(U0_overvoltage) + 2.0 * (1.0 - sqrt_U0) if U0_overvoltage > 0 else 0
            
            S_jk_bracket_term_sqrt_arg = 0.0
            if E_edge_jk_keV > 0: 
                 S_jk_bracket_term_sqrt_arg = J_keV / E_edge_jk_keV
            if S_jk_bracket_term_sqrt_arg < 0: S_jk_bracket_term_sqrt_arg = 0 

            S_jk_bracket_term = 1.0 + 16.05 * np.sqrt(S_jk_bracket_term_sqrt_arg) * \
                                (numerator_S_jk_bracket / term_U0lnU0_plus_1_minus_U0)
            inv_S_jk = (z_shell * b_shell / anode_Z) * term_U0lnU0_plus_1_minus_U0 * S_jk_bracket_term
            if inv_S_jk < 0 : inv_S_jk = 0

        R_backscatter = 1.0 - 0.0081517 * anode_Z + 3.613e-5 * (anode_Z**2) + \
                        0.009583 * anode_Z * np.exp(-U0_overvoltage) + 0.001141 * E_edge_jk_keV 
        if R_backscatter < 0 : R_backscatter = 0

        omega_jk_fluoryield = xraylib.FluorYield(anode_Z, shell_idx)

        for line_code in specific_lines:
            try:
                E_line_keV = xraylib.LineEnergy(anode_Z, line_code)
            except xraylib.XRayLibError: continue 
            if E_line_keV == 0 or E_line_keV >= E0_keV: continue
            
            p_jkl_radrate = xraylib.RadRate(anode_Z, line_code)
            if p_jkl_radrate == 0: continue

            const_char_line = CONST_L_OTHER_EBEL 
            if line_code in [xraylib.KA1_LINE, xraylib.KA2_LINE]: const_char_line = CONST_K_ALPHA_EBEL
            elif line_code in [xraylib.KB1_LINE, xraylib.KB3_LINE, xraylib.KB5_LINE]: const_char_line = CONST_K_BETA_EBEL
            elif line_code in [xraylib.LA1_LINE, xraylib.LA2_LINE]: const_char_line = CONST_L_ALPHA_EBEL
            
            tau_E_line_anode_cm2_g = xraylib.CS_Total(anode_Z, E_line_keV)
            X_char_denom = tau_E_line_anode_cm2_g * 2 * rho_z_bar_g_cm2 * np.sin(psi_rad) / np.sin(epsilon_rad)
            
            absorption_factor_char = 1.0
            if abs(X_char_denom) > 1e-9:
                absorption_factor_char = (1.0 - np.exp(-X_char_denom)) / X_char_denom
            
            N_line_photons_s_sr_mA = omega_sr * i_mA * const_char_line * \
                                     inv_S_jk * R_backscatter * omega_jk_fluoryield * \
                                     p_jkl_radrate * absorption_factor_char
            
            if dE_keV > 0:
                line_intensity_density = N_line_photons_s_sr_mA / dE_keV
                bin_index = np.argmin(np.abs(energies_keV - E_line_keV))
                final_intensity_density[bin_index] += line_intensity_density
    
    if be_thickness_cm > 0:
        be_props = get_material_data("Be")
        if be_props:
            mu_be_cm2_g = np.array([xraylib.CS_Total(be_props["Z"], E) for E in energies_keV])
            transmission_be = np.exp(-mu_be_cm2_g * be_props["rho"] * be_thickness_cm)
            final_intensity_density *= transmission_be
    
    if filter_sym and filter_thickness_cm > 0:
        filter_props = get_material_data(filter_sym)
        if filter_props:
            mu_filter_cm2_g = np.array([xraylib.CS_Total(filter_props["Z"], E) for E in energies_keV])
            transmission_filter = np.exp(-mu_filter_cm2_g * filter_props["rho"] * filter_thickness_cm)
            final_intensity_density *= transmission_filter

    final_intensity_density[final_intensity_density < 0] = 0.0

    if params["enable_detector_response"]:
        final_intensity_density = apply_detector_response(
            energies_keV,
            final_intensity_density,
            dE_keV,
            params["detector_fwhm_ref_kev"],
            params["detector_energy_ref_kev"]
        )
    return energies_keV, final_intensity_density

if __name__ == '__main__':
    test_params_direct = {
        "voltage_kv": 30.0,
        "current_ma": 1.0,
        "electron_angle_deg": 90.0,
        "xray_angle_deg": 45.0,
        "be_window_thickness_mm": 0.075,
        "filter_material_symbol": None, 
        "filter_thickness_mm": 0.0,
        "solid_angle_sr": 1.0, 
        "energy_min_keV": 0.1,
        "energy_step_keV": 0.05,
        "enable_detector_response": True,
        "detector_fwhm_ref_kev": 0.150,
        "detector_energy_ref_kev": 5.895 
    }
    
    anode_symbol_for_test = "Rh"
    anode_data_for_test = get_material_data(anode_symbol_for_test)
    if not anode_data_for_test:
        print(f"CRITICAL TEST ERROR: Could not get material data for {anode_symbol_for_test}. Check xraylib.")
        exit()
    
    test_params_direct["anode_material_symbol"] = anode_symbol_for_test 
    test_params_direct["anode_Z"] = anode_data_for_test["Z"]
    test_params_direct["anode_A"] = anode_data_for_test["A"]
    test_params_direct["anode_rho_g_cm3"] = anode_data_for_test["rho"]
    
    print(f"Testing with parameters: {test_params_direct}")
    
    try:
        energies, intensities = calculate_ebel_spectrum_data(test_params_direct)
        
        print("\nEnergies (keV, first/last 10):")
        print(energies[:10]); print("..."); print(energies[-10:])
        print("\nIntensities (photons/s/sr/mA/keV, first/last 10):")
        print(intensities[:10]); print("..."); print(intensities[-10:])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        plt.plot(energies, intensities, label="Calculated Spectrum")
        
        if np.any(intensities > 0): 
            plt.yscale('log')
            plt.ylim(bottom=max(1e-3, np.min(intensities[intensities > 0])/10), top=np.max(intensities)*2) 
        else:
            plt.ylim(bottom=0)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Intensity (photons/s/sr/mA/keV)")
        plt.title(f"Ebel Spectrum: {test_params_direct['anode_material_symbol']} Anode, {test_params_direct['voltage_kv']} kV (Detector Resp: {test_params_direct['enable_detector_response']})")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    except ImportError:
        print("Matplotlib not installed. Skipping plot test.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()