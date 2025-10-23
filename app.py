from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import io
import csv
from ebel_calculations import calculate_ebel_spectrum_data, get_material_data

app = Flask(__name__)

# --- Default Parameters and Choices ---
DEFAULT_PARAMS = {
    "anode_material": "Rh",
    "voltage_kv": 30.0,
    "current_ma": 10.0,
    "electron_angle_deg": 90.0,
    "xray_angle_deg": 45.0,
    "be_window_thickness_mm": 0.075, # 75 microns
    "filter_type": "None",
    "solid_angle_sr": 1e-6,
    "enable_detector_response": False,
    "detector_fwhm_ref_kev": 0.150, # FWHM at reference energy
    "detector_energy_ref_kev": 5.9    # Reference energy (e.g., Mn Ka)
}

ANODE_MATERIALS = {
    "Cr": {"Z": 24, "symbol": "Cr"}, # Z here is for reference, xraylib is the source of truth
    "Rh": {"Z": 45, "symbol": "Rh"},
    "Ag": {"Z": 47, "symbol": "Ag"},
    "Pd": {"Z": 46, "symbol": "Pd"},
    "Pb": {"Z": 82, "symbol": "Pb"}
}

# Filters: Material, Thickness (mm), Symbol for xraylib
FILTERS = {
    "None": None,
    "C_0.125mm": {"material_symbol": "C", "thickness_mm": 0.125},
    "C_0.5mm": {"material_symbol": "C", "thickness_mm": 0.5},
    "Al_0.130mm": {"material_symbol": "Al", "thickness_mm": 0.130},
    "Pd_0.025mm": {"material_symbol": "Pd", "thickness_mm": 0.025},
    "Pd_0.050mm": {"material_symbol": "Pd", "thickness_mm": 0.050},
    "Pd_0.125mm": {"material_symbol": "Pd", "thickness_mm": 0.125},
    "Cu_0.370mm": {"material_symbol": "Cu", "thickness_mm": 0.370},
    "Cu_0.630mm": {"material_symbol": "Cu", "thickness_mm": 0.630},
}

@app.route('/')
def index():
    """Renders the main page with input forms."""
    return render_template('index.html',
                           default_params=DEFAULT_PARAMS,
                           anode_materials=ANODE_MATERIALS.keys(),
                           filters=FILTERS)

def get_params_from_form(form_data):
    """Helper function to extract and validate parameters from form data."""
    params = {
        "anode_material_symbol": form_data.get('anode_material', DEFAULT_PARAMS['anode_material']),
        "voltage_kv": float(form_data.get('voltage_kv', DEFAULT_PARAMS['voltage_kv'])),
        "current_ma": float(form_data.get('current_ma', DEFAULT_PARAMS['current_ma'])),
        "electron_angle_deg": float(form_data.get('electron_angle_deg', DEFAULT_PARAMS['electron_angle_deg'])),
        "xray_angle_deg": float(form_data.get('xray_angle_deg', DEFAULT_PARAMS['xray_angle_deg'])),
        "be_window_thickness_mm": float(form_data.get('be_window_thickness_mm', DEFAULT_PARAMS['be_window_thickness_mm'])),
        "filter_key": form_data.get('filter_type', DEFAULT_PARAMS['filter_type']),
        "solid_angle_sr": float(form_data.get('solid_angle_sr', DEFAULT_PARAMS['solid_angle_sr'])),
        "enable_detector_response": form_data.get('enable_detector_response') == 'on',
        "detector_fwhm_ref_kev": float(form_data.get('detector_fwhm_ref_kev', DEFAULT_PARAMS['detector_fwhm_ref_kev'])),
        "detector_energy_ref_kev": float(form_data.get('detector_energy_ref_kev', DEFAULT_PARAMS['detector_energy_ref_kev'])),
        "energy_min_keV": 0.1,
        "energy_step_keV": 0.05
    }

    # Validate parameters
    if not (1 <= params["voltage_kv"] <= 100): 
        raise ValueError("Voltage must be between 1 and 100 kV.")
    if params["current_ma"] <= 0:
        raise ValueError("Current must be positive.")
    if not (1 <= params["electron_angle_deg"] <= 90):
        raise ValueError("Electron incidence angle must be between 1 and 90 degrees.")
    if not (1 <= params["xray_angle_deg"] <= 90):
        raise ValueError("X-ray take-off angle must be between 1 and 90 degrees.")
    if params["be_window_thickness_mm"] < 0:
        raise ValueError("Beryllium window thickness cannot be negative.")
    if params["solid_angle_sr"] <= 0:
        raise ValueError("Solid angle must be positive.")
    if params["enable_detector_response"]:
        if params["detector_fwhm_ref_kev"] <= 0:
            raise ValueError("Detector FWHM at reference energy must be positive.")
        if params["detector_energy_ref_kev"] <= 0:
            raise ValueError("Detector reference energy for FWHM must be positive.")
    
    # Get anode properties (Z, A, rho) using the symbol
    anode_data = get_material_data(params["anode_material_symbol"])
    if not anode_data:
        # This error will now be more indicative if xraylib fails early
        raise ValueError(f"Invalid anode material: '{params['anode_material_symbol']}'. Could not retrieve data from xraylib.")
    
    params["anode_Z"] = anode_data["Z"]
    params["anode_A"] = anode_data["A"]
    params["anode_rho_g_cm3"] = anode_data["rho"] # Density in g/cm^3

    # Get filter details
    selected_filter = FILTERS.get(params["filter_key"])
    if selected_filter:
        params["filter_material_symbol"] = selected_filter["material_symbol"]
        params["filter_thickness_mm"] = selected_filter["thickness_mm"]
    else:
        params["filter_material_symbol"] = None
        params["filter_thickness_mm"] = 0.0
    
    return params

@app.route('/calculate_spectrum', methods=['POST'])
def calculate_spectrum_route():
    try:
        params = get_params_from_form(request.form)
        energies, intensities = calculate_ebel_spectrum_data(params)
        chart_data = {
            "labels": [f"{e:.2f}" for e in energies],
            "datasets": [{
                "label": f"{params['anode_material_symbol']} Spectrum ({params['voltage_kv']} kV)",
                "data": intensities.tolist(),
                "borderColor": 'rgb(75, 192, 192)',
                "tension": 0.1,
                "fill": False
            }]
        }
        return jsonify(success=True, data=chart_data)
    except ValueError as ve:
        app.logger.warning(f"Validation error in /calculate_spectrum: {ve}")
        return jsonify(success=False, error=str(ve))
    except Exception as e:
        app.logger.error(f"Error calculating spectrum: {e}", exc_info=True)
        return jsonify(success=False, error=f"An unexpected error occurred: {str(e)}")

@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        params = get_params_from_form(request.form)
        energies, intensities = calculate_ebel_spectrum_data(params)
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Energy (keV)', 'Intensity (photons/s/sr/mA/keV)'])
        for energy, intensity in zip(energies, intensities):
            cw.writerow([f"{energy:.3f}", f"{intensity:.6e}"])
        output = si.getvalue()
        si.close()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition":
                     f"attachment; filename=ebel_spectrum_{params['anode_material_symbol']}_{params['voltage_kv']}kV.csv"})
    except ValueError as ve:
        app.logger.warning(f"CSV Generation - Validation error: {ve}")
        return f"Error generating CSV: {str(ve)}", 400
    except Exception as e:
        app.logger.error(f"Error generating CSV: {e}", exc_info=True)
        return f"Error generating CSV: An unexpected error occurred.", 500

if __name__ == '__main__':
    app.run(debug=True)