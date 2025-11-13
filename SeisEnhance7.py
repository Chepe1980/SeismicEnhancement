import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class KerogenVelocityModel:
    def __init__(self):
        # Mineral properties (in GPa and g/cc)
        self.mineral_props = {
            'quartz': {'K': 37, 'G': 44, 'rho': 2.65},
            'clay': {'K': 21, 'G': 7, 'rho': 2.58},
            'calcite': {'K': 76.8, 'G': 32, 'rho': 2.71},
            'kerogen': {'K': 2.9, 'G': 2.7, 'rho': 1.3}  # Typical kerogen properties
        }
    
    def hill_average(self, fractions, moduli):
        """Hill's average for elastic moduli mixing"""
        if len(fractions) != len(moduli):
            raise ValueError("Fractions and moduli must have same length")
        
        fractions = np.array(fractions) / np.sum(fractions)
        K_voigt = np.sum(fractions * np.array([m['K'] for m in moduli]))
        G_voigt = np.sum(fractions * np.array([m['G'] for m in moduli]))
        
        K_reuss = 1.0 / np.sum(fractions / np.array([m['K'] for m in moduli]))
        G_reuss = 1.0 / np.sum(fractions / np.array([m['G'] for m in moduli]))
        
        K_hill = (K_voigt + K_reuss) / 2
        G_hill = (G_voigt + G_reuss) / 2
        
        return K_hill, G_hill
    
    def dem_inclusion(self, K_m, G_m, rho_m, K_inc, G_inc, rho_inc, porosity, aspect_ratio):
        """
        Differential Effective Medium for kerogen inclusion
        Simplified implementation for flat inclusions (VTI)
        """
        # Alpha parameter for oblate spheroids (flat inclusions)
        alpha = aspect_ratio
        
        # Eshelby's tensor components for oblate spheroids
        # Simplified expressions for very flat inclusions
        if alpha < 0.1:
            S1111 = 0.5 * (1 - alpha**2 * (1 - np.log(alpha)))
            S3333 = 1 - 2 * S1111
        else:
            # More general expression
            e = np.sqrt(1 - alpha**2)
            S3333 = 1 - 2 * (1 - 2*alpha**2) / (e**3) * np.arcsin(e) + 2 * alpha**2 / (e**2)
            S1111 = (3/(8*e**2)) - ((1-2*alpha**2)/(4*e**3)) * np.arcsin(e) - alpha**2/(4*e**2)
        
        # DEM iteration (simplified)
        f = porosity
        K_eff = K_m * (1 - f) + K_inc * f  # Simplified linear mix
        G_eff = G_m * (1 - f) + G_inc * f  # Simplified linear mix
        
        # Apply anisotropy factor for flat inclusions
        # Flat inclusions reduce vertical stiffness more than horizontal
        anisotropy_factor = 1.0 / (1.0 + 2.0 * f * (1 - aspect_ratio))
        
        # For VTI medium, we need different moduli for different directions
        K_eff_v = K_eff * anisotropy_factor  # More reduction in vertical direction
        K_eff_h = K_eff * (1 - 0.3 * f * (1 - aspect_ratio))  # Less reduction in horizontal
        
        return K_eff_h, K_eff_v, G_eff
    
    def calculate_elastic_constants(self, K_h, K_v, G, rho):
        """Calculate VTI elastic constants from bulk and shear moduli"""
        # For VTI medium approximation
        C33 = K_v + (4/3) * G  # Vertical P-wave modulus
        C11 = K_h + (4/3) * G  # Horizontal P-wave modulus  
        C44 = G  # Vertical shear modulus
        C66 = G  # Horizontal shear modulus (simplified)
        C13 = K_v - (2/3) * G  # Cross-term (approximate)
        
        return {
            'C11': C11, 'C33': C33, 'C44': C44, 
            'C66': C66, 'C13': C13, 'rho': rho
        }
    
    def calculate_velocities(self, elastic_constants):
        """Calculate velocities from elastic constants"""
        C11 = elastic_constants['C11']
        C33 = elastic_constants['C33'] 
        C44 = elastic_constants['C44']
        C66 = elastic_constants['C66']
        rho = elastic_constants['rho']
        
        Vp_vertical = np.sqrt(C33 / rho)
        Vp_horizontal = np.sqrt(C11 / rho)
        Vs_vertical = np.sqrt(C44 / rho)
        Vs_horizontal = np.sqrt(C66 / rho)
        
        return Vp_vertical, Vp_horizontal, Vs_vertical, Vs_horizontal

def process_well_data(csv_file):
    """Main function to process well data and model kerogen effects"""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Initialize model
    model = KerogenVelocityModel()
    
    # Results storage
    results_with_kerogen = []
    results_without_kerogen = []
    
    # Process each depth point
    for idx, row in df.iterrows():
        # Extract data
        Vp_measured = row['Vp']
        Vs_measured = row['Vs'] 
        rho_measured = row['density']
        TOC = row['TOC']
        vcl = row['vcl']
        kerogen_vol = row['kerogen']  # Kerogen volume fraction
        
        # Assume mineral composition (simplified - adjust based on your data)
        quartz_vol = 0.6 * (1 - vcl - kerogen_vol)  # 60% of non-clay, non-kerogen
        calcite_vol = 0.4 * (1 - vcl - kerogen_vol)  # 40% of non-clay, non-kerogen
        clay_vol = vcl
        
        # Case 1: Without kerogen (replace kerogen with quartz)
        fractions_no_kerogen = [quartz_vol + kerogen_vol, calcite_vol, clay_vol]
        minerals_no_kerogen = [
            model.mineral_props['quartz'],
            model.mineral_props['calcite'], 
            model.mineral_props['clay']
        ]
        
        K_no_kerogen, G_no_kerogen = model.hill_average(fractions_no_kerogen, minerals_no_kerogen)
        rho_no_kerogen = (fractions_no_kerogen[0] * model.mineral_props['quartz']['rho'] + 
                         fractions_no_kerogen[1] * model.mineral_props['calcite']['rho'] + 
                         fractions_no_kerogen[2] * model.mineral_props['clay']['rho'])
        
        # Case 2: With kerogen
        fractions_with_kerogen = [quartz_vol, calcite_vol, clay_vol, kerogen_vol]
        minerals_with_kerogen = [
            model.mineral_props['quartz'],
            model.mineral_props['calcite'],
            model.mineral_props['clay'],
            model.mineral_props['kerogen']
        ]
        
        K_matrix, G_matrix = model.hill_average([quartz_vol, calcite_vol, clay_vol], 
                                              minerals_with_kerogen[:3])
        rho_matrix = (quartz_vol * model.mineral_props['quartz']['rho'] + 
                     calcite_vol * model.mineral_props['calcite']['rho'] + 
                     clay_vol * model.mineral_props['clay']['rho'])
        
        # Add kerogen as inclusion
        K_h_with, K_v_with, G_with = model.dem_inclusion(
            K_matrix, G_matrix, rho_matrix,
            model.mineral_props['kerogen']['K'],
            model.mineral_props['kerogen']['G'], 
            model.mineral_props['kerogen']['rho'],
            kerogen_vol, aspect_ratio=0.1  # Flat kerogen inclusions
        )
        
        rho_with_kerogen = rho_matrix * (1 - kerogen_vol) + model.mineral_props['kerogen']['rho'] * kerogen_vol
        
        # Calculate elastic constants
        elastic_no_kerogen = model.calculate_elastic_constants(
            K_no_kerogen, K_no_kerogen, G_no_kerogen, rho_no_kerogen
        )
        elastic_with_kerogen = model.calculate_elastic_constants(
            K_h_with, K_v_with, G_with, rho_with_kerogen
        )
        
        # Calculate velocities
        Vp_v_no, Vp_h_no, Vs_v_no, Vs_h_no = model.calculate_velocities(elastic_no_kerogen)
        Vp_v_with, Vp_h_with, Vs_v_with, Vs_h_with = model.calculate_velocities(elastic_with_kerogen)
        
        # Store results
        results_without_kerogen.append({
            'depth': idx,
            'Vp_vertical': Vp_v_no, 'Vp_horizontal': Vp_h_no,
            'Vs_vertical': Vs_v_no, 'Vs_horizontal': Vs_h_no,
            'elastic': elastic_no_kerogen
        })
        
        results_with_kerogen.append({
            'depth': idx, 
            'Vp_vertical': Vp_v_with, 'Vp_horizontal': Vp_h_with,
            'Vs_vertical': Vs_v_with, 'Vs_horizontal': Vs_h_with, 
            'elastic': elastic_with_kerogen
        })
    
    return df, results_without_kerogen, results_with_kerogen

def plot_results(df, results_without, results_with):
    """Plot the comparison results"""
    
    depths = df.index
    Vp_without = [r['Vp_vertical'] for r in results_without]
    Vp_with = [r['Vp_vertical'] for r in results_with]
    Vs_without = [r['Vs_vertical'] for r in results_without] 
    Vs_with = [r['Vs_vertical'] for r in results_with]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Vp comparison
    ax1.plot(Vp_without, depths, 'b-', label='Without Kerogen', linewidth=2)
    ax1.plot(Vp_with, depths, 'r-', label='With Kerogen', linewidth=2)
    ax1.plot(df['Vp'], depths, 'k--', label='Measured', linewidth=1)
    ax1.set_xlabel('Vp (km/s)')
    ax1.set_ylabel('Depth')
    ax1.set_title('P-wave Velocity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot Vs comparison  
    ax2.plot(Vs_without, depths, 'b-', label='Without Kerogen', linewidth=2)
    ax2.plot(Vs_with, depths, 'r-', label='With Kerogen', linewidth=2)
    ax2.plot(df['Vs'], depths, 'k--', label='Measured', linewidth=1)
    ax2.set_xlabel('Vs (km/s)')
    ax2.set_ylabel('Depth')
    ax2.set_title('S-wave Velocity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot velocity difference
    vp_diff = np.array(Vp_without) - np.array(Vp_with)
    vs_diff = np.array(Vs_without) - np.array(Vs_with)
    
    ax3.plot(vp_diff, depths, 'g-', linewidth=2)
    ax3.set_xlabel('ΔVp (Without - With) km/s')
    ax3.set_ylabel('Depth')
    ax3.set_title('P-wave Velocity Reduction due to Kerogen')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    ax4.plot(vs_diff, depths, 'g-', linewidth=2)
    ax4.set_xlabel('ΔVs (Without - With) km/s')
    ax4.set_ylabel('Depth')
    ax4.set_title('S-wave Velocity Reduction due to Kerogen')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # Plot elastic constants at a specific depth (e.g., middle of formation)
    mid_depth = len(results_with) // 2
    
    elastic_without = results_without[mid_depth]['elastic']
    elastic_with = results_with[mid_depth]['elastic']
    
    constants = ['C11', 'C33', 'C44', 'C66', 'C13']
    values_without = [elastic_without[c] for c in constants]
    values_with = [elastic_with[c] for c in constants]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(constants))
    width = 0.35
    
    ax.bar(x_pos - width/2, values_without, width, label='Without Kerogen', alpha=0.7)
    ax.bar(x_pos + width/2, values_with, width, label='With Kerogen', alpha=0.7)
    
    ax.set_xlabel('Elastic Constants')
    ax.set_ylabel('Value (GPa)')
    ax.set_title(f'Elastic Constants Comparison at Depth {mid_depth}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(constants)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "well_log_data.csv"
    
    try:
        # Process the data
        df, results_without_kerogen, results_with_kerogen = process_well_data(csv_file)
        
        # Plot results
        plot_results(df, results_without_kerogen, results_with_kerogen)
        
        # Print summary statistics
        print("Summary Statistics:")
        print(f"Average Vp reduction due to kerogen: {np.mean([r1['Vp_vertical'] - r2['Vp_vertical'] for r1, r2 in zip(results_without_kerogen, results_with_kerogen)]):.3f} km/s")
        print(f"Average Vs reduction due to kerogen: {np.mean([r1['Vs_vertical'] - r2['Vs_vertical'] for r1, r2 in zip(results_without_kerogen, results_with_kerogen)]):.3f} km/s")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Please check your CSV file has columns: Vp, Vs, density, TOC, vcl, kerogen")
