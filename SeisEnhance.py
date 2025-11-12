import streamlit as st
import numpy as np
import segyio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import os
import time
from tqdm import tqdm
import tempfile
import io

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0  # Default 4ms, adjust if needed
        
    def read_segy(self, filename):
        """Read SEG-Y file and return seismic data as numpy array"""
        try:
            with segyio.open(filename, "r") as segyfile:
                # Get the data shape and handle 2D/3D data
                if segyio.tools.metadata(segyfile).tracecount == segyfile.tracecount:
                    st.info("2D seismic data detected")
                    # Read as 2D data (traces, samples)
                    data = np.stack([segyfile.trace[i] for i in range(segyfile.tracecount)])
                    data = data.reshape(1, data.shape[0], data.shape[1])  # Make it 3D with 1 inline
                else:
                    # Try to read as 3D data
                    data = segyio.tools.cube(segyfile)
                
                st.success(f"SEG-Y file loaded with shape: {data.shape}")
                st.info(f"Data range: {np.min(data):.3f} to {np.max(data):.3f}")
                
                # Get sample rate
                try:
                    self.sample_rate = segyio.tools.dt(segyfile) / 1000.0  # Convert to ms
                    st.info(f"Sample rate: {self.sample_rate} ms")
                except:
                    st.info(f"Using default sample rate: {self.sample_rate} ms")
                
                return data
                
        except Exception as e:
            st.error(f"Error reading SEG-Y file: {e}")
            # Alternative reading method
            try:
                return self.read_segy_alternative(filename)
            except Exception as e2:
                st.error(f"Alternative reading also failed: {e2}")
                return None

    def read_segy_alternative(self, filename):
        """Alternative method to read SEG-Y files"""
        with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
            n_traces = segyfile.tracecount
            n_samples = segyfile.samples.size
            
            # Read all traces
            data = np.zeros((1, n_traces, n_samples))
            for i in range(n_traces):
                data[0, i, :] = segyfile.trace[i]
            
            st.info(f"Alternative read - Shape: {data.shape}")
            return data

    def spectral_blueing(self, seismic_data, target_freq=80, enhancement_factor=1.5,
                        low_freq_boost=1.2, mid_freq_range=(30, 80)):
        """
        Spectral blueing to enhance high frequencies
        Optimized for 2D data (1 inline)
        """
        st.info("Applying spectral blueing...")
        enhanced_data = np.zeros_like(seismic_data)
        
        n_inlines, n_xlines, n_samples = seismic_data.shape
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_traces = n_inlines * n_xlines
        processed_traces = 0
        
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = seismic_data[i, j, :].copy()
                
                # Remove mean and detrend
                trace = signal.detrend(trace)
                
                # FFT
                trace_fft = fft(trace)
                freqs = fftfreq(len(trace), d=self.sample_rate/1000.0)  # Frequency in Hz
                
                # Create frequency-dependent weighting
                weights = np.ones_like(freqs, dtype=complex)
                
                # Boost mid-to-high frequencies more aggressively
                freq_magnitude = np.abs(freqs)
                
                # Low frequencies (5-30 Hz): slight boost
                low_freq_mask = (freq_magnitude > 5) & (freq_magnitude <= mid_freq_range[0])
                weights[low_freq_mask] = low_freq_boost
                
                # Target frequencies (30-80 Hz): moderate boost
                target_mask = (freq_magnitude > mid_freq_range[0]) & (freq_magnitude <= target_freq)
                weights[target_mask] = enhancement_factor
                
                # High frequencies (80+ Hz): strong but controlled boost
                high_freq_mask = freq_magnitude > target_freq
                # Apply roll-off to avoid noise amplification
                rolloff_factor = enhancement_factor * np.exp(-0.001 * (freq_magnitude[high_freq_mask] - target_freq)**2)
                weights[high_freq_mask] = np.maximum(1.0, rolloff_factor)
                
                # Apply weights
                enhanced_fft = trace_fft * weights
                
                # Inverse FFT and take real part
                enhanced_trace = np.real(ifft(enhanced_fft))
                
                # Normalize to preserve amplitude characteristics
                if np.std(enhanced_trace) > 0:
                    enhanced_trace = enhanced_trace * (np.std(trace) / np.std(enhanced_trace))
                
                enhanced_data[i, j, :] = enhanced_trace
                
                # Update progress
                processed_traces += 1
                if processed_traces % 100 == 0:
                    progress = processed_traces / total_traces
                    progress_bar.progress(progress)
                    status_text.text(f"Processing trace {processed_traces}/{total_traces}")
        
        progress_bar.progress(1.0)
        status_text.text("Spectral blueing completed!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
                
        return enhanced_data

    def bandpass_filter(self, seismic_data, lowcut=8, highcut=120, order=3):
        """
        Apply bandpass filter to remove very low and very high frequency noise
        """
        st.info("Applying bandpass filter...")
        
        # Calculate Nyquist frequency correctly
        sampling_interval = self.sample_rate / 1000.0  # Convert ms to seconds
        sampling_freq = 1.0 / sampling_interval  # Sampling frequency in Hz
        nyquist = sampling_freq / 2.0  # Nyquist frequency in Hz
        
        st.info(f"Sampling frequency: {sampling_freq:.1f} Hz")
        st.info(f"Nyquist frequency: {nyquist:.1f} Hz")
        
        # Auto-adjust filter range if needed
        if highcut >= nyquist * 0.95:
            highcut = nyquist * 0.9
            st.warning(f"Adjusted highcut to {highcut:.1f} Hz for stability")
        
        # Normalize frequencies
        low_normalized = lowcut / nyquist
        high_normalized = highcut / nyquist
        
        st.info(f"Filter range: {lowcut}-{highcut} Hz")
        st.info(f"Normalized range: {low_normalized:.3f}-{high_normalized:.3f}")
        
        # Design Butterworth filter
        try:
            b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        except ValueError as e:
            st.error(f"Filter design error: {e}")
            st.info("Using safer filter parameters...")
            # Use more conservative parameters
            low_normalized = 0.05  # ~10 Hz for 5ms sample rate
            high_normalized = 0.45  # ~90 Hz for 5ms sample rate
            b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        
        enhanced_data = np.zeros_like(seismic_data)
        n_inlines, n_xlines, n_samples = seismic_data.shape
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_traces = n_inlines * n_xlines
        processed_traces = 0
        
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = seismic_data[i, j, :].copy()
                
                # Remove any NaN or Inf values
                trace = np.nan_to_num(trace)
                
                # Use filtfilt for zero-phase filtering
                try:
                    enhanced_trace = signal.filtfilt(b, a, trace)
                    enhanced_data[i, j, :] = enhanced_trace
                except Exception as e:
                    st.warning(f"Error filtering trace {j}: {e}")
                    enhanced_data[i, j, :] = trace  # Use original if filtering fails
                
                # Update progress
                processed_traces += 1
                if processed_traces % 100 == 0:
                    progress = processed_traces / total_traces
                    progress_bar.progress(progress)
                    status_text.text(f"Filtering trace {processed_traces}/{total_traces}")
        
        progress_bar.progress(1.0)
        status_text.text("Bandpass filtering completed!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    
        return enhanced_data

    def plot_spectral_comparison(self, original_trace, enhanced_trace, trace_idx=0):
        """Plot comparison between original and enhanced spectra"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain comparison
        time_axis = np.arange(len(original_trace)) * self.sample_rate
        axes[0, 0].plot(time_axis, original_trace, 'b-', alpha=0.8, label='Original', linewidth=1.5)
        axes[0, 0].plot(time_axis, enhanced_trace, 'r-', alpha=0.7, label='Enhanced', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'Trace Comparison (Trace #{trace_idx})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Frequency spectrum comparison
        orig_fft = np.abs(fft(original_trace))
        enh_fft = np.abs(fft(enhanced_trace))
        freqs = fftfreq(len(original_trace), d=self.sample_rate/1000.0)
        
        positive_freqs = freqs > 0
        max_freq = 200  # Show up to 200 Hz
        
        valid_freqs = freqs[positive_freqs] <= max_freq
        plot_freqs = freqs[positive_freqs][valid_freqs]
        
        axes[0, 1].plot(plot_freqs, orig_fft[positive_freqs][valid_freqs], 'b-', 
                       alpha=0.8, label='Original', linewidth=2)
        axes[0, 1].plot(plot_freqs, enh_fft[positive_freqs][valid_freqs], 'r-', 
                       alpha=0.7, label='Enhanced', linewidth=2)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title('Frequency Spectrum Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectral ratio
        spectral_ratio = enh_fft[positive_freqs] / (orig_fft[positive_freqs] + 1e-10)
        axes[1, 0].plot(plot_freqs, spectral_ratio[valid_freqs], 'g-', linewidth=2)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Spectral Ratio')
        axes[1, 0].set_title('Enhancement Factor vs Frequency')
        axes[1, 0].set_ylim([0, 3])  # Limit y-axis for better visualization
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of amplitudes
        axes[1, 1].hist(original_trace, bins=100, alpha=0.7, label='Original', color='blue', density=True)
        axes[1, 1].hist(enhanced_trace, bins=100, alpha=0.7, label='Enhanced', color='red', density=True)
        axes[1, 1].set_xlabel('Amplitude')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Amplitude Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_seismic_section(self, trace_idx=100):
        """Plot a section of original vs enhanced seismic"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Calculate reasonable display limits
        start_trace = max(0, trace_idx - 50)
        end_trace = min(self.original_data.shape[1], trace_idx + 50)
        
        # Original data
        vmax = np.percentile(np.abs(self.original_data[0, start_trace:end_trace, :]), 95)
        im1 = ax1.imshow(self.original_data[0, start_trace:end_trace, :].T, 
                        aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        ax1.set_title('Original Seismic')
        ax1.set_xlabel('Trace Number')
        ax1.set_ylabel('Time Sample')
        plt.colorbar(im1, ax=ax1)
        
        # Enhanced data
        im2 = ax2.imshow(self.enhanced_data[0, start_trace:end_trace, :].T, 
                        aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        ax2.set_title('Enhanced Seismic')
        ax2.set_xlabel('Trace Number')
        ax2.set_ylabel('Time Sample')
        plt.colorbar(im2, ax=ax2)
        
        # Difference
        diff = self.enhanced_data - self.original_data
        vmax_diff = np.percentile(np.abs(diff[0, start_trace:end_trace, :]), 95)
        im3 = ax3.imshow(diff[0, start_trace:end_trace, :].T, 
                        aspect='auto', cmap='RdBu', vmin=-vmax_diff, vmax=vmax_diff)
        ax3.set_title('Difference (Enhanced - Original)')
        ax3.set_xlabel('Trace Number')
        ax3.set_ylabel('Time Sample')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        return fig

    def plot_frequency_analysis(self, num_traces=10):
        """Plot frequency content analysis for multiple traces"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Select random traces for analysis
        trace_indices = np.random.choice(self.original_data.shape[1], min(num_traces, self.original_data.shape[1]), replace=False)
        
        # Calculate average frequency spectra
        avg_orig_spectrum = np.zeros(self.original_data.shape[2] // 2)
        avg_enh_spectrum = np.zeros(self.enhanced_data.shape[2] // 2)
        
        freqs = fftfreq(self.original_data.shape[2], d=self.sample_rate/1000.0)
        positive_freqs = freqs > 0
        
        for idx in trace_indices:
            orig_trace = self.original_data[0, idx, :]
            enh_trace = self.enhanced_data[0, idx, :]
            
            orig_fft = np.abs(fft(orig_trace))[positive_freqs]
            enh_fft = np.abs(fft(enh_trace))[positive_freqs]
            
            avg_orig_spectrum += orig_fft
            avg_enh_spectrum += enh_fft
        
        avg_orig_spectrum /= len(trace_indices)
        avg_enh_spectrum /= len(trace_indices)
        
        plot_freqs = freqs[positive_freqs]
        max_freq = 150  # Show up to 150 Hz
        
        valid_freqs = plot_freqs <= max_freq
        
        # Plot average spectra
        ax1.semilogy(plot_freqs[valid_freqs], avg_orig_spectrum[valid_freqs], 'b-', 
                    alpha=0.8, label='Original', linewidth=2)
        ax1.semilogy(plot_freqs[valid_freqs], avg_enh_spectrum[valid_freqs], 'r-', 
                    alpha=0.8, label='Enhanced', linewidth=2)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude (log)')
        ax1.set_title('Average Frequency Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot spectral ratio
        spectral_ratio = avg_enh_spectrum / (avg_orig_spectrum + 1e-10)
        ax2.plot(plot_freqs[valid_freqs], spectral_ratio[valid_freqs], 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Spectral Ratio')
        ax2.set_title('Average Enhancement Factor')
        ax2.set_ylim([0.5, 2.5])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def enhance_bandwidth(self, file_path, method='spectral_blueing', 
                         target_freq=80, enhancement_factor=1.5, low_freq_boost=1.2,
                         mid_freq_start=30, lowcut=8, highcut=120, filter_order=3):
        """
        Main method to enhance seismic bandwidth
        """
        st.info(f"Loading SEG-Y file...")
        self.original_data = self.read_segy(file_path)
        
        if self.original_data is None:
            raise ValueError("Failed to load SEG-Y file")
        
        st.success(f"Original data shape: {self.original_data.shape}")
        st.info(f"Original data range: {np.min(self.original_data):.3f} to {np.max(self.original_data):.3f}")
        
        start_time = time.time()
        
        # Apply spectral blueing with correct parameters
        st.info("Starting spectral blueing...")
        self.enhanced_data = self.spectral_blueing(
            self.original_data, 
            target_freq=target_freq,
            enhancement_factor=enhancement_factor,
            low_freq_boost=low_freq_boost,
            mid_freq_range=(mid_freq_start, target_freq)
        )
        
        # Apply bandpass filter with correct parameters
        st.info("Applying bandpass filter...")
        self.enhanced_data = self.bandpass_filter(
            self.enhanced_data, 
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order
        )
        
        processing_time = time.time() - start_time
        st.success(f"Processing completed in {processing_time:.2f} seconds")
        st.info(f"Enhanced data range: {np.min(self.enhanced_data):.3f} to {np.max(self.enhanced_data):.3f}")
        
        return self.enhanced_data

def main():
    st.set_page_config(
        page_title="Seismic Bandwidth Enhancer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌊 Seismic Bandwidth Enhancement Tool")
    st.markdown("""
    Enhance the frequency content of your seismic data using spectral blueing techniques.
    Upload a SEG-Y file and adjust the parameters to optimize the bandwidth enhancement.
    """)
    
    # Initialize enhancer
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = SeismicBandwidthEnhancer()
    
    enhancer = st.session_state.enhancer
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your seismic data in SEG-Y format"
    )
    
    st.sidebar.header("⚙️ Processing Parameters")
    
    # Spectral blueing parameters
    st.sidebar.subheader("Spectral Blueing")
    target_freq = st.sidebar.slider(
        "Target Frequency (Hz)",
        min_value=30,
        max_value=120,
        value=80,
        help="Primary frequency for enhancement"
    )
    
    enhancement_factor = st.sidebar.slider(
        "Enhancement Factor",
        min_value=1.0,
        max_value=3.0,
        value=1.8,
        step=0.1,
        help="Boost factor for target frequencies"
    )
    
    low_freq_boost = st.sidebar.slider(
        "Low Frequency Boost",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Boost factor for low frequencies (5-30 Hz)"
    )
    
    mid_freq_start = st.sidebar.slider(
        "Mid Frequency Range Start (Hz)",
        min_value=10,
        max_value=50,
        value=30,
        help="Start of mid-frequency range for moderate enhancement"
    )
    
    # Bandpass filter parameters
    st.sidebar.subheader("Bandpass Filter")
    lowcut = st.sidebar.slider(
        "Low Cut Frequency (Hz)",
        min_value=1,
        max_value=50,
        value=8,
        help="Lower frequency cutoff for bandpass filter"
    )
    
    highcut = st.sidebar.slider(
        "High Cut Frequency (Hz)",
        min_value=60,
        max_value=200,
        value=120,
        help="Higher frequency cutoff for bandpass filter"
    )
    
    filter_order = st.sidebar.slider(
        "Filter Order",
        min_value=2,
        max_value=6,
        value=3,
        help="Order of the Butterworth filter"
    )
    
    # Visualization parameters
    st.sidebar.subheader("Visualization")
    trace_idx = st.sidebar.slider(
        "Reference Trace Index",
        min_value=0,
        max_value=1000,
        value=100,
        help="Trace index for detailed comparison plots"
    )
    
    num_analysis_traces = st.sidebar.slider(
        "Number of Traces for Analysis",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of traces to include in frequency analysis"
    )
    
    # Main processing area
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
        
        try:
            # Process button
            if st.button("🚀 Process Seismic Data", type="primary"):
                with st.spinner("Processing seismic data..."):
                    # Enhance bandwidth with explicit parameters
                    enhanced_data = enhancer.enhance_bandwidth(
                        temp_filename,
                        target_freq=target_freq,
                        enhancement_factor=enhancement_factor,
                        low_freq_boost=low_freq_boost,
                        mid_freq_start=mid_freq_start,
                        lowcut=lowcut,
                        highcut=highcut,
                        filter_order=filter_order
                    )
                
                st.success("✅ Processing completed!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Trace Comparison", 
                    "🖼️ Seismic Section", 
                    "📊 Frequency Analysis",
                    "📋 Data Statistics"
                ])
                
                with tab1:
                    st.header("Trace Comparison")
                    # Plot spectral comparison
                    safe_trace_idx = min(trace_idx, enhancer.original_data.shape[1] - 1)
                    original_trace = enhancer.original_data[0, safe_trace_idx, :]
                    enhanced_trace = enhanced_data[0, safe_trace_idx, :]
                    
                    fig_comparison = enhancer.plot_spectral_comparison(
                        original_trace, enhanced_trace, safe_trace_idx
                    )
                    st.pyplot(fig_comparison)
                
                with tab2:
                    st.header("Seismic Section View")
                    fig_section = enhancer.plot_seismic_section(trace_idx)
                    if fig_section:
                        st.pyplot(fig_section)
                
                with tab3:
                    st.header("Frequency Analysis")
                    fig_freq = enhancer.plot_frequency_analysis(num_analysis_traces)
                    if fig_freq:
                        st.pyplot(fig_freq)
                
                with tab4:
                    st.header("Data Statistics")
                    
                    # Display data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Min", f"{np.min(enhancer.original_data):.3f}")
                        st.metric("Enhanced Min", f"{np.min(enhanced_data):.3f}")
                    with col2:
                        st.metric("Original Max", f"{np.max(enhancer.original_data):.3f}")
                        st.metric("Enhanced Max", f"{np.max(enhanced_data):.3f}")
                    with col3:
                        st.metric("Original Std", f"{np.std(enhancer.original_data):.3f}")
                        st.metric("Enhanced Std", f"{np.std(enhanced_data):.3f}")
                    
                    # Additional statistics
                    st.subheader("Additional Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data Shape:**", enhancer.original_data.shape)
                        st.write("**Sample Rate:**", f"{enhancer.sample_rate} ms")
                    with col2:
                        st.write("**Enhanced Data Shape:**", enhanced_data.shape)
                        st.write("**Processing Parameters:**")
                        st.write(f"- Target Frequency: {target_freq} Hz")
                        st.write(f"- Enhancement Factor: {enhancement_factor}x")
                        st.write(f"- Filter Range: {lowcut}-{highcut} Hz")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a SEG-Y file using the sidebar to begin processing.")
        
        # Example of what the tool does
        st.header("About This Tool")
        st.markdown("""
        This tool enhances seismic data bandwidth using **spectral blueing** techniques:
        
        - **Spectral Blueing**: Frequency-dependent enhancement that boosts mid-to-high frequencies
        - **Bandpass Filtering**: Removes very low and very high frequency noise
        - **Zero-phase Processing**: Maintains timing relationships in the data
        
        ### Typical Workflow:
        1. Upload a SEG-Y file using the sidebar
        2. Adjust enhancement parameters based on your data characteristics
        3. Click 'Process Seismic Data' to run the enhancement
        4. Review the results in the various visualization tabs
        
        ### Parameter Guidelines:
        - **Target Frequency**: Typically 60-80 Hz for most seismic data
        - **Enhancement Factor**: 1.5-2.0 for moderate enhancement
        - **Filter Range**: 8-120 Hz works well for most land/marine data
        """)

if __name__ == "__main__":
    main()
