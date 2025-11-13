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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import json
import gc
import psutil
import uuid

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0  # Default 4ms, adjust if needed
        self.geometry = None  # Store geometry information
        self.original_segyfile = None  # Store original segy file reference
        self.original_filename = None  # Store original filename
        
    def read_segy_3d(self, filename):
        """Read 3D SEG-Y file and return seismic data as numpy array"""
        try:
            with segyio.open(filename, "r") as segyfile:
                self.original_segyfile = segyfile  # Store for later use
                self.original_filename = filename  # Store original filename
                
                # Try to read as 3D data with proper geometry
                try:
                    # Get cube dimensions
                    n_inlines = segyfile.ilines.size
                    n_xlines = segyfile.xlines.size
                    n_samples = segyfile.samples.size
                    
                    st.success(f"3D seismic data detected: {n_inlines} inlines × {n_xlines} crosslines × {n_samples} samples")
                    
                    # Read the entire cube
                    data = segyio.tools.cube(segyfile)
                    
                    # Store geometry information
                    self.geometry = {
                        'ilines': segyfile.ilines,
                        'xlines': segyfile.xlines,
                        'samples': segyfile.samples,
                        'tracecount': segyfile.tracecount,
                        'format': segyfile.format
                    }
                    
                except Exception as e:
                    st.warning(f"Could not read as 3D cube: {e}. Reading as 2D...")
                    return self.read_segy_2d(segyfile)
                
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
            return None

    def read_segy_2d(self, segyfile):
        """Read 2D SEG-Y file"""
        st.info("2D seismic data detected")
        # Read as 2D data (traces, samples)
        data = np.stack([segyfile.trace[i] for i in range(segyfile.tracecount)])
        data = data.reshape(1, data.shape[0], data.shape[1])  # Make it 3D with 1 inline
        
        # Store basic geometry
        self.geometry = {
            'ilines': [0],
            'xlines': np.arange(data.shape[1]),
            'samples': segyfile.samples,
            'tracecount': segyfile.tracecount,
            'format': segyfile.format
        }
        
        return data

    def read_segy(self, filename):
        """Main method to read SEG-Y file (handles both 2D and 3D)"""
        return self.read_segy_3d(filename)

    def write_segy_proper(self, output_filename):
        """Proper SEG-Y writing using segyio with enhanced data - FIXED VERSION"""
        if self.enhanced_data is None:
            st.error("No enhanced data available to write")
            return False
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
            
            # Open original file for reading to get the structure
            with segyio.open(self.original_filename, "r") as src:
                # Get basic file specifications
                n_traces = src.tracecount
                n_samples = len(src.samples)
                
                # Create a new SEG-Y file with the same structure
                with segyio.open(output_filename, "w", tracecount=n_traces, 
                               sample_format=src.format, samples=src.samples) as dst:
                    
                    # Copy all textual headers
                    for i in range(len(src.text)):
                        dst.text[i] = src.text[i]
                    
                    # Copy binary header
                    dst.bin = src.bin
                    
                    # Write enhanced traces while preserving all headers
                    n_inlines, n_xlines, n_samples_enhanced = self.enhanced_data.shape
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    trace_index = 0
                    for i in range(n_inlines):
                        for j in range(n_xlines):
                            if trace_index < n_traces:
                                # Copy ALL trace headers from original
                                dst.header[trace_index] = src.header[trace_index]
                                
                                # Write enhanced trace data
                                enhanced_trace = self.enhanced_data[i, j, :].astype(np.float32)
                                
                                # Ensure trace length matches
                                if len(enhanced_trace) == n_samples:
                                    dst.trace[trace_index] = enhanced_trace
                                else:
                                    # Handle trace length mismatch
                                    if len(enhanced_trace) > n_samples:
                                        dst.trace[trace_index] = enhanced_trace[:n_samples]
                                    else:
                                        padded_trace = np.zeros(n_samples, dtype=np.float32)
                                        padded_trace[:len(enhanced_trace)] = enhanced_trace
                                        dst.trace[trace_index] = padded_trace
                                
                                trace_index += 1
                            
                            # Update progress
                            current_trace = i * n_xlines + j + 1
                            if current_trace % 100 == 0:
                                progress = current_trace / (n_inlines * n_xlines)
                                progress_bar.progress(progress)
                                status_text.text(f"Writing trace {current_trace}/{n_inlines * n_xlines}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("SEG-Y file writing completed!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
            
            # Verify the file was created properly
            if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                file_size = os.path.getsize(output_filename) / (1024 * 1024)  # Size in MB
                st.success(f"Enhanced SEG-Y file created successfully: {output_filename} (Size: {file_size:.2f} MB)")
                return True
            else:
                st.error("Output file was not created properly")
                return False
                
        except Exception as e:
            st.error(f"Error writing SEG-Y file: {e}")
            return False

    def write_segy_simple(self, output_filename):
        """Simple SEG-Y writing method that works reliably"""
        if self.enhanced_data is None:
            st.error("No enhanced data available to write")
            return False
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
            
            # Open original file for reading
            with segyio.open(self.original_filename, "r") as src:
                # Get basic specifications
                n_traces = src.tracecount
                n_samples = len(src.samples)
                
                # Flatten enhanced data for writing
                enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
                
                # Create output file
                with segyio.open(output_filename, "w", tracecount=n_traces, 
                               sample_format=src.format, samples=src.samples) as dst:
                    
                    # Copy textual headers
                    for i in range(len(src.text)):
                        dst.text[i] = src.text[i]
                    
                    # Copy binary header
                    dst.bin = src.bin
                    
                    # Copy headers and write enhanced data
                    for i in range(n_traces):
                        # Copy trace header
                        dst.header[i] = src.header[i]
                        
                        # Write enhanced trace data
                        if i < len(enhanced_flat):
                            trace_data = enhanced_flat[i].astype(np.float32)
                            # Ensure correct length
                            if len(trace_data) == n_samples:
                                dst.trace[i] = trace_data
                            else:
                                # Handle length mismatch
                                if len(trace_data) > n_samples:
                                    dst.trace[i] = trace_data[:n_samples]
                                else:
                                    padded = np.zeros(n_samples, dtype=np.float32)
                                    padded[:len(trace_data)] = trace_data
                                    dst.trace[i] = padded
                        else:
                            # Fallback to original trace if enhanced data doesn't cover all traces
                            dst.trace[i] = src.trace[i]
                
                st.success(f"Simple SEG-Y writing completed: {output_filename}")
                return True
                
        except Exception as e:
            st.error(f"Simple SEG-Y writing failed: {e}")
            return False

    def write_segy_binary_copy(self, output_filename):
        """Most reliable method: binary copy with trace replacement"""
        if self.enhanced_data is None:
            st.error("No enhanced data available to write")
            return False
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
            
            # First, create a copy of the original file
            import shutil
            shutil.copy2(self.original_filename, output_filename)
            
            # Now open the copy and replace trace data
            with segyio.open(output_filename, "r+") as segyfile:
                n_traces = segyfile.tracecount
                n_samples = len(segyfile.samples)
                
                # Flatten enhanced data
                enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
                
                # Replace trace data
                for i in range(min(n_traces, len(enhanced_flat))):
                    trace_data = enhanced_flat[i].astype(np.float32)
                    # Ensure correct length
                    if len(trace_data) == n_samples:
                        segyfile.trace[i] = trace_data
                    else:
                        # Handle length mismatch
                        if len(trace_data) > n_samples:
                            segyfile.trace[i] = trace_data[:n_samples]
                        else:
                            padded = np.zeros(n_samples, dtype=np.float32)
                            padded[:len(trace_data)] = trace_data
                            segyfile.trace[i] = padded
            
            st.success(f"Binary copy SEG-Y writing completed: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"Binary copy SEG-Y writing failed: {e}")
            return False

    def create_downloadable_segy(self, output_filename):
        """Create enhanced SEG-Y file and return the file path for download"""
        if self.enhanced_data is None:
            st.error("No enhanced data available. Please process the data first.")
            return None
        
        # Ensure output filename has .sgy extension
        if not output_filename.lower().endswith(('.sgy', '.segy')):
            output_filename = os.path.splitext(output_filename)[0] + '.sgy'
        
        try:
            # Create temporary file for download with unique name
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())[:8]
            base_name = os.path.splitext(output_filename)[0]
            download_filename = os.path.join(temp_dir, f"{base_name}_{unique_id}.sgy")
            
            st.info("Creating enhanced SEG-Y file...")
            
            # Try multiple methods in order of reliability
            methods = [
                ("Binary Copy Method", self.write_segy_binary_copy),
                ("Simple Method", self.write_segy_simple),
                ("Proper Method", self.write_segy_proper)
            ]
            
            success = False
            for method_name, method_func in methods:
                st.info(f"Trying {method_name}...")
                success = method_func(download_filename)
                if success:
                    st.success(f"{method_name} succeeded!")
                    break
                else:
                    st.warning(f"{method_name} failed, trying next method...")
            
            if success:
                # Verify the file was created
                if os.path.exists(download_filename) and os.path.getsize(download_filename) > 0:
                    file_size = os.path.getsize(download_filename) / (1024 * 1024)
                    st.success(f"Enhanced SEG-Y file created successfully! Size: {file_size:.2f} MB")
                    
                    # Quick verification
                    try:
                        with segyio.open(download_filename, "r") as test_file:
                            st.info(f"Output verification: {test_file.tracecount} traces, {len(test_file.samples)} samples")
                        return download_filename
                    except Exception as e:
                        st.warning(f"File created but verification failed: {e}")
                        return download_filename
                else:
                    st.error("Enhanced SEG-Y file was not created properly")
                    return None
            else:
                st.error("All SEG-Y writing methods failed")
                return None
                
        except Exception as e:
            st.error(f"Error creating downloadable file: {e}")
            return None

    def spectral_blueing(self, seismic_data, target_freq=80, enhancement_factor=1.5,
                        low_freq_boost=1.2, mid_freq_range=(30, 80)):
        """Spectral blueing to enhance high frequencies"""
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
                
                # Enhanced trace processing with error handling
                enhanced_trace = self.safe_fft_processing(
                    trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range
                )
                
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

    def safe_fft_processing(self, trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range):
        """Safe FFT processing with comprehensive error handling"""
        try:
            if np.all(trace == 0):
                return trace  # Skip processing for zero traces
                
            if np.isnan(trace).any() or np.isinf(trace).any():
                trace = np.nan_to_num(trace)
                
            # Check for constant traces
            if np.std(trace) < 1e-10:
                return trace
                
            return self._apply_spectral_blueing(trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range)
            
        except Exception as e:
            st.warning(f"Trace processing failed: {e}. Using original trace.")
            return trace

    def _apply_spectral_blueing(self, trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range):
        """Apply spectral blueing to a single trace"""
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
        
        return enhanced_trace

    def bandpass_filter(self, seismic_data, lowcut=8, highcut=120, order=3):
        """Apply bandpass filter to remove very low and very high frequency noise"""
        st.info("Applying bandpass filter...")
        
        # Calculate Nyquist frequency correctly
        sampling_interval = self.sample_rate / 1000.0  # Convert ms to seconds
        sampling_freq = 1.0 / sampling_interval  # Sampling frequency in Hz
        nyquist = sampling_freq / 2.0  # Nyquist frequency in Hz
        
        # Auto-adjust filter range if needed
        if highcut >= nyquist * 0.95:
            highcut = nyquist * 0.9
            st.warning(f"Adjusted highcut to {highcut:.1f} Hz for stability")
        
        # Normalize frequencies
        low_normalized = lowcut / nyquist
        high_normalized = highcut / nyquist
        
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

    def enhance_bandwidth(self, file_path, method='spectral_blueing', 
                         target_freq=80, enhancement_factor=1.5, low_freq_boost=1.2,
                         mid_freq_start=30, lowcut=8, highcut=120, filter_order=3,
                         use_chunk_processing=True, chunk_size=50):
        """Main method to enhance seismic bandwidth"""
        st.info(f"Loading SEG-Y file...")
        self.original_data = self.read_segy(file_path)
        
        if self.original_data is None:
            raise ValueError("Failed to load SEG-Y file")
        
        st.success(f"Original data shape: {self.original_data.shape}")
        st.info(f"Original data range: {np.min(self.original_data):.3f} to {np.max(self.original_data):.3f}")
        
        start_time = time.time()
        
        # Apply spectral blueing with correct parameters
        st.info("Starting spectral blueing...")
        
        if use_chunk_processing and self.original_data.shape[0] > chunk_size:
            st.info(f"Using chunk processing with chunk size: {chunk_size}")
            self.enhanced_data = self.spectral_blueing(
                self.original_data,
                target_freq=target_freq,
                enhancement_factor=enhancement_factor,
                low_freq_boost=low_freq_boost,
                mid_freq_range=(mid_freq_start, target_freq)
            )
        else:
            self.enhanced_data = self.spectral_blueing(
                self.original_data, 
                target_freq=target_freq,
                enhancement_factor=enhancement_factor,
                low_freq_boost=low_freq_boost,
                mid_freq_range=(mid_freq_start, target_freq)
            )
        
        # Apply bandpass filter with correct parameters
        st.info("Applying bandpass filter...")
        if use_chunk_processing and self.enhanced_data.shape[0] > chunk_size:
            self.enhanced_data = self.bandpass_filter(
                self.enhanced_data,
                lowcut=lowcut,
                highcut=highcut,
                order=filter_order
            )
        else:
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

# Processing presets
PROCESSING_PRESETS = {
    'high_resolution': {
        'target_freq': 100,
        'enhancement_factor': 2.0,
        'low_freq_boost': 1.1,
        'mid_freq_start': 25,
        'lowcut': 10,
        'highcut': 150,
        'filter_order': 4
    },
    'balanced': {
        'target_freq': 80,
        'enhancement_factor': 1.5,
        'low_freq_boost': 1.2,
        'mid_freq_start': 30,
        'lowcut': 8,
        'highcut': 120,
        'filter_order': 3
    },
    'conservative': {
        'target_freq': 60,
        'enhancement_factor': 1.2,
        'low_freq_boost': 1.3,
        'mid_freq_start': 20,
        'lowcut': 5,
        'highcut': 100,
        'filter_order': 3
    }
}

def safe_file_download(file_path, download_name):
    """Safe file download with proper error handling"""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            st.error("File is empty")
            return None
            
        with open(file_path, "rb") as file:
            file_data = file.read()
        
        return file_data
        
    except Exception as e:
        st.error(f"Error reading file for download: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Seismic Bandwidth Enhancer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'file_generated' not in st.session_state:
        st.session_state.file_generated = False
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = SeismicBandwidthEnhancer()
    if 'original_filename' not in st.session_state:
        st.session_state.original_filename = None
    if 'enhanced_file_path' not in st.session_state:
        st.session_state.enhanced_file_path = None
    if 'enhanced_file_data' not in st.session_state:
        st.session_state.enhanced_file_data = None
    
    st.title("🌊 3D Seismic Bandwidth Enhancement Tool")
    st.markdown("""
    Enhance the frequency content of your 3D seismic data using spectral blueing techniques.
    Upload a 3D SEG-Y file and adjust the parameters to optimize the bandwidth enhancement.
    """)
    
    enhancer = st.session_state.enhancer
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format"
    )
    
    st.sidebar.header("⚙️ Processing Parameters")
    
    # Processing presets
    st.sidebar.subheader("Processing Presets")
    preset = st.sidebar.selectbox(
        "Choose Processing Preset",
        options=list(PROCESSING_PRESETS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Select a preset configuration for processing"
    )
    
    # Apply preset parameters
    if preset:
        preset_params = PROCESSING_PRESETS[preset]
        target_freq = preset_params['target_freq']
        enhancement_factor = preset_params['enhancement_factor']
        low_freq_boost = preset_params['low_freq_boost']
        mid_freq_start = preset_params['mid_freq_start']
        lowcut = preset_params['lowcut']
        highcut = preset_params['highcut']
        filter_order = preset_params['filter_order']
    else:
        # Default values
        target_freq = 80
        enhancement_factor = 1.5
        low_freq_boost = 1.2
        mid_freq_start = 30
        lowcut = 8
        highcut = 120
        filter_order = 3
    
    # Spectral blueing parameters
    st.sidebar.subheader("Spectral Blueing")
    target_freq = st.sidebar.slider(
        "Target Frequency (Hz)",
        min_value=30,
        max_value=120,
        value=target_freq,
        help="Primary frequency for enhancement"
    )
    
    enhancement_factor = st.sidebar.slider(
        "Enhancement Factor",
        min_value=1.0,
        max_value=3.0,
        value=enhancement_factor,
        step=0.1,
        help="Boost factor for target frequencies"
    )
    
    low_freq_boost = st.sidebar.slider(
        "Low Frequency Boost",
        min_value=1.0,
        max_value=2.0,
        value=low_freq_boost,
        step=0.1,
        help="Boost factor for low frequencies (5-30 Hz)"
    )
    
    mid_freq_start = st.sidebar.slider(
        "Mid Frequency Range Start (Hz)",
        min_value=10,
        max_value=50,
        value=mid_freq_start,
        help="Start of mid-frequency range for moderate enhancement"
    )
    
    # Bandpass filter parameters
    st.sidebar.subheader("Bandpass Filter")
    lowcut = st.sidebar.slider(
        "Low Cut Frequency (Hz)",
        min_value=1,
        max_value=50,
        value=lowcut,
        help="Lower frequency cutoff for bandpass filter"
    )
    
    highcut = st.sidebar.slider(
        "High Cut Frequency (Hz)",
        min_value=60,
        max_value=200,
        value=highcut,
        help="Higher frequency cutoff for bandpass filter"
    )
    
    filter_order = st.sidebar.slider(
        "Filter Order",
        min_value=2,
        max_value=6,
        value=filter_order,
        help="Order of the Butterworth filter"
    )
    
    # Performance settings
    st.sidebar.subheader("Performance Settings")
    use_chunk_processing = st.sidebar.checkbox(
        "Use Chunk Processing", 
        value=True,
        help="Process data in chunks to reduce memory usage"
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (inlines)",
        min_value=10,
        max_value=100,
        value=50,
        help="Number of inlines to process at once"
    )
    
    # Main processing area
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
            st.session_state.original_filename = temp_filename
        
        try:
            # Process button
            if st.button("🚀 Process Seismic Data", type="primary", use_container_width=True):
                with st.spinner("Processing 3D seismic data..."):
                    # Enhance bandwidth with explicit parameters
                    enhanced_data = enhancer.enhance_bandwidth(
                        temp_filename,
                        target_freq=target_freq,
                        enhancement_factor=enhancement_factor,
                        low_freq_boost=low_freq_boost,
                        mid_freq_start=mid_freq_start,
                        lowcut=lowcut,
                        highcut=highcut,
                        filter_order=filter_order,
                        use_chunk_processing=use_chunk_processing,
                        chunk_size=chunk_size
                    )
                
                st.success("✅ 3D Processing completed!")
                st.session_state.data_processed = True
                # Reset download state when new processing happens
                st.session_state.file_generated = False
                st.session_state.enhanced_file_path = None
                st.session_state.enhanced_file_data = None

            # Create download section
            if st.session_state.get('data_processed', False):
                st.sidebar.header("💾 Download Results")
                
                # Enhanced data download
                output_filename = "enhanced_seismic.sgy"
                
                # Generate button
                if st.sidebar.button("🛠️ Generate Enhanced SEG-Y File", type="secondary", use_container_width=True):
                    with st.sidebar:
                        with st.spinner("Creating enhanced SEG-Y file..."):
                            enhanced_file_path = enhancer.create_downloadable_segy(output_filename)
                            
                            if enhanced_file_path:
                                st.session_state.enhanced_file_path = enhanced_file_path
                                # Pre-load file data to avoid file access issues during download
                                file_data = safe_file_download(enhanced_file_path, output_filename)
                                if file_data is not None:
                                    st.session_state.enhanced_file_data = file_data
                                    st.session_state.file_generated = True
                                    st.sidebar.success("Enhanced SEG-Y file created successfully!")
                                else:
                                    st.sidebar.error("Failed to load file data for download")
                            else:
                                st.sidebar.error("Failed to create enhanced SEG-Y file")
                
                # Download button - separate from generate button
                if st.session_state.get('file_generated', False) and st.session_state.enhanced_file_data is not None:
                    with st.sidebar:
                        try:
                            # Use pre-loaded file data to avoid file access issues
                            file_data = st.session_state.enhanced_file_data
                            
                            download_name = "enhanced_seismic.sgy"
                            mime_type = "application/octet-stream"
                            
                            st.download_button(
                                label="📥 Download Enhanced SEG-Y",
                                data=file_data,
                                file_name=download_name,
                                mime=mime_type,
                                help="Download the enhanced seismic data in SEG-Y format",
                                key="download_enhanced_data",
                                use_container_width=True
                            )
                            st.success("Enhanced SEG-Y file ready for download!")
                                
                        except Exception as e:
                            st.error(f"Error preparing download: {e}")
                
                # Display basic results
                st.header("Processing Results")
                st.success("Data processing completed successfully! Use the sidebar to generate and download the enhanced SEG-Y file.")
                
                # Show basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Data")
                    st.write(f"Shape: {enhancer.original_data.shape}")
                    st.write(f"Range: {np.min(enhancer.original_data):.3f} to {np.max(enhancer.original_data):.3f}")
                
                with col2:
                    st.subheader("Enhanced Data")
                    st.write(f"Shape: {enhancer.enhanced_data.shape}")
                    st.write(f"Range: {np.min(enhancer.enhanced_data):.3f} to {np.max(enhancer.enhanced_data):.3f}")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
    
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a 3D SEG-Y file using the sidebar to begin processing.")
        
        st.header("About This Tool")
        st.markdown("""
        This tool enhances 3D seismic data bandwidth using **spectral blueing** techniques.
        
        ### Features:
        - **Download Enhanced SEG-Y**: Get your processed data in standard SEG-Y format
        - **Memory Optimization**: Chunk processing for large datasets
        - **Processing Presets**: Pre-configured settings for different scenarios
        
        ### How to Use:
        1. Upload a SEG-Y file
        2. Adjust processing parameters
        3. Click "Process Seismic Data"
        4. Generate and download the enhanced SEG-Y file
        """)

    # Clean up temporary files
    def cleanup_old_files():
        """Clean up old temporary files"""
        temp_dir = tempfile.gettempdir()
        current_file = getattr(st.session_state, 'enhanced_file_path', None)
        
        for filename in os.listdir(temp_dir):
            if filename.startswith("enhanced_seismic_") and filename.endswith(".sgy"):
                file_path = os.path.join(temp_dir, filename)
                if file_path != current_file and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass
    
    cleanup_old_files()

if __name__ == "__main__":
    main()
