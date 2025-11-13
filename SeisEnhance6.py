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
import struct

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0  # Default 4ms, adjust if needed
        self.geometry = None
        self.original_segyfile = None
        self.original_filename = None
        
    def read_segy_3d(self, filename):
        """Read 3D SEG-Y file and return seismic data as numpy array"""
        try:
            with segyio.open(filename, "r") as segyfile:
                self.original_segyfile = segyfile
                self.original_filename = filename
                
                try:
                    n_inlines = segyfile.ilines.size
                    n_xlines = segyfile.xlines.size
                    n_samples = segyfile.samples.size
                    
                    st.success(f"3D seismic data detected: {n_inlines} inlines × {n_xlines} crosslines × {n_samples} samples")
                    
                    data = segyio.tools.cube(segyfile)
                    
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
                
                try:
                    self.sample_rate = segyio.tools.dt(segyfile) / 1000.0
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
        data = np.stack([segyfile.trace[i] for i in range(segyfile.tracecount)])
        data = data.reshape(1, data.shape[0], data.shape[1])
        
        self.geometry = {
            'ilines': [0],
            'xlines': np.arange(data.shape[1]),
            'samples': segyfile.samples,
            'tracecount': segyfile.tracecount,
            'format': segyfile.format
        }
        
        return data

    def read_segy(self, filename):
        """Main method to read SEG-Y file"""
        return self.read_segy_3d(filename)

    def write_segy_robust(self, output_filename):
        """Most robust SEG-Y writing method using direct binary operations"""
        if self.enhanced_data is None:
            st.error("No enhanced data available to write")
            return False
            
        try:
            # Read original file structure
            with segyio.open(self.original_filename, "r") as src:
                n_traces = src.tracecount
                n_samples = len(src.samples)
                format_code = src.format
                
                # Get all textual headers
                textual_headers = []
                for i in range(len(src.text)):
                    textual_headers.append(src.text[i])
                
                # Get binary header
                binary_header = src.bin
                
                # Get all trace headers
                trace_headers = []
                for i in range(n_traces):
                    trace_headers.append(dict(src.header[i]))
            
            # Flatten enhanced data
            enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
            
            # Create new SEG-Y file using direct binary writing
            with open(output_filename, 'wb') as f:
                # Write textual headers (3200 bytes each)
                for header in textual_headers:
                    f.write(header.encode('ascii')[:3200].ljust(3200, b' '))
                
                # Write binary header (400 bytes)
                self._write_binary_header(f, binary_header, format_code)
                
                # Write traces
                for i in range(n_traces):
                    # Write trace header (240 bytes)
                    self._write_trace_header(f, trace_headers[i])
                    
                    # Write trace data
                    if i < len(enhanced_flat):
                        trace_data = enhanced_flat[i].astype(np.float32)
                        if len(trace_data) > n_samples:
                            trace_data = trace_data[:n_samples]
                        elif len(trace_data) < n_samples:
                            trace_data = np.pad(trace_data, (0, n_samples - len(trace_data)), 
                                              mode='constant')
                        f.write(trace_data.tobytes())
                    else:
                        # Fallback: write zeros if no enhanced data
                        f.write(np.zeros(n_samples, dtype=np.float32).tobytes())
            
            st.success(f"Robust SEG-Y writing completed: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"Robust SEG-Y writing failed: {e}")
            return False

    def _write_binary_header(self, file_obj, binary_header, format_code):
        """Write binary header in proper SEG-Y format"""
        # Start with 400 bytes of zeros
        header_data = bytearray(400)
        
        # Set basic binary header values
        # Job identification number (bytes 0-3)
        job_id = binary_header.get(segyio.BinField.JobID, 0)
        struct.pack_into('>i', header_data, 0, job_id)
        
        # Line number (bytes 4-7)
        line_number = binary_header.get(segyio.BinField.LineNumber, 0)
        struct.pack_into('>i', header_data, 4, line_number)
        
        # Reel number (bytes 8-11)
        reel_number = binary_header.get(segyio.BinField.ReelNumber, 0)
        struct.pack_into('>i', header_data, 8, reel_number)
        
        # Number of data traces per ensemble (bytes 12-13)
        traces_ensemble = binary_header.get(segyio.BinField.Traces, 0)
        struct.pack_into('>h', header_data, 12, traces_ensemble)
        
        # Number of auxiliary traces per ensemble (bytes 14-15)
        aux_traces = binary_header.get(segyio.BinField.AuxTraces, 0)
        struct.pack_into('>h', header_data, 14, aux_traces)
        
        # Sample interval in microseconds (bytes 16-17)
        sample_interval = int(self.sample_rate * 1000)  # Convert ms to microseconds
        struct.pack_into('>h', header_data, 16, sample_interval)
        
        # Number of samples per data trace (bytes 20-21)
        samples_per_trace = binary_header.get(segyio.BinField.Samples, 0)
        struct.pack_into('>h', header_data, 20, samples_per_trace)
        
        # Data sample format code (bytes 24-25)
        # 1 = 4-byte IBM floating point, 5 = 4-byte IEEE floating point
        struct.pack_into('>h', header_data, 24, 5)  # Use IEEE float
        
        # Ensemble fold (bytes 28-29)
        ensemble_fold = binary_header.get(segyio.BinField.EnsembleFold, 0)
        struct.pack_into('>h', header_data, 28, ensemble_fold)
        
        # Write the binary header
        file_obj.write(header_data)

    def _write_trace_header(self, file_obj, trace_header):
        """Write trace header in proper SEG-Y format"""
        # Start with 240 bytes of zeros
        header_data = bytearray(240)
        
        # Set important trace header values
        # Trace sequence number (bytes 0-3)
        trace_seq = trace_header.get(segyio.TraceField.TRACE_SEQUENCE_FILE, 0)
        struct.pack_into('>i', header_data, 0, trace_seq)
        
        # Field record number (bytes 8-11)
        field_record = trace_header.get(segyio.TraceField.FieldRecord, 0)
        struct.pack_into('>i', header_data, 8, field_record)
        
        # Trace number (bytes 12-15)
        trace_number = trace_header.get(segyio.TraceField.TRACE_NUMBER, 0)
        struct.pack_into('>i', header_data, 12, trace_number)
        
        # Energy source point number (bytes 16-19)
        source_point = trace_header.get(segyio.TraceField.SourcePoint, 0)
        struct.pack_into('>i', header_data, 16, source_point)
        
        # CDP number (bytes 20-23)
        cdp_number = trace_header.get(segyio.TraceField.CDP, 0)
        struct.pack_into('>i', header_data, 20, cdp_number)
        
        # CDP trace number (bytes 24-27)
        cdp_trace = trace_header.get(segyio.TraceField.CDP_TRACE, 0)
        struct.pack_into('>i', header_data, 24, cdp_trace)
        
        # Trace identification code (bytes 28-29)
        trace_id = trace_header.get(segyio.TraceField.TRACE_IDENTIFICATION_CODE, 1)
        struct.pack_into('>h', header_data, 28, trace_id)
        
        # Number of samples in this trace (bytes 114-115)
        num_samples = trace_header.get(segyio.TraceField.TRACE_SAMPLE_COUNT, 0)
        struct.pack_into('>h', header_data, 114, num_samples)
        
        # Sample interval in microseconds (bytes 116-117)
        sample_interval = int(self.sample_rate * 1000)  # Convert ms to microseconds
        struct.pack_into('>h', header_data, 116, sample_interval)
        
        # Write the trace header
        file_obj.write(header_data)

    def write_segy_simple(self, output_filename):
        """Simple SEG-Y writing using basic segyio functionality"""
        try:
            with segyio.open(self.original_filename, "r") as src:
                # Create new file with same structure
                spec = segyio.spec()
                spec.sorting = src.sorting
                spec.format = src.format
                spec.samples = src.samples
                
                # Handle 3D geometry if available
                if hasattr(src, 'ilines') and hasattr(src, 'xlines'):
                    spec.ilines = src.ilines
                    spec.xlines = src.xlines
                
                with segyio.open(output_filename, "w", spec) as dst:
                    # Copy textual headers
                    for i in range(len(src.text)):
                        dst.text[i] = src.text[i]
                    
                    # Copy binary header
                    dst.bin = src.bin
                    
                    # Copy headers and write enhanced data
                    n_traces = src.tracecount
                    enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
                    
                    for i in range(n_traces):
                        dst.header[i] = src.header[i]
                        if i < len(enhanced_flat):
                            dst.trace[i] = enhanced_flat[i].astype(np.float32)
                        else:
                            dst.trace[i] = src.trace[i]
            
            return True
        except Exception as e:
            st.warning(f"Simple SEG-Y writing failed: {e}")
            return False

    def create_downloadable_segy(self, output_filename):
        """Create enhanced SEG-Y file using the most reliable method"""
        if self.enhanced_data is None:
            st.error("No enhanced data available. Please process the data first.")
            return None
        
        # Ensure .sgy extension
        if not output_filename.lower().endswith(('.sgy', '.segy')):
            output_filename = os.path.splitext(output_filename)[0] + '.sgy'
        
        try:
            # Create temporary file with unique name
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())[:8]
            download_filename = os.path.join(temp_dir, f"enhanced_{unique_id}.sgy")
            
            st.info("Creating enhanced SEG-Y file...")
            
            # Try robust binary method first (most reliable)
            success = self.write_segy_robust(download_filename)
            
            if not success:
                # Fallback to simple method
                st.info("Trying alternative SEG-Y writing method...")
                success = self.write_segy_simple(download_filename)
            
            if success:
                # Verify file was created
                if os.path.exists(download_filename) and os.path.getsize(download_filename) > 0:
                    file_size = os.path.getsize(download_filename) / (1024 * 1024)
                    st.success(f"Enhanced SEG-Y file created successfully! Size: {file_size:.2f} MB")
                    
                    # Quick verification
                    try:
                        with segyio.open(download_filename, "r") as test_file:
                            st.info(f"File verified: {test_file.tracecount} traces, {len(test_file.samples)} samples")
                        return download_filename
                    except Exception as e:
                        st.warning(f"File created but verification failed: {e}")
                        return download_filename
                else:
                    st.error("File was not created properly")
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_traces = n_inlines * n_xlines
        processed_traces = 0
        
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = seismic_data[i, j, :].copy()
                enhanced_trace = self.safe_fft_processing(
                    trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range
                )
                enhanced_data[i, j, :] = enhanced_trace
                
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
        """Safe FFT processing with error handling"""
        try:
            if np.all(trace == 0):
                return trace
                
            if np.isnan(trace).any() or np.isinf(trace).any():
                trace = np.nan_to_num(trace)
                
            if np.std(trace) < 1e-10:
                return trace
                
            return self._apply_spectral_blueing(trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range)
            
        except Exception as e:
            st.warning(f"Trace processing failed: {e}. Using original trace.")
            return trace

    def _apply_spectral_blueing(self, trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range):
        """Apply spectral blueing to a single trace"""
        trace = signal.detrend(trace)
        trace_fft = fft(trace)
        freqs = fftfreq(len(trace), d=self.sample_rate/1000.0)
        
        weights = np.ones_like(freqs, dtype=complex)
        freq_magnitude = np.abs(freqs)
        
        # Low frequencies
        low_freq_mask = (freq_magnitude > 5) & (freq_magnitude <= mid_freq_range[0])
        weights[low_freq_mask] = low_freq_boost
        
        # Target frequencies
        target_mask = (freq_magnitude > mid_freq_range[0]) & (freq_magnitude <= target_freq)
        weights[target_mask] = enhancement_factor
        
        # High frequencies
        high_freq_mask = freq_magnitude > target_freq
        rolloff_factor = enhancement_factor * np.exp(-0.001 * (freq_magnitude[high_freq_mask] - target_freq)**2)
        weights[high_freq_mask] = np.maximum(1.0, rolloff_factor)
        
        enhanced_fft = trace_fft * weights
        enhanced_trace = np.real(ifft(enhanced_fft))
        
        if np.std(enhanced_trace) > 0:
            enhanced_trace = enhanced_trace * (np.std(trace) / np.std(enhanced_trace))
        
        return enhanced_trace

    def bandpass_filter(self, seismic_data, lowcut=8, highcut=120, order=3):
        """Apply bandpass filter"""
        st.info("Applying bandpass filter...")
        
        sampling_interval = self.sample_rate / 1000.0
        sampling_freq = 1.0 / sampling_interval
        nyquist = sampling_freq / 2.0
        
        if highcut >= nyquist * 0.95:
            highcut = nyquist * 0.9
            st.warning(f"Adjusted highcut to {highcut:.1f} Hz for stability")
        
        low_normalized = lowcut / nyquist
        high_normalized = highcut / nyquist
        
        try:
            b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        except ValueError:
            low_normalized = 0.05
            high_normalized = 0.45
            b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
        
        enhanced_data = np.zeros_like(seismic_data)
        n_inlines, n_xlines, n_samples = seismic_data.shape
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_traces = n_inlines * n_xlines
        processed_traces = 0
        
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = seismic_data[i, j, :].copy()
                trace = np.nan_to_num(trace)
                
                try:
                    enhanced_trace = signal.filtfilt(b, a, trace)
                    enhanced_data[i, j, :] = enhanced_trace
                except Exception as e:
                    st.warning(f"Error filtering trace {j}: {e}")
                    enhanced_data[i, j, :] = trace
                
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
        
        # Apply spectral blueing
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
        
        # Apply bandpass filter
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
    
    # Initialize session state
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
    
    enhancer = st.session_state.enhancer
    
    # Sidebar
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
        target_freq = 80
        enhancement_factor = 1.5
        low_freq_boost = 1.2
        mid_freq_start = 30
        lowcut = 8
        highcut = 120
        filter_order = 3
    
    # Parameter sliders
    st.sidebar.subheader("Spectral Blueing")
    target_freq = st.sidebar.slider("Target Frequency (Hz)", 30, 120, target_freq)
    enhancement_factor = st.sidebar.slider("Enhancement Factor", 1.0, 3.0, enhancement_factor, 0.1)
    low_freq_boost = st.sidebar.slider("Low Frequency Boost", 1.0, 2.0, low_freq_boost, 0.1)
    mid_freq_start = st.sidebar.slider("Mid Frequency Start (Hz)", 10, 50, mid_freq_start)
    
    st.sidebar.subheader("Bandpass Filter")
    lowcut = st.sidebar.slider("Low Cut (Hz)", 1, 50, lowcut)
    highcut = st.sidebar.slider("High Cut (Hz)", 60, 200, highcut)
    filter_order = st.sidebar.slider("Filter Order", 2, 6, filter_order)
    
    # Main processing
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
            st.session_state.original_filename = temp_filename
        
        try:
            if st.button("🚀 Process Seismic Data", type="primary", use_container_width=True):
                with st.spinner("Processing 3D seismic data..."):
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
                
                st.success("✅ 3D Processing completed!")
                st.session_state.data_processed = True
                st.session_state.file_generated = False
                st.session_state.enhanced_file_path = None
                st.session_state.enhanced_file_data = None

            # Download section
            if st.session_state.get('data_processed', False):
                st.sidebar.header("💾 Download Results")
                
                if st.sidebar.button("🛠️ Generate Enhanced SEG-Y File", type="secondary", use_container_width=True):
                    with st.sidebar:
                        with st.spinner("Creating enhanced SEG-Y file..."):
                            enhanced_file_path = enhancer.create_downloadable_segy("enhanced_seismic.sgy")
                            
                            if enhanced_file_path:
                                st.session_state.enhanced_file_path = enhanced_file_path
                                file_data = safe_file_download(enhanced_file_path, "enhanced_seismic.sgy")
                                if file_data is not None:
                                    st.session_state.enhanced_file_data = file_data
                                    st.session_state.file_generated = True
                                    st.sidebar.success("Enhanced SEG-Y file created successfully!")
                                else:
                                    st.sidebar.error("Failed to load file data")
                            else:
                                st.sidebar.error("Failed to create enhanced SEG-Y file")
                
                if st.session_state.get('file_generated', False) and st.session_state.enhanced_file_data is not None:
                    with st.sidebar:
                        file_data = st.session_state.enhanced_file_data
                        
                        st.download_button(
                            label="📥 Download Enhanced SEG-Y",
                            data=file_data,
                            file_name="enhanced_seismic.sgy",
                            mime="application/octet-stream",
                            help="Download the enhanced seismic data in SEG-Y format",
                            key="download_enhanced_data",
                            use_container_width=True
                        )
                        st.success("Enhanced SEG-Y file ready for download!")
                
                # Display results
                st.header("Processing Results")
                st.success("Data processing completed successfully!")
                
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
    
    else:
        st.info("👈 Please upload a 3D SEG-Y file to begin processing.")
        
        st.header("About This Tool")
        st.markdown("""
        This tool enhances 3D seismic data bandwidth using **spectral blueing** techniques.
        
        ### How to Use:
        1. Upload a SEG-Y file
        2. Adjust processing parameters
        3. Click "Process Seismic Data"
        4. Generate and download the enhanced SEG-Y file
        
        ### Features:
        - **Robust SEG-Y Support**: Handles various SEG-Y formats
        - **Spectral Enhancement**: Improves frequency content
        - **Bandpass Filtering**: Removes noise
        - **Download Ready**: Get enhanced data in standard SEG-Y format
        """)

if __name__ == "__main__":
    main()
