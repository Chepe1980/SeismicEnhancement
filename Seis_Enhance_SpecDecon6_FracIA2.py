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
import pandas as pd

# ============================================================================
# IMPORT WITH FALLBACKS
# ============================================================================

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not available. Using fallback implementations.")
    
    # Create simple fallbacks for OpenCV functions
    from scipy.ndimage import gaussian_filter, sobel, laplace
    
    class CV2Fallback:
        @staticmethod
        def GaussianBlur(src, ksize, sigmaX):
            return gaussian_filter(src, sigma=sigmaX)
        
        @staticmethod 
        def Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False):
            from scipy.ndimage import gaussian_gradient_magnitude
            edges = gaussian_gradient_magnitude(image, sigma=1)
            return (edges > threshold1).astype(np.uint8) * 255
        
        @staticmethod
        def filter2D(src, ddepth, kernel):
            from scipy.ndimage import convolve
            return convolve(src, kernel)
        
        @staticmethod
        def Sobel(src, ddepth, dx, dy, ksize=3):
            from scipy.ndimage import sobel
            if dx == 1 and dy == 0:
                return sobel(src, axis=1)
            elif dx == 0 and dy == 1:
                return sobel(src, axis=0)
            else:
                return np.zeros_like(src)
        
        @staticmethod
        def Laplacian(src, ddepth, ksize=3):
            from scipy.ndimage import laplace
            return laplace(src)
    
    cv2 = CV2Fallback()

# Try to import scikit-learn with fallback
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Using simplified ML methods.")

# ============================================================================
# MAIN CLASSES
# ============================================================================

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0
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
        return self.read_segy_3d(filename)

    def write_segy_numpy_based(self, output_filename):
        """Reliable SEG-Y writing using numpy and direct file operations"""
        if self.enhanced_data is None:
            st.error("No enhanced data available to write")
            return False
            
        try:
            # Read original file structure
            with segyio.open(self.original_filename, "r") as src:
                n_traces = src.tracecount
                n_samples = len(src.samples)
                
                # Get all textual headers
                textual_headers = []
                for i in range(len(src.text)):
                    textual_headers.append(src.text[i])
                
                # Get binary header information
                bin_header = src.bin
                
                # Get all trace headers
                trace_headers = []
                for i in range(n_traces):
                    trace_headers.append(dict(src.header[i]))
            
            # Flatten enhanced data
            enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
            
            # Create new SEG-Y file
            with open(output_filename, 'wb') as f:
                # 1. Write textual headers (3200 bytes each)
                for header in textual_headers:
                    # Ensure header is exactly 3200 bytes
                    header_bytes = header.encode('ascii', errors='replace')[:3200]
                    header_bytes = header_bytes.ljust(3200, b' ')
                    f.write(header_bytes)
                
                # 2. Write binary header (400 bytes)
                bin_header_bytes = self._create_binary_header(bin_header, n_samples)
                f.write(bin_header_bytes)
                
                # 3. Write traces with headers
                for i in range(n_traces):
                    # Write trace header (240 bytes)
                    trace_header_bytes = self._create_trace_header(trace_headers[i], n_samples)
                    f.write(trace_header_bytes)
                    
                    # Write trace data
                    if i < len(enhanced_flat):
                        trace_data = enhanced_flat[i].astype(np.float32)
                        # Ensure correct length
                        if len(trace_data) != n_samples:
                            if len(trace_data) > n_samples:
                                trace_data = trace_data[:n_samples]
                            else:
                                trace_data = np.pad(trace_data, (0, n_samples - len(trace_data)), 
                                                  mode='constant')
                        f.write(trace_data.tobytes())
                    else:
                        # Write zeros if no enhanced data
                        f.write(np.zeros(n_samples, dtype=np.float32).tobytes())
            
            st.success(f"SEG-Y file created successfully: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"SEG-Y writing failed: {e}")
            return False

    def _create_binary_header(self, bin_header, n_samples):
        """Create binary header bytes"""
        header_data = bytearray(400)  # 400 bytes for binary header
        
        # Set important binary header fields
        # Job identification number (bytes 0-3)
        job_id = bin_header.get(segyio.BinField.JobID, 1)
        struct.pack_into('>i', header_data, 0, job_id)
        
        # Line number (bytes 4-7)
        line_number = bin_header.get(segyio.BinField.LineNumber, 1)
        struct.pack_into('>i', header_data, 4, line_number)
        
        # Reel number (bytes 8-11)
        reel_number = bin_header.get(segyio.BinField.ReelNumber, 1)
        struct.pack_into('>i', header_data, 8, reel_number)
        
        # Number of data traces per ensemble (bytes 12-13)
        traces_ensemble = bin_header.get(segyio.BinField.Traces, 1)
        struct.pack_into('>h', header_data, 12, traces_ensemble)
        
        # Number of auxiliary traces per ensemble (bytes 14-15)
        aux_traces = bin_header.get(segyio.BinField.AuxTraces, 0)
        struct.pack_into('>h', header_data, 14, aux_traces)
        
        # Sample interval in microseconds (bytes 16-17)
        sample_interval = int(self.sample_rate * 1000)  # Convert ms to microseconds
        struct.pack_into('>h', header_data, 16, sample_interval)
        
        # Number of samples per data trace (bytes 20-21)
        struct.pack_into('>h', header_data, 20, n_samples)
        
        # Data sample format code (bytes 24-25) - 5 for IEEE floating point
        struct.pack_into('>h', header_data, 24, 5)
        
        # Ensemble fold (bytes 28-29)
        ensemble_fold = bin_header.get(segyio.BinField.EnsembleFold, 1)
        struct.pack_into('>h', header_data, 28, ensemble_fold)
        
        return header_data

    def _create_trace_header(self, trace_header, n_samples):
        """Create trace header bytes"""
        header_data = bytearray(240)  # 240 bytes for trace header
        
        # Set important trace header fields
        # Trace sequence number (bytes 0-3)
        trace_seq = trace_header.get(segyio.TraceField.TRACE_SEQUENCE_FILE, 1)
        struct.pack_into('>i', header_data, 0, trace_seq)
        
        # Field record number (bytes 8-11)
        field_record = trace_header.get(segyio.TraceField.FieldRecord, 1)
        struct.pack_into('>i', header_data, 8, field_record)
        
        # Trace number (bytes 12-15)
        trace_number = trace_header.get(segyio.TraceField.TRACE_NUMBER, 1)
        struct.pack_into('>i', header_data, 12, trace_number)
        
        # Energy source point number (bytes 16-19)
        source_point = trace_header.get(segyio.TraceField.SourcePoint, 1)
        struct.pack_into('>i', header_data, 16, source_point)
        
        # CDP number (bytes 20-23)
        cdp_number = trace_header.get(segyio.TraceField.CDP, 1)
        struct.pack_into('>i', header_data, 20, cdp_number)
        
        # CDP trace number (bytes 24-27)
        cdp_trace = trace_header.get(segyio.TraceField.CDP_TRACE, 1)
        struct.pack_into('>i', header_data, 24, cdp_trace)
        
        # Trace identification code (bytes 28-29)
        trace_id = trace_header.get(segyio.TraceField.TRACE_IDENTIFICATION_CODE, 1)
        struct.pack_into('>h', header_data, 28, trace_id)
        
        # Number of samples in this trace (bytes 114-115)
        struct.pack_into('>h', header_data, 114, n_samples)
        
        # Sample interval in microseconds (bytes 116-117)
        sample_interval = int(self.sample_rate * 1000)
        struct.pack_into('>h', header_data, 116, sample_interval)
        
        return header_data

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
            
            # Try methods in order of reliability
            success = self.write_segy_numpy_based(download_filename)
            
            if success:
                # Verify file was created
                if os.path.exists(download_filename) and os.path.getsize(download_filename) > 0:
                    file_size = os.path.getsize(download_filename) / (1024 * 1024)
                    st.success(f"Enhanced file created successfully! Size: {file_size:.2f} MB")
                    
                    # Quick verification for SEG-Y files
                    if download_filename.endswith(('.sgy', '.segy')):
                        try:
                            with segyio.open(download_filename, "r") as test_file:
                                st.info(f"SEG-Y verification: {test_file.tracecount} traces, {len(test_file.samples)} samples")
                        except:
                            st.warning("File created but SEG-Y verification failed")
                    
                    return download_filename
                else:
                    st.error("File was not created properly")
                    return None
            else:
                st.error("File creation method failed")
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

    def enhance_bandwidth(self, file_path, target_freq=80, enhancement_factor=1.5, low_freq_boost=1.2,
                         mid_freq_start=30, lowcut=8, highcut=120, filter_order=3):
        """Main method to enhance seismic bandwidth"""
        st.info(f"Loading SEG-Y file...")
        self.original_data = self.read_segy(file_path)
        
        if self.original_data is None:
            raise ValueError("Failed to load SEG-Y file")
        
        # Store parameters for metadata
        self.target_freq = target_freq
        self.enhancement_factor = enhancement_factor
        self.lowcut = lowcut
        self.highcut = highcut
        
        st.success(f"Original data shape: {self.original_data.shape}")
        st.info(f"Original data range: {np.min(self.original_data):.3f} to {np.max(self.original_data):.3f}")
        
        start_time = time.time()
        
        # Apply spectral blueing
        st.info("Starting spectral blueing...")
        self.enhanced_data = self.spectral_blueing(
            self.original_data,
            target_freq=target_freq,
            enhancement_factor=enhancement_factor,
            low_freq_boost=low_freq_boost,
            mid_freq_range=(mid_freq_start, target_freq)
        )
        
        # Apply bandpass filter
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

# ============================================================================
# FAULT DETECTION CLASSES
# ============================================================================

class SeismicAttributeCalculator:
    """Calculate advanced seismic attributes for fault and fracture detection"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.attributes = {}
        
    def calculate_all_attributes(self, window_size=11, gradient_kernel_size=3):
        """Calculate comprehensive set of seismic attributes"""
        st.info("Calculating advanced seismic attributes...")
        
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed = 0
        total_operations = 12  # Number of attributes to calculate
        
        # 1. Basic Attributes
        status_text.text("Calculating basic attributes...")
        self.attributes['amplitude'] = self.seismic_data.copy()
        self.attributes['energy'] = self.calculate_energy(window_size)
        processed += 2
        progress_bar.progress(processed / total_operations)
        
        # 2. Structural Attributes
        status_text.text("Calculating structural attributes...")
        self.attributes['dip'] = self.calculate_dip(gradient_kernel_size)
        self.attributes['azimuth'] = self.calculate_azimuth(gradient_kernel_size)
        self.attributes['curvature'] = self.calculate_curvature(gradient_kernel_size)
        self.attributes['coherence'] = self.calculate_coherence(window_size)
        processed += 4
        progress_bar.progress(processed / total_operations)
        
        # 3. Texture Attributes
        status_text.text("Calculating texture attributes...")
        self.attributes['variance'] = self.calculate_variance(window_size)
        self.attributes['entropy'] = self.calculate_entropy(window_size)
        processed += 2
        progress_bar.progress(processed / total_operations)
        
        # 4. Edge Detection Attributes
        status_text.text("Calculating edge detection attributes...")
        self.attributes['sobel'] = self.calculate_sobel_edge()
        self.attributes['laplacian'] = self.calculate_laplacian()
        processed += 2
        progress_bar.progress(processed / total_operations)
        
        # 5. Advanced Attributes
        status_text.text("Calculating advanced attributes...")
        self.attributes['instantaneous_phase'] = self.calculate_instantaneous_phase()
        self.attributes['instantaneous_frequency'] = self.calculate_instantaneous_frequency()
        self.attributes['sweetness'] = self.calculate_sweetness()
        processed += 3
        progress_bar.progress(1.0)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Calculated {len(self.attributes)} seismic attributes")
        
        return self.attributes
    
    def calculate_energy(self, window_size):
        """Calculate energy attribute (RMS amplitude)"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        energy = np.zeros_like(self.seismic_data)
        
        # Use vectorized operations for speed
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = self.seismic_data[i, j, :]
                # Use rolling window for RMS
                if len(trace) > window_size:
                    trace_sq = trace**2
                    energy[i, j, :] = np.sqrt(np.convolve(trace_sq, np.ones(window_size)/window_size, mode='same'))
                else:
                    energy[i, j, :] = np.sqrt(np.mean(trace**2))
        
        return energy
    
    def calculate_dip(self, kernel_size=3):
        """Calculate dip attribute using gradient"""
        gradient_z = np.gradient(self.seismic_data, axis=2)
        gradient_x = np.gradient(self.seismic_data, axis=0)
        
        # Calculate dip magnitude
        dip = np.arctan2(np.abs(gradient_z), np.abs(gradient_x)) * 180 / np.pi
        
        return dip
    
    def calculate_azimuth(self, kernel_size=3):
        """Calculate azimuth attribute"""
        gradient_y = np.gradient(self.seismic_data, axis=1)
        gradient_x = np.gradient(self.seismic_data, axis=0)
        
        azimuth = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        azimuth = (azimuth + 360) % 360  # Normalize to 0-360
        
        return azimuth
    
    def calculate_curvature(self, kernel_size=3):
        """Calculate curvature attribute"""
        # First derivatives
        gradient_x = np.gradient(self.seismic_data, axis=0)
        gradient_y = np.gradient(self.seismic_data, axis=1)
        gradient_z = np.gradient(self.seismic_data, axis=2)
        
        # Second derivatives
        gradient_xx = np.gradient(gradient_x, axis=0)
        gradient_yy = np.gradient(gradient_y, axis=1)
        gradient_zz = np.gradient(gradient_z, axis=2)
        
        # Mixed derivatives
        gradient_xy = np.gradient(gradient_x, axis=1)
        gradient_xz = np.gradient(gradient_x, axis=2)
        gradient_yz = np.gradient(gradient_y, axis=2)
        
        # Calculate Gaussian curvature
        numerator = (gradient_xx * gradient_yy * gradient_zz + 
                    2 * gradient_xy * gradient_xz * gradient_yz -
                    gradient_xx * gradient_yz**2 - 
                    gradient_yy * gradient_xz**2 - 
                    gradient_zz * gradient_xy**2)
        
        denominator = (gradient_x**2 + gradient_y**2 + gradient_z**2 + 1e-10)**2
        curvature = numerator / denominator
        
        return curvature
    
    def calculate_coherence(self, window_size=9):
        """Calculate seismic coherence (similarity)"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        coherence = np.zeros_like(self.seismic_data)
        
        # Simplified coherence calculation using gradient similarity
        grad_x = np.gradient(self.seismic_data, axis=0)
        grad_y = np.gradient(self.seismic_data, axis=1)
        grad_z = np.gradient(self.seismic_data, axis=2)
        
        # Calculate gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalize gradients
        grad_x_norm = np.where(grad_mag > 0, grad_x / grad_mag, 0)
        grad_y_norm = np.where(grad_mag > 0, grad_y / grad_mag, 0)
        grad_z_norm = np.where(grad_mag > 0, grad_z / grad_mag, 0)
        
        # Calculate coherence as similarity of gradient directions
        half_window = window_size // 2
        
        # Limit calculation to avoid memory issues
        max_inlines = min(n_inlines, 100)
        max_xlines = min(n_xlines, 100)
        max_samples = min(n_samples, 100)
        
        for i in range(half_window, max_inlines - half_window):
            for j in range(half_window, max_xlines - half_window):
                for k in range(half_window, max_samples - half_window):
                    # Get gradient vectors in window
                    gx_window = grad_x_norm[i-half_window:i+half_window+1,
                                          j-half_window:j+half_window+1,
                                          k-half_window:k+half_window+1]
                    gy_window = grad_y_norm[i-half_window:i+half_window+1,
                                          j-half_window:j+half_window+1,
                                          k-half_window:k+half_window+1]
                    gz_window = grad_z_norm[i-half_window:i+half_window+1,
                                          j-half_window:j+half_window+1,
                                          k-half_window:k+half_window+1]
                    
                    # Calculate similarity (dot product)
                    center_gx = grad_x_norm[i, j, k]
                    center_gy = grad_y_norm[i, j, k]
                    center_gz = grad_z_norm[i, j, k]
                    
                    dot_products = (gx_window * center_gx + 
                                  gy_window * center_gy + 
                                  gz_window * center_gz)
                    
                    coherence[i, j, k] = np.mean(dot_products)
        
        return coherence
    
    def calculate_variance(self, window_size):
        """Calculate variance attribute"""
        return self._apply_window_operation(np.var, window_size)
    
    def calculate_entropy(self, window_size):
        """Calculate entropy attribute"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        entropy = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        # Limit calculation
        max_inlines = min(n_inlines, 50)
        max_xlines = min(n_xlines, 50)
        max_samples = min(n_samples, 50)
        
        for i in range(half_window, max_inlines - half_window):
            for j in range(half_window, max_xlines - half_window):
                for k in range(half_window, max_samples - half_window):
                    window = self.seismic_data[i-half_window:i+half_window+1,
                                             j-half_window:j+half_window+1,
                                             k-half_window:k+half_window+1]
                    # Normalize window values
                    window_norm = (window - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)
                    # Calculate histogram
                    hist, _ = np.histogram(window_norm, bins=16, range=(0, 1))
                    hist = hist / (hist.sum() + 1e-10)
                    # Calculate entropy
                    entropy[i, j, k] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _apply_window_operation(self, operation, window_size):
        """Helper function to apply window-based operations"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        result = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        # Optimized using vectorized operations
        for i in range(min(n_inlines, 100)):  # Limit for performance
            for j in range(min(n_xlines, 100)):
                trace = self.seismic_data[i, j, :]
                if len(trace) > window_size:
                    # Use rolling window
                    trace_2d = np.lib.stride_tricks.sliding_window_view(trace, window_size)
                    result[i, j, half_window:-half_window] = operation(trace_2d, axis=1)
                    # Fill edges
                    result[i, j, :half_window] = result[i, j, half_window]
                    result[i, j, -half_window:] = result[i, j, -half_window-1]
        
        return result
    
    def calculate_sobel_edge(self):
        """Calculate Sobel edge detection"""
        sobel_magnitude = np.zeros_like(self.seismic_data)
        
        # Use numpy gradient for Sobel-like effect
        for i in range(1, min(self.seismic_data.shape[0], 100) - 1):
            for j in range(1, min(self.seismic_data.shape[1], 100) - 1):
                for k in range(1, min(self.seismic_data.shape[2], 100) - 1):
                    # Simple 3x3x3 Sobel approximation
                    dx = (self.seismic_data[i+1, j, k] - self.seismic_data[i-1, j, k]) / 2
                    dy = (self.seismic_data[i, j+1, k] - self.seismic_data[i, j-1, k]) / 2
                    dz = (self.seismic_data[i, j, k+1] - self.seismic_data[i, j, k-1]) / 2
                    
                    sobel_magnitude[i, j, k] = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return sobel_magnitude
    
    def calculate_laplacian(self):
        """Calculate Laplacian edge detection"""
        laplacian = np.zeros_like(self.seismic_data)
        
        # Use second derivatives for Laplacian
        for i in range(1, min(self.seismic_data.shape[0], 100) - 1):
            for j in range(1, min(self.seismic_data.shape[1], 100) - 1):
                for k in range(1, min(self.seismic_data.shape[2], 100) - 1):
                    # 3D Laplacian
                    lap = (self.seismic_data[i+1, j, k] + self.seismic_data[i-1, j, k] +
                          self.seismic_data[i, j+1, k] + self.seismic_data[i, j-1, k] +
                          self.seismic_data[i, j, k+1] + self.seismic_data[i, j, k-1] -
                          6 * self.seismic_data[i, j, k])
                    
                    laplacian[i, j, k] = lap
        
        return laplacian
    
    def calculate_instantaneous_phase(self):
        """Calculate instantaneous phase using Hilbert transform"""
        # Use scipy's hilbert transform on a subset for performance
        subset = self.seismic_data[:min(50, self.seismic_data.shape[0]), 
                                  :min(50, self.seismic_data.shape[1]), 
                                  :min(100, self.seismic_data.shape[2])]
        analytic_signal = signal.hilbert(subset, axis=2)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Create full array with zeros
        full_phase = np.zeros_like(self.seismic_data)
        full_phase[:instantaneous_phase.shape[0], 
                  :instantaneous_phase.shape[1], 
                  :instantaneous_phase.shape[2]] = instantaneous_phase
        
        return full_phase
    
    def calculate_instantaneous_frequency(self):
        """Calculate instantaneous frequency"""
        instantaneous_phase = self.calculate_instantaneous_phase()
        dt = self.sample_rate / 1000.0
        
        # Calculate on subset
        subset_phase = instantaneous_phase[:min(50, instantaneous_phase.shape[0]),
                                         :min(50, instantaneous_phase.shape[1]), 
                                         :min(100, instantaneous_phase.shape[2])]
        
        instantaneous_frequency = np.gradient(subset_phase, axis=2) / (2 * np.pi * dt)
        
        # Create full array
        full_freq = np.zeros_like(self.seismic_data)
        full_freq[:instantaneous_frequency.shape[0],
                 :instantaneous_frequency.shape[1],
                 :instantaneous_frequency.shape[2]] = instantaneous_frequency
        
        return full_freq
    
    def calculate_sweetness(self):
        """Calculate sweetness attribute (amplitude/frequency)"""
        amplitude = np.abs(self.seismic_data)
        frequency = self.calculate_instantaneous_frequency()
        frequency = np.where(frequency > 0, frequency, 1e-10)  # Avoid division by zero
        sweetness = amplitude / frequency
        return sweetness

class SimpleFaultDetector:
    """Simplified fault detector for when advanced ML libraries are not available"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.fault_probabilities = None
        
    def detect_faults_simple(self):
        """Simple fault detection based on edge detection and coherence"""
        st.info("Running simplified fault detection...")
        
        # Calculate basic attributes
        attribute_calc = SeismicAttributeCalculator(self.seismic_data, self.sample_rate)
        attributes = attribute_calc.calculate_all_attributes()
        
        # Combine attributes for fault detection
        coherence = attributes.get('coherence', np.zeros_like(self.seismic_data))
        sobel = attributes.get('sobel', np.zeros_like(self.seismic_data))
        curvature = attributes.get('curvature', np.zeros_like(self.seismic_data))
        
        # Normalize each attribute
        def normalize(data):
            if data.size == 0:
                return np.zeros_like(self.seismic_data)
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            return np.zeros_like(data)
        
        coherence_norm = normalize(coherence)
        sobel_norm = normalize(sobel)
        curvature_norm = normalize(curvature)
        
        # Faults are low coherence, high gradient, high curvature
        fault_prob = (1 - coherence_norm) * 0.4 + sobel_norm * 0.4 + curvature_norm * 0.2
        
        self.fault_probabilities = fault_prob
        
        return fault_prob
    
    def visualize_faults(self, inline_idx=None, crossline_idx=None, time_slice=None):
        """Visualize detected faults - FIXED VERSION"""
        if self.fault_probabilities is None:
            st.error("No fault detection results available.")
            return None
        
        n_inlines, n_xlines, n_samples = self.fault_probabilities.shape
        
        # Determine visualization type
        if inline_idx is not None:
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
            
        elif crossline_idx is not None:
            data_slice = self.fault_probabilities[:, crossline_idx, :]
            title = f"Fault Detection - Crossline {crossline_idx}"
            
        elif time_slice is not None:
            data_slice = self.fault_probabilities[:, :, time_slice]
            title = f"Fault Detection - Time Slice {time_slice}"
            
        else:
            inline_idx = n_inlines // 2
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
        
        # Transpose for proper orientation
        if inline_idx is not None or crossline_idx is not None:
            display_data = data_slice.T
        else:
            display_data = data_slice
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=display_data,
            colorscale='hot',
            opacity=0.8,
            name='Fault Probability',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability")
        ))
        
        # Determine axis labels
        if inline_idx is not None:
            xaxis_title = "Crossline"
            yaxis_title = "Time Sample"
        elif crossline_idx is not None:
            xaxis_title = "Inline"
            yaxis_title = "Time Sample"
        else:
            xaxis_title = "Crossline"
            yaxis_title = "Inline"
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig

class AdvancedFaultDetector:
    """Advanced fault and fracture detection using available ML methods"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.attribute_calculator = SeismicAttributeCalculator(seismic_data, sample_rate)
        self.fault_probabilities = None
        self.fracture_network = None
        
    def detect_faults(self, method='ensemble'):
        """Detect faults using available methods"""
        st.info(f"Starting {method} fault detection...")
        
        # Calculate attributes
        attributes = self.attribute_calculator.calculate_all_attributes()
        
        if method == 'simple' or not SKLEARN_AVAILABLE:
            # Use simple method if sklearn not available
            simple_detector = SimpleFaultDetector(self.seismic_data, self.sample_rate)
            faults = simple_detector.detect_faults_simple()
            self.fault_probabilities = faults
            return faults, None
        
        elif method == 'ensemble' and SKLEARN_AVAILABLE:
            # Use ensemble method
            return self._ensemble_detection(attributes)
        
        elif method == 'anomaly' and SKLEARN_AVAILABLE:
            # Use anomaly detection
            return self._anomaly_detection(attributes)
        
        else:
            # Default to simple method
            simple_detector = SimpleFaultDetector(self.seismic_data, self.sample_rate)
            faults = simple_detector.detect_faults_simple()
            self.fault_probabilities = faults
            return faults, None
    
    def _ensemble_detection(self, attributes):
        """Ensemble detection using available sklearn methods"""
        st.info("Running ensemble fault detection...")
        
        # Prepare feature cube
        feature_cube = self._prepare_feature_cube(attributes)
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        
        # Flatten features for ML
        X = feature_cube.reshape(-1, n_features)
        
        # Sample for efficiency
        n_samples_train = min(5000, len(X))
        indices = np.random.choice(len(X), n_samples_train, replace=False)
        X_train = X[indices]
        
        # Generate synthetic labels for training
        y_train = self._generate_synthetic_labels(X_train)
        
        # Train Random Forest if available
        if SKLEARN_AVAILABLE:
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predict in batches
            batch_size = 10000
            predictions = np.zeros(len(X))
            
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                X_batch = X[i:end_idx]
                pred_batch = model.predict_proba(X_batch)[:, 1]
                predictions[i:end_idx] = pred_batch
            
            fault_prob = predictions.reshape(n_inlines, n_xlines, n_samples)
        
        else:
            # Fallback to simple method
            coherence = attributes.get('coherence', np.zeros_like(self.seismic_data))
            sobel = attributes.get('sobel', np.zeros_like(self.seismic_data))
            fault_prob = (1 - coherence) * 0.5 + sobel * 0.5
        
        self.fault_probabilities = fault_prob
        
        # Build fracture network
        try:
            self.fracture_network = self._build_fracture_network(fault_prob)
        except Exception as e:
            st.warning(f"Could not build fracture network: {e}")
            self.fracture_network = None
        
        return fault_prob, self.fracture_network
    
    def _prepare_feature_cube(self, attributes):
        """Prepare feature cube from attributes"""
        feature_list = []
        
        # Select key attributes
        key_attributes = ['coherence', 'dip', 'curvature', 'sobel', 
                         'variance', 'entropy', 'instantaneous_frequency']
        
        for attr_name in key_attributes:
            if attr_name in attributes:
                attr_data = attributes[attr_name]
                # Normalize
                if attr_data.size > 0:
                    attr_min, attr_max = attr_data.min(), attr_data.max()
                    if attr_max > attr_min:
                        attr_norm = (attr_data - attr_min) / (attr_max - attr_min)
                    else:
                        attr_norm = np.zeros_like(attr_data)
                    feature_list.append(attr_norm)
        
        # Add amplitude
        amplitude = attributes['amplitude']
        if amplitude.size > 0:
            amp_min, amp_max = amplitude.min(), amplitude.max()
            if amp_max > amp_min:
                amp_norm = (amplitude - amp_min) / (amp_max - amp_min)
            else:
                amp_norm = np.zeros_like(amplitude)
            feature_list.append(amp_norm)
        
        # Stack features
        if feature_list:
            feature_cube = np.stack(feature_list, axis=-1)
        else:
            # Create dummy feature cube
            shape = list(self.seismic_data.shape) + [1]
            feature_cube = np.zeros(shape)
        
        return feature_cube
    
    def _generate_synthetic_labels(self, X):
        """Generate synthetic labels for training"""
        # Based on feature patterns
        weights = np.ones(X.shape[1]) * 0.1
        if len(weights) > 0:
            weights[0] = 0.3  # Coherence weight
        if len(weights) > 2:
            weights[2] = 0.2  # Curvature weight
        if len(weights) > 3:
            weights[3] = 0.2  # Sobel weight
        
        scores = np.dot(X, weights)
        if scores.std() > 0:
            scores = (scores - scores.mean()) / scores.std()
        labels = (scores > 1.0).astype(int)
        
        # Add noise
        noise = np.random.rand(len(labels)) > 0.97
        labels[noise] = 1 - labels[noise]
        
        return labels
    
    def _anomaly_detection(self, attributes):
        """Anomaly detection based fault detection"""
        st.info("Running anomaly detection...")
        
        feature_cube = self._prepare_feature_cube(attributes)
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        
        X = feature_cube.reshape(-1, n_features)
        
        if SKLEARN_AVAILABLE:
            # Sample for efficiency
            n_samples_anomaly = min(3000, len(X))
            indices = np.random.choice(len(X), n_samples_anomaly, replace=False)
            X_sample = X[indices]
            
            iso_forest = IsolationForest(
                n_estimators=50,
                contamination=0.1,
                random_state=42
            )
            
            iso_forest.fit(X_sample)
            
            # Predict in batches
            batch_size = 10000
            anomaly_scores = np.zeros(len(X))
            
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                X_batch = X[i:end_idx]
                scores_batch = iso_forest.decision_function(X_batch)
                anomaly_scores[i:end_idx] = scores_batch
            
            # Convert to probability
            anomaly_scores = -anomaly_scores  # Invert
            if anomaly_scores.max() > anomaly_scores.min():
                anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            else:
                anomaly_scores = np.zeros_like(anomaly_scores)
            
            fault_prob = anomaly_scores.reshape(n_inlines, n_xlines, n_samples)
        
        else:
            # Fallback
            coherence = attributes.get('coherence', np.zeros_like(self.seismic_data))
            fault_prob = 1 - coherence
        
        self.fault_probabilities = fault_prob
        
        # Build fracture network
        try:
            self.fracture_network = self._build_fracture_network(fault_prob)
        except Exception as e:
            st.warning(f"Could not build fracture network: {e}")
            self.fracture_network = None
        
        return fault_prob, self.fracture_network
    
    def _build_fracture_network(self, fault_probabilities):
        """Build connected fracture network from fault probabilities"""
        # Use a subset for performance
        subset_size = min(50, fault_probabilities.shape[0])
        fault_subset = fault_probabilities[:subset_size, :subset_size, :subset_size]
        
        # Threshold fault probabilities
        fault_binary = fault_subset > 0.5
        
        # Simple connected component analysis
        from scipy.ndimage import label
        labeled_array, num_features = label(fault_binary)
        
        fracture_properties = []
        for i in range(1, min(num_features + 1, 100)):  # Limit to 100 features
            fracture_mask = labeled_array == i
            if np.sum(fracture_mask) > 10:  # Minimum size
                indices = np.where(fracture_mask)
                center = [np.mean(idx) for idx in indices]
                size = np.sum(fracture_mask)
                
                # Simple orientation
                if len(indices[0]) > 2:
                    coords = np.array(indices).T
                    cov = np.cov(coords.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    orientation = eigenvectors[:, np.argmax(eigenvalues)]
                    
                    fracture_properties.append({
                        'id': i,
                        'center': center,
                        'size': size,
                        'orientation': orientation.tolist()
                    })
        
        st.info(f"Detected {len(fracture_properties)} fracture segments")
        return fracture_properties
    
    def visualize_faults(self, inline_idx=None, crossline_idx=None, time_slice=None):
        """Visualize detected faults"""
        if self.fault_probabilities is None:
            st.error("No fault detection results available.")
            return None
        
        # Create a simple detector for visualization
        simple_detector = SimpleFaultDetector(self.seismic_data, self.sample_rate)
        simple_detector.fault_probabilities = self.fault_probabilities
        
        return simple_detector.visualize_faults(
            inline_idx, crossline_idx, time_slice
        )
    
    def export_fault_network(self, output_format='csv'):
        """Export fault network data"""
        if self.fracture_network is None:
            st.error("No fracture network available.")
            return None
        
        try:
            export_data = []
            for fracture in self.fracture_network:
                export_data.append({
                    'fracture_id': fracture['id'],
                    'center_inline': f"{fracture['center'][0]:.1f}",
                    'center_crossline': f"{fracture['center'][1]:.1f}",
                    'center_time': f"{fracture['center'][2]:.1f}",
                    'size_voxels': fracture['size']
                })
            
            if output_format == 'csv':
                df = pd.DataFrame(export_data)
                return df.to_csv(index=False)
            elif output_format == 'json':
                return json.dumps(export_data, indent=2)
            else:
                return export_data
                
        except Exception as e:
            st.error(f"Error exporting fault network: {e}")
            return None

# ============================================================================
# PROCESSING PRESETS AND CONSTANTS
# ============================================================================

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

COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'hot', 'cool', 'jet', 'rainbow', 'turbo',
    'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
    'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG', 'RdGy',
    'Reds', 'Greens', 'Blues', 'Oranges', 'Purples'
]

FAULT_DETECTION_METHODS = {
    'simple': 'Simple (Edge Detection)',
    'ensemble': 'Ensemble (Random Forest)',
    'anomaly': 'Anomaly Detection'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def compute_frequency_spectrum(trace, sample_rate):
    """Compute frequency spectrum for a trace"""
    if trace is None or len(trace) == 0:
        return np.array([]), np.array([])
    
    try:
        n = len(trace)
        fft_result = fft(trace)
        freqs = fftfreq(n, d=sample_rate/1000.0)
        
        positive_freq_idx = freqs > 0
        freqs_positive = freqs[positive_freq_idx]
        amplitude = np.abs(fft_result[positive_freq_idx])
        
        return freqs_positive, amplitude
    except Exception as e:
        st.warning(f"Error computing frequency spectrum: {e}")
        return np.array([]), np.array([])

def safe_get_colormap_index(colormap_name, default_index=0):
    """Safely get the index of a colormap"""
    try:
        return COLORMAPS.index(colormap_name)
    except ValueError:
        return default_index

# ============================================================================
# TAB FUNCTIONS
# ============================================================================

def display_bandwidth_enhancement_tab(enhancer):
    """Display the bandwidth enhancement tab"""
    st.title("🌊 3D Seismic Bandwidth Enhancement Tool")
    
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format",
        key="bw_uploader"
    )
    
    st.sidebar.header("⚙️ Processing Parameters")
    
    st.sidebar.subheader("Processing Presets")
    preset = st.sidebar.selectbox(
        "Choose Processing Preset",
        options=list(PROCESSING_PRESETS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="bw_preset"
    )
    
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
    
    st.sidebar.subheader("Spectral Blueing")
    target_freq = st.sidebar.slider("Target Frequency (Hz)", 30, 120, target_freq, key="bw_target_freq")
    enhancement_factor = st.sidebar.slider("Enhancement Factor", 1.0, 3.0, enhancement_factor, 0.1, key="bw_enhancement")
    low_freq_boost = st.sidebar.slider("Low Frequency Boost", 1.0, 2.0, low_freq_boost, 0.1, key="bw_low_boost")
    mid_freq_start = st.sidebar.slider("Mid Frequency Start (Hz)", 10, 50, mid_freq_start, key="bw_mid_start")
    
    st.sidebar.subheader("Bandpass Filter")
    lowcut = st.sidebar.slider("Low Cut (Hz)", 1, 50, lowcut, key="bw_lowcut")
    highcut = st.sidebar.slider("High Cut (Hz)", 60, 200, highcut, key="bw_highcut")
    filter_order = st.sidebar.slider("Filter Order", 2, 6, filter_order, key="bw_filter_order")
    
    st.sidebar.header("🎨 Visualization Settings")
    st.sidebar.subheader("Colormap Selection")
    amplitude_colormap = st.sidebar.selectbox("Amplitude Colormap", COLORMAPS, index=0, key="bw_amp_cmap")
    difference_colormap = st.sidebar.selectbox("Difference Colormap", COLORMAPS, 
                                             index=safe_get_colormap_index('RdBu', 17), key="bw_diff_cmap")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
            st.session_state.original_filename = temp_filename
        
        try:
            if st.button("🚀 Process Seismic Data", type="primary", use_container_width=True, key="bw_process"):
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

            if st.session_state.get('data_processed', False):
                st.sidebar.header("💾 Download Results")
                
                if st.sidebar.button("🛠️ Generate Enhanced File", type="secondary", use_container_width=True, key="bw_generate"):
                    with st.sidebar:
                        with st.spinner("Creating enhanced file..."):
                            enhanced_file_path = enhancer.create_downloadable_segy("enhanced_data")
                            
                            if enhanced_file_path:
                                st.session_state.enhanced_file_path = enhanced_file_path
                                file_data = safe_file_download(enhanced_file_path, "enhanced_data")
                                if file_data is not None:
                                    st.session_state.enhanced_file_data = file_data
                                    st.session_state.file_generated = True
                                    
                                    if enhanced_file_path.endswith('.sgy'):
                                        st.sidebar.success("Enhanced SEG-Y file created successfully!")
                                    else:
                                        st.sidebar.success("Enhanced data file created successfully!")
                                else:
                                    st.sidebar.error("Failed to load file data")
                            else:
                                st.sidebar.error("Failed to create enhanced file")
                
                if st.session_state.get('file_generated', False) and st.session_state.enhanced_file_data is not None:
                    with st.sidebar:
                        file_data = st.session_state.enhanced_file_data
                        file_path = st.session_state.enhanced_file_path
                        
                        if file_path.endswith('.sgy'):
                            download_name = "enhanced_seismic.sgy"
                            label = "📥 Download Enhanced SEG-Y"
                        else:
                            download_name = "enhanced_seismic_data.dat"
                            label = "📥 Download Enhanced Data"
                        
                        st.download_button(
                            label=label,
                            data=file_data,
                            file_name=download_name,
                            mime="application/octet-stream",
                            help="Download the enhanced seismic data",
                            key="bw_download",
                            use_container_width=True
                        )
                
                # Display results
                st.header("Processing Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Data")
                    st.write(f"Shape: {enhancer.original_data.shape}")
                    st.write(f"Range: {np.min(enhancer.original_data):.3f} to {np.max(enhancer.original_data):.3f}")
                
                with col2:
                    st.subheader("Enhanced Data")
                    st.write(f"Shape: {enhancer.enhanced_data.shape}")
                    st.write(f"Range: {np.min(enhancer.enhanced_data):.3f} to {np.max(enhancer.enhanced_data):.3f}")
                
                # Interactive visualization
                st.header("📊 Interactive Data Comparison")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    section_type = st.selectbox("Select Section Type", 
                                              ["Inline", "Crossline", "Time Slice"],
                                              key="bw_section_type")
                
                n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                
                inline_idx = n_inlines // 2 if n_inlines > 0 else 0
                crossline_idx = n_xlines // 2 if n_xlines > 0 else 0
                time_slice = n_samples // 2 if n_samples > 0 else 0
                
                if section_type == "Inline":
                    with col2:
                        inline_idx = st.slider("Inline", 0, n_inlines-1, inline_idx, key="bw_inline")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_inline_display")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        original_inline = enhancer.original_data[inline_idx, :, :]
                        fig_orig = px.imshow(original_inline.T, 
                                           title=f"Original Inline {inline_idx}",
                                           color_continuous_scale=amplitude_colormap,
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_inline = enhancer.enhanced_data[inline_idx, :, :]
                            fig_enh = px.imshow(enhanced_inline.T, 
                                              title=f"Enhanced Inline {inline_idx}",
                                              color_continuous_scale=amplitude_colormap,
                                              aspect='auto')
                        else:
                            diff_inline = enhancer.enhanced_data[inline_idx, :, :] - enhancer.original_data[inline_idx, :, :]
                            fig_enh = px.imshow(diff_inline.T, 
                                              title=f"Difference Inline {inline_idx}",
                                              color_continuous_scale=difference_colormap,
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                elif section_type == "Crossline":
                    with col2:
                        crossline_idx = st.slider("Crossline", 0, n_xlines-1, crossline_idx, key="bw_crossline")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_xline_display")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        original_xline = enhancer.original_data[:, crossline_idx, :]
                        fig_orig = px.imshow(original_xline.T, 
                                           title=f"Original Crossline {crossline_idx}",
                                           color_continuous_scale=amplitude_colormap,
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_xline = enhancer.enhanced_data[:, crossline_idx, :]
                            fig_enh = px.imshow(enhanced_xline.T, 
                                              title=f"Enhanced Crossline {crossline_idx}",
                                              color_continuous_scale=amplitude_colormap,
                                              aspect='auto')
                        else:
                            diff_xline = enhancer.enhanced_data[:, crossline_idx, :] - enhancer.original_data[:, crossline_idx, :]
                            fig_enh = px.imshow(diff_xline.T, 
                                              title=f"Difference Crossline {crossline_idx}",
                                              color_continuous_scale=difference_colormap,
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                else:  # Time Slice
                    with col2:
                        time_slice = st.slider("Time Slice", 0, n_samples-1, time_slice, key="bw_time")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_time_display")
                    
                    actual_time = enhancer.geometry['samples'][time_slice] if enhancer.geometry and 'samples' in enhancer.geometry else time_slice
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        original_slice = enhancer.original_data[:, :, time_slice]
                        fig_orig = px.imshow(original_slice, 
                                           title=f"Original Time Slice {actual_time} ms",
                                           color_continuous_scale=amplitude_colormap,
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_slice = enhancer.enhanced_data[:, :, time_slice]
                            fig_enh = px.imshow(enhanced_slice, 
                                              title=f"Enhanced Time Slice {actual_time} ms",
                                              color_continuous_scale=amplitude_colormap,
                                              aspect='auto')
                        else:
                            diff_slice = enhancer.enhanced_data[:, :, time_slice] - enhancer.original_data[:, :, time_slice]
                            fig_enh = px.imshow(diff_slice, 
                                              title=f"Difference Time Slice {actual_time} ms",
                                              color_continuous_scale=difference_colormap,
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👈 Please upload a 3D SEG-Y file to begin processing.")

def display_fault_detection_tab(enhancer):
    """Display the fault detection tab - FIXED VERSION"""
    st.header("🔍 Fault & Fracture Detection")
    
    if enhancer.enhanced_data is None:
        st.info("Please process seismic data in the Bandwidth Enhancement tab first.")
        st.warning("No enhanced data available. Run bandwidth enhancement first.")
        return
    
    st.sidebar.header("⚙️ Fault Detection Parameters")
    
    st.sidebar.subheader("Detection Method")
    method = st.sidebar.selectbox(
        "Select Detection Method",
        options=list(FAULT_DETECTION_METHODS.keys()),
        format_func=lambda x: FAULT_DETECTION_METHODS[x],
        key="fd_method"
    )
    
    st.sidebar.subheader("Detection Sensitivity")
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher values reduce false positives but may miss subtle faults",
        key="fd_threshold"
    )
    
    st.sidebar.header("🎨 Visualization Settings")
    fault_colormap = st.sidebar.selectbox(
        "Fault Probability Colormap",
        COLORMAPS,
        index=safe_get_colormap_index('hot', 6),
        key="fd_cmap"
    )
    
    # Main detection section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Run Fault Detection", type="primary", use_container_width=True, key="fd_run"):
            with st.spinner(f"Running {FAULT_DETECTION_METHODS[method]} detection..."):
                # Initialize fault detector
                fault_detector = AdvancedFaultDetector(
                    enhancer.enhanced_data,
                    enhancer.sample_rate
                )
                
                # Run detection
                faults, fracture_network = fault_detector.detect_faults(method=method)
                
                # Store in session state
                st.session_state.fault_detector = fault_detector
                st.session_state.faults = faults
                st.session_state.fracture_network = fracture_network
                st.session_state.detection_completed = True
                st.session_state.detection_method = method
                st.session_state.detection_threshold = threshold
                
                st.success("✅ Fault detection completed!")
    
    # Display results if available
    if st.session_state.get('detection_completed', False):
        fault_detector = st.session_state.fault_detector
        faults = st.session_state.faults
        
        st.subheader("📈 Detection Results")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_voxels = np.prod(faults.shape)
            fault_voxels = np.sum(faults > threshold)
            fault_percentage = (fault_voxels / total_voxels) * 100
            st.metric("Fault Probability > Threshold", f"{fault_percentage:.2f}%")
        
        with col2:
            avg_probability = np.mean(faults)
            st.metric("Average Probability", f"{avg_probability:.3f}")
        
        with col3:
            if st.session_state.fracture_network:
                st.metric("Fracture Segments", len(st.session_state.fracture_network))
            else:
                st.metric("Fracture Segments", "0")
        
        # Interactive visualization
        st.subheader("📊 Interactive Visualization")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            view_type = st.selectbox(
                "View Type",
                ["Inline", "Crossline", "Time Slice"],
                key="fd_view_type"
            )
        
        n_inlines, n_xlines, n_samples = faults.shape
        
        if view_type == "Inline":
            with col2:
                inline_idx = st.slider(
                    "Inline Index",
                    0, n_inlines - 1,
                    n_inlines // 2,
                    key="fd_inline"
                )
            
            # Get the visualization
            fig = fault_detector.visualize_faults(inline_idx=inline_idx)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate visualization")
        
        elif view_type == "Crossline":
            with col2:
                crossline_idx = st.slider(
                    "Crossline Index",
                    0, n_xlines - 1,
                    n_xlines // 2,
                    key="fd_crossline"
                )
            
            fig = fault_detector.visualize_faults(crossline_idx=crossline_idx)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate visualization")
        
        else:  # Time Slice
            with col2:
                time_slice = st.slider(
                    "Time Slice",
                    0, n_samples - 1,
                    n_samples // 2,
                    key="fd_time"
                )
            
            fig = fault_detector.visualize_faults(time_slice=time_slice)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate visualization")
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create numpy file
            if st.button("Export Fault Probabilities", use_container_width=True):
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, f"fault_probabilities_{uuid.uuid4()[:8]}.npy")
                np.save(output_file, faults)
                
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download Fault Probabilities",
                        data=f.read(),
                        file_name="fault_probabilities.npy",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        with col2:
            if st.session_state.fracture_network:
                export_data = fault_detector.export_fault_network(output_format='csv')
                if export_data:
                    st.download_button(
                        label="📥 Download Fracture Network (CSV)",
                        data=export_data,
                        file_name="fracture_network.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No fracture network available for export")
    
    else:
        st.info("""
        ### Ready to Detect Faults?
        
        This system detects faults and fractures in seismic data using:
        
        **Available Methods:**
        1. **Simple Detection** - Edge detection and coherence analysis
        2. **Ensemble Detection** - Random Forest classifier with multiple attributes
        3. **Anomaly Detection** - Isolation Forest for outlier detection
        
        **Features:**
        - Interactive visualization of fault probabilities
        - Fracture network extraction
        - Exportable results for further analysis
        
        Click **"Run Fault Detection"** to begin!
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Seismic AI Processor",
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
    if 'detection_completed' not in st.session_state:
        st.session_state.detection_completed = False
    if 'faults' not in st.session_state:
        st.session_state.faults = None
    if 'fracture_network' not in st.session_state:
        st.session_state.fracture_network = None
    
    # Create tabs
    tab1, tab3 = st.tabs([
        "🎯 Bandwidth Enhancement", 
        "🤖 Fault Detection"
    ])
    
    with tab1:
        display_bandwidth_enhancement_tab(st.session_state.enhancer)
    
    with tab3:
        display_fault_detection_tab(st.session_state.enhancer)

if __name__ == "__main__":
    main()
