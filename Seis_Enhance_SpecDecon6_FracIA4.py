import streamlit as st
import numpy as np
import segyio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import os
import time
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import json
import uuid
import struct
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel, laplace, convolve
from scipy.signal import hilbert
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numba
from numba import jit, prange

# ============================================================================
# OPTIMIZED IMPORTS WITH FALLBACKS
# ============================================================================

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not available. Using fallback implementations.")
    cv2 = None

# Try to import scikit-learn with fallback
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Using simplified ML methods.")

# ============================================================================
# OPTIMIZED NUMBA FUNCTIONS
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def apply_spectral_weights_parallel(trace_fft, freqs, target_freq, enhancement_factor, 
                                   low_freq_boost, mid_freq_range):
    """Numba-accelerated spectral weighting"""
    n = len(trace_fft)
    weights = np.ones(n, dtype=np.complex128)
    freq_magnitude = np.abs(freqs)
    
    for i in prange(n):
        f = freq_magnitude[i]
        
        if f > 5 and f <= mid_freq_range[0]:
            weights[i] = low_freq_boost
        elif f > mid_freq_range[0] and f <= target_freq:
            weights[i] = enhancement_factor
        elif f > target_freq:
            rolloff = enhancement_factor * np.exp(-0.001 * (f - target_freq)**2)
            weights[i] = max(1.0, rolloff)
    
    return trace_fft * weights

@jit(nopython=True, parallel=True, cache=True)
def calculate_dip_parallel(seismic_data, dip_result):
    """Parallel dip calculation"""
    n_inlines, n_xlines, n_samples = seismic_data.shape
    
    for i in prange(1, n_inlines - 1):
        for j in prange(1, n_xlines - 1):
            for k in prange(1, n_samples - 1):
                dz = (seismic_data[i, j, k+1] - seismic_data[i, j, k-1]) / 2
                dx = (seismic_data[i+1, j, k] - seismic_data[i-1, j, k]) / 2
                
                if dx != 0:
                    dip_result[i, j, k] = np.arctan2(np.abs(dz), np.abs(dx)) * 180 / np.pi

@jit(nopython=True, parallel=True, cache=True)
def calculate_sobel_parallel(seismic_data, sobel_result):
    """Parallel Sobel edge detection"""
    n_inlines, n_xlines, n_samples = seismic_data.shape
    
    for i in prange(1, n_inlines - 1):
        for j in prange(1, n_xlines - 1):
            for k in prange(1, n_samples - 1):
                dx = (seismic_data[i+1, j, k] - seismic_data[i-1, j, k]) / 2
                dy = (seismic_data[i, j+1, k] - seismic_data[i, j-1, k]) / 2
                dz = (seismic_data[i, j, k+1] - seismic_data[i, j, k-1]) / 2
                
                sobel_result[i, j, k] = np.sqrt(dx**2 + dy**2 + dz**2)

@jit(nopython=True, parallel=True, cache=True)
def calculate_variance_parallel(seismic_data, window_size, variance_result):
    """Parallel variance calculation with sliding window"""
    n_inlines, n_xlines, n_samples = seismic_data.shape
    half_window = window_size // 2
    
    for i in prange(half_window, n_inlines - half_window):
        for j in prange(half_window, n_xlines - half_window):
            for k in prange(half_window, n_samples - half_window):
                # Extract window
                window_sum = 0.0
                window_sum_sq = 0.0
                count = 0
                
                for wi in range(-half_window, half_window + 1):
                    for wj in range(-half_window, half_window + 1):
                        for wk in range(-half_window, half_window + 1):
                            val = seismic_data[i+wi, j+wj, k+wk]
                            window_sum += val
                            window_sum_sq += val * val
                            count += 1
                
                mean = window_sum / count
                variance = (window_sum_sq / count) - (mean * mean)
                variance_result[i, j, k] = max(variance, 0)

@jit(nopython=True, cache=True)
def normalize_trace(trace):
    """Normalize trace to zero mean and unit variance"""
    mean = np.mean(trace)
    std = np.std(trace)
    if std > 1e-10:
        return (trace - mean) / std
    else:
        return trace - mean

# ============================================================================
# OPTIMIZED MAIN CLASSES
# ============================================================================

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0
        self.geometry = None
        self.original_segyfile = None
        self.original_filename = None
        self.use_multiprocessing = mp.cpu_count() > 1
        self.num_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers
        
    def read_segy_3d_fast(self, filename):
        """Fast SEG-Y reading with optimized memory usage"""
        try:
            with segyio.open(filename, "r") as segyfile:
                self.original_segyfile = segyfile
                self.original_filename = filename
                
                # Get dimensions
                n_inlines = segyfile.ilines.size
                n_xlines = segyfile.xlines.size
                n_samples = segyfile.samples.size
                
                st.success(f"3D seismic data detected: {n_inlines} inlines × {n_xlines} crosslines × {n_samples} samples")
                
                # Read data in chunks for large files
                chunk_size = 50  # Process 50 inlines at a time
                data_chunks = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for chunk_start in range(0, n_inlines, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_inlines)
                    
                    # Read chunk
                    chunk_data = np.zeros((chunk_end - chunk_start, n_xlines, n_samples))
                    for i, inline_idx in enumerate(range(chunk_start, chunk_end)):
                        for j in range(n_xlines):
                            try:
                                chunk_data[i, j, :] = segyfile.trace[inline_idx * n_xlines + j]
                            except:
                                chunk_data[i, j, :] = 0
                    
                    data_chunks.append(chunk_data)
                    
                    progress = chunk_end / n_inlines
                    progress_bar.progress(progress)
                    status_text.text(f"Reading data: {chunk_end}/{n_inlines} inlines")
                
                progress_bar.empty()
                status_text.empty()
                
                # Combine chunks
                data = np.vstack(data_chunks)
                
                self.geometry = {
                    'ilines': segyfile.ilines,
                    'xlines': segyfile.xlines,
                    'samples': segyfile.samples,
                    'tracecount': segyfile.tracecount,
                    'format': segyfile.format
                }
                
                try:
                    self.sample_rate = segyio.tools.dt(segyfile) / 1000.0
                    st.info(f"Sample rate: {self.sample_rate} ms")
                except:
                    st.info(f"Using default sample rate: {self.sample_rate} ms")
                
                st.success(f"SEG-Y file loaded with shape: {data.shape}")
                st.info(f"Data range: {np.min(data):.3f} to {np.max(data):.3f}")
                
                return data
                
        except Exception as e:
            st.error(f"Error reading SEG-Y file: {e}")
            return None

    def process_trace_parallel(self, args):
        """Process a single trace in parallel"""
        trace, target_freq, enhancement_factor, low_freq_boost, mid_freq_range, sample_rate = args
        
        if np.all(trace == 0) or np.std(trace) < 1e-10:
            return trace
            
        trace = np.nan_to_num(trace)
        trace = signal.detrend(trace)
        
        n = len(trace)
        trace_fft = fft(trace)
        freqs = fftfreq(n, d=sample_rate/1000.0)
        
        # Use Numba-accelerated function
        enhanced_fft = apply_spectral_weights_parallel(
            trace_fft, freqs, target_freq, enhancement_factor, low_freq_boost, mid_freq_range
        )
        
        enhanced_trace = np.real(ifft(enhanced_fft))
        
        if np.std(enhanced_trace) > 1e-10:
            enhanced_trace = enhanced_trace * (np.std(trace) / np.std(enhanced_trace))
        
        return enhanced_trace

    def spectral_blueing_fast(self, seismic_data, target_freq=80, enhancement_factor=1.5,
                             low_freq_boost=1.2, mid_freq_range=(30, 80)):
        """Fast spectral blueing using parallel processing"""
        st.info("Applying spectral blueing (parallel processing)...")
        
        n_inlines, n_xlines, n_samples = seismic_data.shape
        
        # Flatten data for parallel processing
        traces_flat = seismic_data.reshape(-1, n_samples)
        
        # Prepare arguments for parallel processing
        args_list = [(traces_flat[i], target_freq, enhancement_factor, 
                     low_freq_boost, mid_freq_range, self.sample_rate) 
                    for i in range(len(traces_flat))]
        
        # Use multiprocessing for large datasets
        if self.use_multiprocessing and len(traces_flat) > 100:
            st.info(f"Using {self.num_workers} CPU cores for parallel processing...")
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self.process_trace_parallel, args_list))
        else:
            # Use ThreadPoolExecutor for smaller datasets
            with ThreadPoolExecutor(max_workers=min(4, len(traces_flat))) as executor:
                results = list(executor.map(self.process_trace_parallel, args_list))
        
        # Reshape results
        enhanced_flat = np.array(results)
        enhanced_data = enhanced_flat.reshape(n_inlines, n_xlines, n_samples)
        
        return enhanced_data

    def bandpass_filter_fast(self, seismic_data, lowcut=8, highcut=120, order=3):
        """Fast bandpass filtering using vectorized operations"""
        st.info("Applying bandpass filter (vectorized)...")
        
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
        
        # Apply filter using vectorized operations
        n_inlines, n_xlines, n_samples = seismic_data.shape
        
        # Reshape for batch processing
        data_flat = seismic_data.reshape(-1, n_samples)
        enhanced_flat = np.zeros_like(data_flat)
        
        # Process in batches
        batch_size = 1000
        n_batches = (len(data_flat) + batch_size - 1) // batch_size
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data_flat))
            
            batch = data_flat[start_idx:end_idx]
            enhanced_batch = signal.filtfilt(b, a, batch, axis=1)
            enhanced_flat[start_idx:end_idx] = enhanced_batch
            
            progress = (batch_idx + 1) / n_batches
            progress_bar.progress(progress)
            status_text.text(f"Filtering batch {batch_idx + 1}/{n_batches}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Reshape back
        enhanced_data = enhanced_flat.reshape(n_inlines, n_xlines, n_samples)
        
        return enhanced_data

    def enhance_bandwidth_fast(self, file_path, target_freq=80, enhancement_factor=1.5, low_freq_boost=1.2,
                              mid_freq_start=30, lowcut=8, highcut=120, filter_order=3):
        """Fast main enhancement method"""
        st.info(f"Loading SEG-Y file (optimized)...")
        
        start_time = time.time()
        self.original_data = self.read_segy_3d_fast(file_path)
        
        if self.original_data is None:
            raise ValueError("Failed to load SEG-Y file")
        
        st.success(f"Original data shape: {self.original_data.shape}")
        st.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        # Store parameters
        self.target_freq = target_freq
        self.enhancement_factor = enhancement_factor
        self.lowcut = lowcut
        self.highcut = highcut
        
        # Apply spectral blueing
        st.info("Starting spectral blueing...")
        blueing_start = time.time()
        self.enhanced_data = self.spectral_blueing_fast(
            self.original_data,
            target_freq=target_freq,
            enhancement_factor=enhancement_factor,
            low_freq_boost=low_freq_boost,
            mid_freq_range=(mid_freq_start, target_freq)
        )
        st.info(f"Spectral blueing completed in {time.time() - blueing_start:.2f} seconds")
        
        # Apply bandpass filter
        st.info("Applying bandpass filter...")
        filter_start = time.time()
        self.enhanced_data = self.bandpass_filter_fast(
            self.enhanced_data,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order
        )
        st.info(f"Bandpass filtering completed in {time.time() - filter_start:.2f} seconds")
        
        total_time = time.time() - start_time
        st.success(f"✅ Total processing completed in {total_time:.2f} seconds")
        st.info(f"Enhanced data range: {np.min(self.enhanced_data):.3f} to {np.max(self.enhanced_data):.3f}")
        
        return self.enhanced_data

# ============================================================================
# OPTIMIZED FAULT DETECTION CLASSES
# ============================================================================

class FastSeismicAttributeCalculator:
    """Fast seismic attribute calculator with Numba acceleration"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.attributes = {}
        self.use_multiprocessing = mp.cpu_count() > 1
        self.num_workers = min(mp.cpu_count(), 4)
        
    def calculate_core_attributes_fast(self):
        """Calculate core attributes quickly using optimized methods"""
        st.info("Calculating core seismic attributes (optimized)...")
        
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        # Initialize attributes
        self.attributes['amplitude'] = self.seismic_data.copy()
        
        # Calculate amplitude envelope (fast approximation)
        st.info("Calculating amplitude envelope...")
        self.attributes['amplitude_envelope'] = np.abs(self.seismic_data)
        
        # Calculate RMS energy with sliding window (vectorized)
        st.info("Calculating RMS energy...")
        self.attributes['energy'] = self.calculate_rms_energy_fast(window_size=9)
        
        # Calculate gradient-based attributes
        st.info("Calculating gradient attributes...")
        gradient_start = time.time()
        
        # Calculate dip using Numba
        dip = np.zeros_like(self.seismic_data)
        calculate_dip_parallel(self.seismic_data, dip)
        self.attributes['dip'] = dip
        
        # Calculate Sobel edges using Numba
        sobel_edges = np.zeros_like(self.seismic_data)
        calculate_sobel_parallel(self.seismic_data, sobel_edges)
        self.attributes['sobel'] = sobel_edges
        
        st.info(f"Gradient attributes calculated in {time.time() - gradient_start:.2f} seconds")
        
        # Calculate variance (texture attribute)
        st.info("Calculating variance...")
        variance = np.zeros_like(self.seismic_data)
        calculate_variance_parallel(self.seismic_data, 5, variance)
        self.attributes['variance'] = variance
        
        # Calculate simple coherence (fast approximation)
        st.info("Calculating coherence...")
        coherence_start = time.time()
        self.attributes['coherence'] = self.calculate_coherence_fast(window_size=5)
        st.info(f"Coherence calculated in {time.time() - coherence_start:.2f} seconds")
        
        st.success(f"✅ Core attributes calculated: {len(self.attributes)} total")
        return self.attributes
    
    def calculate_rms_energy_fast(self, window_size=9):
        """Fast RMS energy calculation"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        half_window = window_size // 2
        
        # Use convolution for fast sliding window
        kernel = np.ones(window_size) / window_size
        
        energy = np.zeros_like(self.seismic_data)
        
        # Process each inline separately
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace = self.seismic_data[i, j, :]
                trace_sq = trace ** 2
                rms = np.sqrt(np.convolve(trace_sq, kernel, mode='same'))
                energy[i, j, :] = rms
        
        return energy
    
    def calculate_coherence_fast(self, window_size=5):
        """Fast coherence calculation using gradient similarity"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        # Calculate gradients
        grad_x = np.gradient(self.seismic_data, axis=0)
        grad_y = np.gradient(self.seismic_data, axis=1)
        grad_z = np.gradient(self.seismic_data, axis=2)
        
        # Normalize gradients
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-10)
        grad_x_norm = grad_x / grad_mag
        grad_y_norm = grad_y / grad_mag
        grad_z_norm = grad_z / grad_mag
        
        # Calculate coherence using vectorized operations
        coherence = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        # Use slicing for faster window operations
        for i in range(half_window, n_inlines - half_window, 2):  # Step by 2 for speed
            for j in range(half_window, n_xlines - half_window, 2):
                for k in range(half_window, n_samples - half_window, 2):
                    # Get window slices
                    i_slice = slice(i-half_window, i+half_window+1)
                    j_slice = slice(j-half_window, j+half_window+1)
                    k_slice = slice(k-half_window, k+half_window+1)
                    
                    # Calculate dot products
                    dot_products = (grad_x_norm[i_slice, j_slice, k_slice] * grad_x_norm[i, j, k] +
                                   grad_y_norm[i_slice, j_slice, k_slice] * grad_y_norm[i, j, k] +
                                   grad_z_norm[i_slice, j_slice, k_slice] * grad_z_norm[i, j, k])
                    
                    coherence[i, j, k] = np.mean(dot_products)
        
        return coherence

class FastFaultDetector:
    """Fast fault detector with optimized processing"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.fault_probabilities = None
        self.fracture_network = None
        self.attribute_calculator = FastSeismicAttributeCalculator(seismic_data, sample_rate)
        
    def detect_faults_fast(self, method='simple', threshold=0.5):
        """Fast fault detection"""
        st.info(f"Starting {method} fault detection (optimized)...")
        
        start_time = time.time()
        
        if method == 'simple':
            result = self.simple_detection_fast()
        elif method == 'ensemble' and SKLEARN_AVAILABLE:
            result = self.ensemble_detection_fast()
        else:
            result = self.simple_detection_fast()
        
        self.fault_probabilities = result
        
        # Apply threshold
        fault_binary = result > threshold
        
        # Build fracture network (simplified)
        st.info("Building fracture network...")
        network_start = time.time()
        self.fracture_network = self.build_fracture_network_fast(fault_binary)
        st.info(f"Fracture network built in {time.time() - network_start:.2f} seconds")
        
        total_time = time.time() - start_time
        st.success(f"✅ Fault detection completed in {total_time:.2f} seconds")
        
        return result, self.fracture_network
    
    def simple_detection_fast(self):
        """Fast simple fault detection"""
        # Calculate core attributes
        attributes = self.attribute_calculator.calculate_core_attributes_fast()
        
        # Get key attributes
        coherence = attributes.get('coherence', np.zeros_like(self.seismic_data))
        sobel = attributes.get('sobel', np.zeros_like(self.seismic_data))
        
        # Normalize
        def safe_normalize(data):
            if data.size == 0:
                return np.zeros_like(data)
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            return np.zeros_like(data)
        
        coherence_norm = safe_normalize(coherence)
        sobel_norm = safe_normalize(sobel)
        
        # Combine (faults have low coherence, high gradient)
        fault_prob = (1 - coherence_norm) * 0.6 + sobel_norm * 0.4
        
        return fault_prob
    
    def ensemble_detection_fast(self):
        """Fast ensemble detection with feature sampling"""
        # Calculate attributes
        attributes = self.attribute_calculator.calculate_core_attributes_fast()
        
        # Prepare feature matrix with sampling
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        # Sample the data (use every 4th voxel for speed)
        sample_step = 4
        sample_indices = np.arange(0, n_inlines * n_xlines * n_samples, sample_step)
        
        # Prepare features
        feature_names = ['coherence', 'sobel', 'dip', 'variance', 'amplitude_envelope']
        n_features = len(feature_names)
        
        # Initialize feature matrix
        n_samples_total = len(sample_indices)
        X = np.zeros((n_samples_total, n_features))
        
        # Fill feature matrix
        for idx, feature_idx in enumerate(sample_indices):
            i = idx // (n_xlines * n_samples)
            j = (idx % (n_xlines * n_samples)) // n_samples
            k = idx % n_samples
            
            for f_idx, feature_name in enumerate(feature_names):
                if feature_name in attributes:
                    X[idx, f_idx] = attributes[feature_name][i, j, k]
        
        # Generate synthetic labels
        y = self.generate_synthetic_labels_fast(X)
        
        # Train Random Forest
        if SKLEARN_AVAILABLE:
            rf = RandomForestClassifier(
                n_estimators=30,  # Reduced for speed
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Predict probabilities for all voxels (in batches)
            fault_prob = np.zeros_like(self.seismic_data)
            batch_size = 10000
            
            # Prepare full feature matrix in batches
            for i in range(0, n_inlines, 2):  # Step by 2 for speed
                for j in range(0, n_xlines, 2):
                    for k in range(0, n_samples, 2):
                        features = []
                        for feature_name in feature_names:
                            if feature_name in attributes:
                                features.append(attributes[feature_name][i, j, k])
                        
                        if len(features) == n_features:
                            features_array = np.array(features).reshape(1, -1)
                            prob = rf.predict_proba(features_array)[0, 1]
                            fault_prob[i, j, k] = prob
            
            return fault_prob
        else:
            return self.simple_detection_fast()
    
    def generate_synthetic_labels_fast(self, X):
        """Generate synthetic labels quickly"""
        # Simple rule: high gradient + low coherence = fault
        if X.shape[1] >= 2:
            # Assume column 0 is coherence, column 1 is sobel
            scores = (1 - X[:, 0]) * 0.7 + X[:, 1] * 0.3
            labels = (scores > np.percentile(scores, 70)).astype(int)
        else:
            labels = np.random.randint(0, 2, len(X))
        
        return labels
    
    def build_fracture_network_fast(self, fault_binary):
        """Fast fracture network building"""
        # Use connected components on a downsampled version
        from scipy.ndimage import label
        
        # Downsample for speed
        downsample_factor = 2
        shape = fault_binary.shape
        downsampled_shape = (shape[0]//downsample_factor, 
                           shape[1]//downsample_factor, 
                           shape[2]//downsample_factor)
        
        fault_downsampled = np.zeros(downsampled_shape, dtype=bool)
        
        for i in range(downsampled_shape[0]):
            for j in range(downsampled_shape[1]):
                for k in range(downsampled_shape[2]):
                    ii = i * downsample_factor
                    jj = j * downsample_factor
                    kk = k * downsample_factor
                    fault_downsampled[i, j, k] = np.any(
                        fault_binary[ii:ii+downsample_factor, 
                                   jj:jj+downsample_factor, 
                                   kk:kk+downsample_factor]
                    )
        
        # Label connected components
        labeled_array, num_features = label(fault_downsampled)
        
        fracture_properties = []
        for i in range(1, min(num_features + 1, 50)):  # Limit to 50 largest
            fracture_mask = labeled_array == i
            size = np.sum(fracture_mask)
            
            if size > 5:  # Minimum size
                indices = np.where(fracture_mask)
                center = [np.mean(idx) * downsample_factor for idx in indices]
                
                fracture_properties.append({
                    'id': i,
                    'center': center,
                    'size': size * (downsample_factor ** 3),
                    'voxels': size
                })
        
        return fracture_properties
    
    def visualize_faults_fast(self, inline_idx=None, crossline_idx=None, time_slice=None, colormap='hot'):
        """Fast fault visualization"""
        if self.fault_probabilities is None:
            st.error("No fault detection results available.")
            return None
        
        n_inlines, n_xlines, n_samples = self.fault_probabilities.shape
        
        # Determine slice
        if inline_idx is not None:
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
            display_data = data_slice.T
            xaxis_title = "Crossline"
            yaxis_title = "Time Sample"
            
        elif crossline_idx is not None:
            data_slice = self.fault_probabilities[:, crossline_idx, :]
            title = f"Fault Detection - Crossline {crossline_idx}"
            display_data = data_slice.T
            xaxis_title = "Inline"
            yaxis_title = "Time Sample"
            
        elif time_slice is not None:
            data_slice = self.fault_probabilities[:, :, time_slice]
            title = f"Fault Detection - Time Slice {time_slice}"
            display_data = data_slice
            xaxis_title = "Crossline"
            yaxis_title = "Inline"
            
        else:
            inline_idx = n_inlines // 2
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
            display_data = data_slice.T
            xaxis_title = "Crossline"
            yaxis_title = "Time Sample"
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=display_data,
            colorscale=colormap,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Fault Probability"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig

# ============================================================================
# OPTIMIZED HELPER FUNCTIONS
# ============================================================================

def check_memory_usage(data_size_gb=1.0):
    """Check if enough memory is available"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_memory < data_size_gb * 2:  # Need 2x data size
            st.warning(f"Low memory available: {available_memory:.1f} GB. "
                      f"Recommended: {data_size_gb * 2:.1f} GB. "
                      "Processing may be slow.")
            return False
        return True
    except:
        return True  # Continue anyway if psutil not available

def estimate_processing_time(data_shape, operation='enhancement'):
    """Estimate processing time"""
    n_voxels = np.prod(data_shape)
    
    if operation == 'enhancement':
        # Rough estimate: 0.1 microseconds per voxel per operation
        time_per_voxel = 0.1e-6
        operations = 2  # spectral blueing + filtering
        estimated_time = n_voxels * time_per_voxel * operations
        
    elif operation == 'fault_detection':
        # Rough estimate: 0.2 microseconds per voxel
        time_per_voxel = 0.2e-6
        estimated_time = n_voxels * time_per_voxel
        
    else:
        estimated_time = 0
    
    # Adjust for parallel processing
    if mp.cpu_count() > 1:
        estimated_time /= min(mp.cpu_count(), 4)
    
    return max(estimated_time, 1)  # Minimum 1 second

# ============================================================================
# PROCESSING PRESETS AND CONSTANTS
# ============================================================================

PROCESSING_PRESETS = {
    'ultra_fast': {
        'target_freq': 70,
        'enhancement_factor': 1.3,
        'low_freq_boost': 1.1,
        'mid_freq_start': 25,
        'lowcut': 10,
        'highcut': 100,
        'filter_order': 2
    },
    'balanced_fast': {
        'target_freq': 80,
        'enhancement_factor': 1.5,
        'low_freq_boost': 1.2,
        'mid_freq_start': 30,
        'lowcut': 8,
        'highcut': 120,
        'filter_order': 3
    },
    'high_quality': {
        'target_freq': 90,
        'enhancement_factor': 1.8,
        'low_freq_boost': 1.3,
        'mid_freq_start': 20,
        'lowcut': 5,
        'highcut': 150,
        'filter_order': 4
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
    'simple': 'Simple (Fast)',
    'ensemble': 'Ensemble (ML - if available)'
}

# ============================================================================
# OPTIMIZED TAB FUNCTIONS
# ============================================================================

def display_bandwidth_enhancement_tab():
    """Display optimized bandwidth enhancement tab"""
    st.title("🚀 Fast 3D Seismic Bandwidth Enhancement")
    
    # Initialize enhancer if not exists
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = SeismicBandwidthEnhancer()
    
    enhancer = st.session_state.enhancer
    
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format",
        key="bw_uploader"
    )
    
    st.sidebar.header("⚙️ Processing Mode")
    processing_mode = st.sidebar.selectbox(
        "Processing Speed",
        ["Ultra Fast", "Balanced", "High Quality"],
        index=1,
        help="Trade-off between speed and quality"
    )
    
    # Map mode to preset
    preset_map = {
        "Ultra Fast": "ultra_fast",
        "Balanced": "balanced_fast", 
        "High Quality": "high_quality"
    }
    
    preset_name = preset_map[processing_mode]
    preset_params = PROCESSING_PRESETS[preset_name]
    
    st.sidebar.header("🎨 Visualization")
    amplitude_colormap = st.sidebar.selectbox("Colormap", COLORMAPS, index=0, key="bw_cmap")
    
    if uploaded_file is not None:
        # Store uploaded file
        if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sgy') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_filename = tmp_file.name
                st.session_state.temp_filename = temp_filename
            
            st.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Preview File Info", use_container_width=True):
                try:
                    with segyio.open(st.session_state.temp_filename, "r") as segyfile:
                        st.info(f"Traces: {segyfile.tracecount:,}")
                        st.info(f"Samples per trace: {len(segyfile.samples)}")
                        if hasattr(segyfile, 'ilines'):
                            st.info(f"Inlines: {segyfile.ilines.size}")
                            st.info(f"Crosslines: {segyfile.xlines.size}")
                except:
                    st.error("Could not read SEG-Y file info")
        
        # Processing button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Process Data", type="primary", use_container_width=True, key="process_btn"):
                # Clear previous results
                for key in ['data_processed', 'file_generated', 'enhanced_file_path', 
                           'detection_completed', 'faults', 'fracture_network']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Start processing
                with st.spinner(f"Processing with {processing_mode} mode..."):
                    try:
                        # Estimate time
                        data_shape = enhancer.read_segy_3d_fast(st.session_state.temp_filename).shape
                        est_time = estimate_processing_time(data_shape, 'enhancement')
                        st.info(f"Estimated processing time: {est_time:.0f} seconds")
                        
                        # Process
                        enhanced_data = enhancer.enhance_bandwidth_fast(
                            st.session_state.temp_filename,
                            **preset_params
                        )
                        
                        st.session_state.data_processed = True
                        st.session_state.enhanced_data = enhanced_data
                        st.success("✅ Processing completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
        
        # Display results if processed
        if st.session_state.get('data_processed', False):
            st.header("📈 Processing Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Data")
                if enhancer.original_data is not None:
                    st.write(f"Shape: {enhancer.original_data.shape}")
                    st.write(f"Min: {np.min(enhancer.original_data):.3f}")
                    st.write(f"Max: {np.max(enhancer.original_data):.3f}")
                    st.write(f"Mean: {np.mean(enhancer.original_data):.3f}")
            
            with col2:
                st.subheader("Enhanced Data")
                if enhancer.enhanced_data is not None:
                    st.write(f"Shape: {enhancer.enhanced_data.shape}")
                    st.write(f"Min: {np.min(enhancer.enhanced_data):.3f}")
                    st.write(f"Max: {np.max(enhancer.enhanced_data):.3f}")
                    st.write(f"Mean: {np.mean(enhancer.enhanced_data):.3f}")
            
            # Quick visualization
            st.subheader("📊 Quick Preview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                view_type = st.selectbox("View Type", ["Inline", "Crossline", "Time Slice"], key="preview_view")
            
            if enhancer.original_data is not None:
                n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                
                if view_type == "Inline":
                    with col2:
                        inline_idx = st.slider("Inline", 0, n_inlines-1, n_inlines//2, key="preview_inline")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_orig = px.imshow(enhancer.original_data[inline_idx, :, :].T,
                                           title=f"Original Inline {inline_idx}",
                                           color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        fig_enh = px.imshow(enhancer.enhanced_data[inline_idx, :, :].T,
                                          title=f"Enhanced Inline {inline_idx}",
                                          color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                elif view_type == "Crossline":
                    with col2:
                        xline_idx = st.slider("Crossline", 0, n_xlines-1, n_xlines//2, key="preview_xline")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_orig = px.imshow(enhancer.original_data[:, xline_idx, :].T,
                                           title=f"Original Crossline {xline_idx}",
                                           color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        fig_enh = px.imshow(enhancer.enhanced_data[:, xline_idx, :].T,
                                          title=f"Enhanced Crossline {xline_idx}",
                                          color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                else:  # Time Slice
                    with col2:
                        time_idx = st.slider("Time Slice", 0, n_samples-1, n_samples//2, key="preview_time")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_orig = px.imshow(enhancer.original_data[:, :, time_idx],
                                           title=f"Original Time Slice {time_idx}",
                                           color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        fig_enh = px.imshow(enhancer.enhanced_data[:, :, time_idx],
                                          title=f"Enhanced Time Slice {time_idx}",
                                          color_continuous_scale=amplitude_colormap)
                        st.plotly_chart(fig_enh, use_container_width=True)
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Downloadable File", use_container_width=True):
                    with st.spinner("Creating file..."):
                        temp_dir = tempfile.gettempdir()
                        unique_id = str(uuid.uuid4())[:8]
                        output_file = os.path.join(temp_dir, f"enhanced_{unique_id}.npy")
                        
                        # Save as numpy for speed
                        np.save(output_file, enhancer.enhanced_data)
                        
                        # Save metadata
                        metadata = {
                            'original_shape': enhancer.original_data.shape if enhancer.original_data is not None else None,
                            'sample_rate': enhancer.sample_rate,
                            'processing_preset': preset_name,
                            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'parameters': preset_params
                        }
                        
                        metadata_file = os.path.join(temp_dir, f"metadata_{unique_id}.json")
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        st.session_state.enhanced_file_path = output_file
                        st.session_state.metadata_file_path = metadata_file
                        st.session_state.file_generated = True
                        st.success("File generated successfully!")
            
            if st.session_state.get('file_generated', False):
                with col2:
                    # Download enhanced data
                    with open(st.session_state.enhanced_file_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Enhanced Data",
                            data=f.read(),
                            file_name="enhanced_seismic.npy",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                
                # Download metadata
                with open(st.session_state.metadata_file_path, 'rb') as f:
                    st.download_button(
                        label="📄 Download Metadata",
                        data=f.read(),
                        file_name="processing_metadata.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    else:
        # Welcome screen
        st.info("👈 Please upload a 3D SEG-Y file to begin processing")
        
        st.header("🚀 Performance Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ⚡ Fast Processing")
            st.markdown("""
            - Parallel processing
            - Numba acceleration
            - Vectorized operations
            - Memory efficient
            """)
        
        with col2:
            st.markdown("### 🎯 Smart Optimization")
            st.markdown("""
            - Adaptive chunking
            - Progress tracking
            - Memory monitoring
            - Time estimation
            """)
        
        with col3:
            st.markdown("### 📊 Quick Visualization")
            st.markdown("""
            - Interactive plots
            - Multiple views
            - Real-time preview
            - Export options
            """)

def display_fault_detection_tab():
    """Display optimized fault detection tab"""
    st.header("🔍 Fast Fault & Fracture Detection")
    
    # Check if data is processed
    if not st.session_state.get('data_processed', False) or 'enhancer' not in st.session_state:
        st.info("⚠️ Please process seismic data in the Bandwidth Enhancement tab first.")
        st.warning("No enhanced data available. Run bandwidth enhancement first.")
        return
    
    enhancer = st.session_state.enhancer
    
    st.sidebar.header("⚙️ Detection Parameters")
    
    # Method selection
    available_methods = ['simple']
    if SKLEARN_AVAILABLE:
        available_methods.append('ensemble')
    
    method = st.sidebar.selectbox(
        "Detection Method",
        options=available_methods,
        format_func=lambda x: FAULT_DETECTION_METHODS.get(x, x.title()),
        key="fd_method"
    )
    
    # Threshold
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key="fd_threshold"
    )
    
    st.sidebar.header("🎨 Visualization")
    fault_colormap = st.sidebar.selectbox(
        "Colormap",
        COLORMAPS,
        index=6,  # 'hot' colormap
        key="fd_cmap"
    )
    
    # Main detection section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Run Fault Detection", type="primary", use_container_width=True, key="fd_run"):
            # Clear previous results
            for key in ['detection_completed', 'faults', 'fracture_network', 'fault_detector']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Start detection
            with st.spinner(f"Running {FAULT_DETECTION_METHODS[method]} detection..."):
                try:
                    # Initialize detector
                    detector = FastFaultDetector(enhancer.enhanced_data, enhancer.sample_rate)
                    
                    # Estimate time
                    data_shape = enhancer.enhanced_data.shape
                    est_time = estimate_processing_time(data_shape, 'fault_detection')
                    st.info(f"Estimated detection time: {est_time:.0f} seconds")
                    
                    # Run detection
                    faults, fracture_network = detector.detect_faults_fast(method=method, threshold=threshold)
                    
                    # Store results
                    st.session_state.fault_detector = detector
                    st.session_state.faults = faults
                    st.session_state.fracture_network = fracture_network
                    st.session_state.detection_completed = True
                    
                    st.success("✅ Fault detection completed!")
                    
                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")
    
    # Display results if available
    if st.session_state.get('detection_completed', False):
        detector = st.session_state.fault_detector
        faults = st.session_state.faults
        
        st.subheader("📈 Detection Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_voxels = np.prod(faults.shape)
            st.metric("Total Voxels", f"{total_voxels:,}")
        
        with col2:
            fault_voxels = np.sum(faults > threshold)
            fault_percentage = (fault_voxels / total_voxels) * 100
            st.metric("Fault Voxels", f"{fault_voxels:,} ({fault_percentage:.1f}%)")
        
        with col3:
            avg_prob = np.mean(faults)
            st.metric("Avg Probability", f"{avg_prob:.3f}")
        
        with col4:
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
            
            fig = detector.visualize_faults_fast(inline_idx=inline_idx, colormap=fault_colormap)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif view_type == "Crossline":
            with col2:
                crossline_idx = st.slider(
                    "Crossline Index",
                    0, n_xlines - 1,
                    n_xlines // 2,
                    key="fd_crossline"
                )
            
            fig = detector.visualize_faults_fast(crossline_idx=crossline_idx, colormap=fault_colormap)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Time Slice
            with col2:
                time_slice = st.slider(
                    "Time Slice",
                    0, n_samples - 1,
                    n_samples // 2,
                    key="fd_time"
                )
            
            fig = detector.visualize_faults_fast(time_slice=time_slice, colormap=fault_colormap)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Export section
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export fault probabilities
            if st.button("Export Fault Probabilities", use_container_width=True):
                temp_dir = tempfile.gettempdir()
                unique_id = str(uuid.uuid4())[:8]
                output_file = os.path.join(temp_dir, f"fault_probabilities_{unique_id}.npy")
                
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
            # Export fracture network
            if st.session_state.fracture_network:
                if st.button("Export Fracture Network", use_container_width=True):
                    # Convert to DataFrame
                    df_data = []
                    for fracture in st.session_state.fracture_network:
                        df_data.append({
                            'ID': fracture['id'],
                            'Center_Inline': fracture['center'][0],
                            'Center_Crossline': fracture['center'][1],
                            'Center_Time': fracture['center'][2],
                            'Size_Voxels': fracture['size'],
                            'Component_Voxels': fracture['voxels']
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv_data = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Fracture Network (CSV)",
                        data=csv_data,
                        file_name="fracture_network.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No fracture network to export")
    
    else:
        # Instructions
        st.info("""
        ### ⚡ Fast Fault Detection
        
        **Available Methods:**
        1. **Simple (Fast)** - Gradient-based detection (always available)
        2. **Ensemble (ML)** - Random Forest classifier (if scikit-learn available)
        
        **Performance Features:**
        - Parallel processing with Numba acceleration
        - Optimized attribute calculations
        - Memory-efficient algorithms
        - Real-time progress tracking
        
        **Click "Run Fault Detection" to begin!**
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Seismic AI Processor - FAST",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better performance
    st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'file_generated' not in st.session_state:
        st.session_state.file_generated = False
    
    # Header
    st.sidebar.title("⚡ Fast Seismic Processor")
    st.sidebar.markdown("---")
    
    # CPU info
    cpu_count = mp.cpu_count()
    st.sidebar.info(f"Available CPU cores: {cpu_count}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🚀 Bandwidth Enhancement", "🔍 Fault Detection"])
    
    with tab1:
        display_bandwidth_enhancement_tab()
    
    with tab2:
        display_fault_detection_tab()

if __name__ == "__main__":
    # Check for required packages
    try:
        import numba
        st.set_option('deprecation.showPyplotGlobalUse', False)
        main()
    except ImportError as e:
        st.error(f"Missing required package: {e}")
        st.info("Please install required packages: pip install numba scipy numpy segyio plotly")
