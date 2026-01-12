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

# Machine Learning/AI imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
import lightgbm as lgb
import joblib
import umap
import hdbscan

# Image processing imports
import cv2
from skimage import feature, filters, morphology, segmentation
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon
import pywt

# Deep learning specific
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

    def write_segy_copy_replace(self, output_filename):
        """Copy original file and replace trace data only"""
        try:
            import shutil
            # Copy original file
            shutil.copy2(self.original_filename, output_filename)
            
            # Open copied file and replace trace data
            with segyio.open(output_filename, "r+") as dst:
                n_traces = dst.tracecount
                enhanced_flat = self.enhanced_data.reshape(-1, self.enhanced_data.shape[-1])
                
                for i in range(min(n_traces, len(enhanced_flat))):
                    trace_data = enhanced_flat[i].astype(np.float32)
                    dst.trace[i] = trace_data
            
            st.success(f"Copy-replace SEG-Y writing completed: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"Copy-replace writing failed: {e}")
            return False

    def create_numpy_alternative(self, output_filename):
        """Create a numpy-based alternative format with metadata"""
        try:
            # Create a comprehensive metadata dictionary
            metadata = {
                'original_shape': self.original_data.shape if self.original_data is not None else None,
                'enhanced_shape': self.enhanced_data.shape,
                'sample_rate': self.sample_rate,
                'data_type': 'enhanced_seismic_3d',
                'description': 'Enhanced 3D seismic data created by Seismic Bandwidth Enhancer',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_parameters': {
                    'target_frequency': getattr(self, 'target_freq', 'unknown'),
                    'enhancement_factor': getattr(self, 'enhancement_factor', 'unknown'),
                    'bandpass_range': f"{getattr(self, 'lowcut', 'unknown')}-{getattr(self, 'highcut', 'unknown')} Hz"
                }
            }
            
            # Create the file
            with open(output_filename, 'wb') as f:
                # Write metadata as JSON
                metadata_json = json.dumps(metadata, indent=2)
                f.write(metadata_json.encode('utf-8'))
                f.write(b'\n' + b'END_METADATA' + b'\n')
                
                # Write the enhanced data
                self.enhanced_data.astype(np.float32).tofile(f)
            
            st.success(f"Created numpy alternative file: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"Numpy alternative creation failed: {e}")
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
            
            # Try methods in order of reliability
            methods = [
                ("Numpy-based SEG-Y", self.write_segy_numpy_based),
                ("Copy-replace", self.write_segy_copy_replace),
            ]
            
            success = False
            for method_name, method_func in methods:
                st.info(f"Trying {method_name}...")
                success = method_func(download_filename)
                if success:
                    st.success(f"{method_name} succeeded!")
                    break
                else:
                    st.warning(f"{method_name} failed")
            
            if not success:
                # Fallback to numpy alternative format
                st.info("SEG-Y methods failed, creating numpy alternative format...")
                alt_filename = os.path.join(temp_dir, f"enhanced_{unique_id}.dat")
                success = self.create_numpy_alternative(alt_filename)
                if success:
                    download_filename = alt_filename
                    st.warning("Created numpy alternative format (.dat) instead of SEG-Y")
            
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
                st.error("All file creation methods failed")
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

class SpectralDecomposition:
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.frequency_data = None
        
    def compute_spectral_decomposition(self, frequencies, wavelet_type='FFT', **wavelet_params):
        """Compute spectral decomposition for given frequencies using specified wavelet"""
        st.info(f"Computing spectral decomposition for {len(frequencies)} frequencies using {wavelet_type}...")
        
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        n_freqs = len(frequencies)
        
        # Initialize frequency data array
        self.frequency_data = np.zeros((n_inlines, n_xlines, n_samples, n_freqs))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for freq_idx, target_freq in enumerate(frequencies):
            status_text.text(f"Processing frequency {target_freq} Hz ({freq_idx+1}/{n_freqs}) using {wavelet_type}")
            
            for i in range(n_inlines):
                for j in range(n_xlines):
                    trace = self.seismic_data[i, j, :]
                    # Apply specified wavelet transform
                    if wavelet_type == 'FFT':
                        filtered_trace = self.fft_spectral_filter(trace, target_freq)
                    elif wavelet_type == 'Morlet':
                        filtered_trace = self.morlet_wavelet_transform(trace, target_freq, **wavelet_params)
                    elif wavelet_type == 'Ricker':
                        filtered_trace = self.ricker_wavelet_transform(trace, target_freq, **wavelet_params)
                    elif wavelet_type == 'CWT':
                        filtered_trace = self.cwt_transform(trace, target_freq, **wavelet_params)
                    else:
                        # Default to FFT
                        filtered_trace = self.fft_spectral_filter(trace, target_freq)
                    
                    # Ensure the filtered trace has the correct length
                    if len(filtered_trace) != n_samples:
                        if len(filtered_trace) > n_samples:
                            filtered_trace = filtered_trace[:n_samples]
                        else:
                            filtered_trace = np.pad(filtered_trace, (0, n_samples - len(filtered_trace)), mode='constant')
                    
                    self.frequency_data[i, j, :, freq_idx] = np.abs(filtered_trace)
            
            progress_bar.progress((freq_idx + 1) / n_freqs)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Spectral decomposition completed using {wavelet_type}!")
        return self.frequency_data
    
    def fft_spectral_filter(self, trace, center_freq, window_length=100):
        """Apply FFT-based spectral filter (original method)"""
        # Create Gaussian filter in frequency domain
        n_samples = len(trace)
        freqs = fftfreq(n_samples, d=self.sample_rate/1000.0)
        
        # Create Gaussian window centered at target frequency
        sigma = center_freq * 0.2  # Bandwidth proportional to center frequency
        gaussian_window = np.exp(-0.5 * ((np.abs(freqs) - center_freq) / sigma) ** 2)
        
        # Apply filter
        trace_fft = fft(trace)
        filtered_fft = trace_fft * gaussian_window
        filtered_trace = np.real(ifft(filtered_fft))
        
        return filtered_trace
    
    def morlet_wavelet_transform(self, trace, center_freq, cycles=6):
        """Apply Morlet wavelet transform with proper convolution handling"""
        n_samples = len(trace)
        dt = self.sample_rate / 1000.0
        
        # Create Morlet wavelet with proper time vector
        wavelet_length = min(int(cycles / center_freq / dt), n_samples // 2)
        if wavelet_length % 2 == 0:
            wavelet_length += 1  # Ensure odd length
            
        t = np.linspace(-wavelet_length//2 * dt, wavelet_length//2 * dt, wavelet_length)
        
        # Morlet wavelet formula (real part)
        sigma = cycles / (2 * np.pi * center_freq)
        wavelet = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * center_freq * t)
        
        # Normalize wavelet
        wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
        
        # Apply convolution with 'same' mode to maintain length
        filtered_trace = np.convolve(trace, wavelet, mode='same')
        
        return filtered_trace
    
    def ricker_wavelet_transform(self, trace, center_freq):
        """Apply Ricker (Mexican Hat) wavelet transform"""
        n_samples = len(trace)
        dt = self.sample_rate / 1000.0
        
        # Create Ricker wavelet with proper length
        wavelet_length = min(int(1.0 / center_freq / dt * 10), n_samples // 2)
        if wavelet_length % 2 == 0:
            wavelet_length += 1  # Ensure odd length
            
        t = np.linspace(-wavelet_length//2 * dt, wavelet_length//2 * dt, wavelet_length)
        
        # Ricker wavelet formula (second derivative of Gaussian)
        sigma = 1.0 / (np.pi * center_freq * np.sqrt(2))
        t_sq = t**2
        wavelet = (1.0 - (t_sq / sigma**2)) * np.exp(-t_sq / (2 * sigma**2))
        
        # Normalize wavelet
        wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
        
        # Apply convolution
        filtered_trace = np.convolve(trace, wavelet, mode='same')
        
        return filtered_trace
    
    def cwt_transform(self, trace, center_freq, scales_factor=1.0):
        """Apply Continuous Wavelet Transform using Ricker wavelet"""
        try:
            n_samples = len(trace)
            dt = self.sample_rate / 1000.0
            
            # Define scales for CWT - convert frequency to scale
            # For Ricker wavelet: scale ≈ (center_freq * dt) / scale_factor
            scale = (center_freq * dt) / scales_factor
            
            # Generate Ricker wavelet at the specified scale
            wavelet_length = min(int(10 * scale / dt), n_samples // 2)
            if wavelet_length % 2 == 0:
                wavelet_length += 1
                
            t = np.linspace(-wavelet_length//2 * dt, wavelet_length//2 * dt, wavelet_length)
            
            # Ricker wavelet
            wavelet = (2/(np.sqrt(3*scale)*np.pi**0.25)) * (1 - (t/scale)**2) * np.exp(-(t**2)/(2*scale**2))
            
            # Apply convolution
            filtered_trace = np.convolve(trace, wavelet, mode='same')
            
            return filtered_trace
            
        except Exception as e:
            st.warning(f"CWT failed: {e}. Using FFT fallback.")
            return self.fft_spectral_filter(trace, center_freq)
    
    def get_frequency_slice(self, frequency_idx, time_slice):
        """Get amplitude slice for specific frequency and time"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed. Run compute_spectral_decomposition first.")
        
        return self.frequency_data[:, :, time_slice, frequency_idx]
    
    def get_frequency_inline(self, frequency_idx, inline_idx):
        """Get inline section for specific frequency"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed.")
        
        return self.frequency_data[inline_idx, :, :, frequency_idx]
    
    def get_frequency_crossline(self, frequency_idx, crossline_idx):
        """Get crossline section for specific frequency"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed.")
        
        return self.frequency_data[:, crossline_idx, :, frequency_idx]
    
    def create_rgb_blend(self, low_freq_idx, mid_freq_idx, high_freq_idx, time_slice, weights=(1.0, 1.0, 1.0)):
        """Create RGB blend from three frequency components"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed.")
        
        # Extract the three frequency components
        low_freq = self.frequency_data[:, :, time_slice, low_freq_idx]
        mid_freq = self.frequency_data[:, :, time_slice, mid_freq_idx]
        high_freq = self.frequency_data[:, :, time_slice, high_freq_idx]
        
        # Normalize each component
        low_freq_norm = self.normalize_data(low_freq) * weights[0]
        mid_freq_norm = self.normalize_data(mid_freq) * weights[1]
        high_freq_norm = self.normalize_data(high_freq) * weights[2]
        
        # Create RGB image
        rgb_image = np.stack([low_freq_norm, mid_freq_norm, high_freq_norm], axis=-1)
        
        return rgb_image
    
    def create_rgb_inline(self, low_freq_idx, mid_freq_idx, high_freq_idx, inline_idx, weights=(1.0, 1.0, 1.0)):
        """Create RGB blend for inline section"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed.")
        
        # Extract the three frequency components for inline
        low_freq = self.frequency_data[inline_idx, :, :, low_freq_idx]
        mid_freq = self.frequency_data[inline_idx, :, :, mid_freq_idx]
        high_freq = self.frequency_data[inline_idx, :, :, high_freq_idx]
        
        # Normalize each component
        low_freq_norm = self.normalize_data(low_freq) * weights[0]
        mid_freq_norm = self.normalize_data(mid_freq) * weights[1]
        high_freq_norm = self.normalize_data(high_freq) * weights[2]
        
        # Create RGB image
        rgb_image = np.stack([low_freq_norm, mid_freq_norm, high_freq_norm], axis=-1)
        
        return rgb_image
    
    def create_rgb_crossline(self, low_freq_idx, mid_freq_idx, high_freq_idx, crossline_idx, weights=(1.0, 1.0, 1.0)):
        """Create RGB blend for crossline section"""
        if self.frequency_data is None:
            raise ValueError("Spectral decomposition not computed.")
        
        # Extract the three frequency components for crossline
        low_freq = self.frequency_data[:, crossline_idx, :, low_freq_idx]
        mid_freq = self.frequency_data[:, crossline_idx, :, mid_freq_idx]
        high_freq = self.frequency_data[:, crossline_idx, :, high_freq_idx]
        
        # Normalize each component
        low_freq_norm = self.normalize_data(low_freq) * weights[0]
        mid_freq_norm = self.normalize_data(mid_freq) * weights[1]
        high_freq_norm = self.normalize_data(high_freq) * weights[2]
        
        # Create RGB image
        rgb_image = np.stack([low_freq_norm, mid_freq_norm, high_freq_norm], axis=-1)
        
        return rgb_image
    
    def normalize_data(self, data):
        """Normalize data to 0-1 range"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 0:
            return (data - data_min) / (data_max - data_min)
        else:
            return np.zeros_like(data)

# ============================================================================
# ADVANCED FAULT AND FRACTURE DETECTION MODULES
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
        total_voxels = n_inlines * n_xlines * n_samples
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed = 0
        
        # 1. Basic Attributes
        status_text.text("Calculating basic attributes...")
        self.attributes['amplitude'] = self.seismic_data.copy()
        self.attributes['energy'] = self.calculate_energy(window_size)
        processed += 1
        progress_bar.progress(processed / 15)
        
        # 2. Structural Attributes
        status_text.text("Calculating structural attributes...")
        self.attributes['dip'] = self.calculate_dip(gradient_kernel_size)
        self.attributes['azimuth'] = self.calculate_azimuth(gradient_kernel_size)
        self.attributes['curvature'] = self.calculate_curvature(gradient_kernel_size)
        self.attributes['coherence'] = self.calculate_coherence(window_size)
        processed += 4
        progress_bar.progress(processed / 15)
        
        # 3. Texture Attributes
        status_text.text("Calculating texture attributes...")
        self.attributes['variance'] = self.calculate_variance(window_size)
        self.attributes['entropy'] = self.calculate_entropy(window_size)
        self.attributes['homogeneity'] = self.calculate_homogeneity(window_size)
        self.attributes['contrast'] = self.calculate_contrast(window_size)
        processed += 4
        progress_bar.progress(processed / 15)
        
        # 4. Edge Detection Attributes
        status_text.text("Calculating edge detection attributes...")
        self.attributes['sobel'] = self.calculate_sobel_edge()
        self.attributes['laplacian'] = self.calculate_laplacian()
        self.attributes['canny'] = self.calculate_canny_edge()
        processed += 3
        progress_bar.progress(processed / 15)
        
        # 5. Advanced Attributes
        status_text.text("Calculating advanced attributes...")
        self.attributes['instantaneous_phase'] = self.calculate_instantaneous_phase()
        self.attributes['instantaneous_frequency'] = self.calculate_instantaneous_frequency()
        self.attributes['sweetness'] = self.calculate_sweetness()
        processed += 3
        progress_bar.progress(processed / 15)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Calculated {len(self.attributes)} seismic attributes")
        
        return self.attributes
    
    def calculate_energy(self, window_size):
        """Calculate energy attribute (RMS amplitude)"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        energy = np.zeros_like(self.seismic_data)
        
        half_window = window_size // 2
        
        for i in range(half_window, n_inlines - half_window):
            for j in range(half_window, n_xlines - half_window):
                for k in range(half_window, n_samples - half_window):
                    window = self.seismic_data[i-half_window:i+half_window+1,
                                             j-half_window:j+half_window+1,
                                             k-half_window:k+half_window+1]
                    energy[i, j, k] = np.sqrt(np.mean(window**2))
        
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
        # Second derivatives
        gradient_x = np.gradient(self.seismic_data, axis=0)
        gradient_y = np.gradient(self.seismic_data, axis=1)
        gradient_z = np.gradient(self.seismic_data, axis=2)
        
        gradient_xx = np.gradient(gradient_x, axis=0)
        gradient_yy = np.gradient(gradient_y, axis=1)
        gradient_zz = np.gradient(gradient_z, axis=2)
        
        # Curvature calculation
        numerator = (gradient_xx * gradient_z**2 + gradient_yy * gradient_z**2 + 
                    gradient_zz * (gradient_x**2 + gradient_y**2) -
                    2 * gradient_x * gradient_z * np.gradient(gradient_x, axis=2) -
                    2 * gradient_y * gradient_z * np.gradient(gradient_y, axis=2))
        
        denominator = (gradient_x**2 + gradient_y**2 + gradient_z**2)**1.5
        curvature = np.where(denominator != 0, numerator / denominator, 0)
        
        return curvature
    
    def calculate_coherence(self, window_size=9):
        """Calculate seismic coherence (similarity)"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        coherence = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        for i in range(half_window, n_inlines - half_window):
            for j in range(half_window, n_xlines - half_window):
                for k in range(half_window, n_samples - half_window):
                    window = self.seismic_data[i-half_window:i+half_window+1,
                                             j-half_window:j+half_window+1,
                                             k-half_window:k+half_window+1]
                    # Calculate coherence using eigenvalue decomposition
                    window_flat = window.reshape(-1, window_size)
                    cov_matrix = np.cov(window_flat, rowvar=False)
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    coherence[i, j, k] = eigenvalues[-1] / np.sum(eigenvalues)
        
        return coherence
    
    def calculate_variance(self, window_size):
        """Calculate variance attribute"""
        return self._apply_window_operation(np.var, window_size)
    
    def calculate_entropy(self, window_size):
        """Calculate entropy attribute"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        entropy = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        for i in range(half_window, n_inlines - half_window):
            for j in range(half_window, n_xlines - half_window):
                for k in range(half_window, n_samples - half_window):
                    window = self.seismic_data[i-half_window:i+half_window+1,
                                             j-half_window:j+half_window+1,
                                             k-half_window:k+half_window+1]
                    hist, _ = np.histogram(window, bins=32, density=True)
                    hist = hist[hist > 0]
                    entropy[i, j, k] = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def calculate_homogeneity(self, window_size):
        """Calculate homogeneity attribute"""
        return self._apply_window_operation(lambda x: 1.0 / (1.0 + np.var(x)), window_size)
    
    def calculate_contrast(self, window_size):
        """Calculate contrast attribute"""
        return self._apply_window_operation(lambda x: np.max(x) - np.min(x), window_size)
    
    def _apply_window_operation(self, operation, window_size):
        """Helper function to apply window-based operations"""
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        result = np.zeros_like(self.seismic_data)
        half_window = window_size // 2
        
        for i in range(half_window, n_inlines - half_window):
            for j in range(half_window, n_xlines - half_window):
                for k in range(half_window, n_samples - half_window):
                    window = self.seismic_data[i-half_window:i+half_window+1,
                                             j-half_window:j+half_window+1,
                                             k-half_window:k+half_window+1]
                    result[i, j, k] = operation(window)
        
        return result
    
    def calculate_sobel_edge(self):
        """Calculate Sobel edge detection"""
        sobel_x = np.zeros_like(self.seismic_data)
        sobel_y = np.zeros_like(self.seismic_data)
        sobel_z = np.zeros_like(self.seismic_data)
        
        # 3D Sobel kernels
        kernel_x = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
        
        kernel_y = np.transpose(kernel_x, (1, 0, 2))
        kernel_z = np.transpose(kernel_x, (2, 1, 0))
        
        # Apply convolution
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        for i in range(1, n_inlines - 1):
            for j in range(1, n_xlines - 1):
                for k in range(1, n_samples - 1):
                    window = self.seismic_data[i-1:i+2, j-1:j+2, k-1:k+2]
                    sobel_x[i, j, k] = np.sum(window * kernel_x)
                    sobel_y[i, j, k] = np.sum(window * kernel_y)
                    sobel_z[i, j, k] = np.sum(window * kernel_z)
        
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)
        return sobel_magnitude
    
    def calculate_laplacian(self):
        """Calculate Laplacian edge detection"""
        laplacian = np.zeros_like(self.seismic_data)
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        
        # 3D Laplacian kernel
        kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        
        for i in range(1, n_inlines - 1):
            for j in range(1, n_xlines - 1):
                for k in range(1, n_samples - 1):
                    window = self.seismic_data[i-1:i+2, j-1:j+2, k-1:k+2]
                    laplacian[i, j, k] = np.sum(window * kernel)
        
        return laplacian
    
    def calculate_canny_edge(self):
        """Calculate Canny edge detection"""
        # Simplified Canny for 3D
        smoothed = self._gaussian_smooth(self.seismic_data, sigma=1.0)
        gradient_mag = self.calculate_sobel_edge()
        
        # Non-maximum suppression
        edges = np.zeros_like(gradient_mag)
        n_inlines, n_xlines, n_samples = gradient_mag.shape
        
        for i in range(1, n_inlines - 1):
            for j in range(1, n_xlines - 1):
                for k in range(1, n_samples - 1):
                    if gradient_mag[i, j, k] > gradient_mag[i-1:i+2, j-1:j+2, k-1:k+2].max():
                        edges[i, j, k] = gradient_mag[i, j, k]
        
        return edges
    
    def _gaussian_smooth(self, data, sigma=1.0):
        """Apply Gaussian smoothing"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma=sigma)
    
    def calculate_instantaneous_phase(self):
        """Calculate instantaneous phase using Hilbert transform"""
        analytic_signal = signal.hilbert(self.seismic_data, axis=2)
        instantaneous_phase = np.angle(analytic_signal)
        return instantaneous_phase
    
    def calculate_instantaneous_frequency(self):
        """Calculate instantaneous frequency"""
        instantaneous_phase = self.calculate_instantaneous_phase()
        dt = self.sample_rate / 1000.0
        instantaneous_frequency = np.gradient(instantaneous_phase, axis=2) / (2 * np.pi * dt)
        return instantaneous_frequency
    
    def calculate_sweetness(self):
        """Calculate sweetness attribute (amplitude/frequency)"""
        amplitude = np.abs(self.seismic_data)
        frequency = self.calculate_instantaneous_frequency()
        frequency = np.where(frequency > 0, frequency, 1e-10)  # Avoid division by zero
        sweetness = amplitude / frequency
        return sweetness

class FaultDetectionModel(nn.Module):
    """Deep learning model for fault detection"""
    
    def __init__(self, input_channels=15, base_filters=32):
        super().__init__()
        
        # Encoder
        self.encoder1 = self._conv_block(input_channels, base_filters)
        self.encoder2 = self._conv_block(base_filters, base_filters * 2)
        self.encoder3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.encoder4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Bridge
        self.bridge = self._conv_block(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.decoder4 = self._upconv_block(base_filters * 16, base_filters * 8)
        self.decoder3 = self._upconv_block(base_filters * 8, base_filters * 4)
        self.decoder2 = self._upconv_block(base_filters * 4, base_filters * 2)
        self.decoder1 = self._upconv_block(base_filters * 2, base_filters)
        
        # Output
        self.output = nn.Conv3d(base_filters, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bridge
        b = self.bridge(e4)
        
        # Decoder with skip connections
        d4 = self.decoder4(b) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1
        
        # Output
        output = self.output(d1)
        return self.sigmoid(output)

class AdvancedFaultDetector:
    """Advanced fault and fracture detection using ensemble of AI methods"""
    
    def __init__(self, seismic_data, sample_rate):
        self.seismic_data = seismic_data
        self.sample_rate = sample_rate
        self.attribute_calculator = SeismicAttributeCalculator(seismic_data, sample_rate)
        self.fault_probabilities = None
        self.fault_orientations = None
        self.fracture_network = None
        
    def detect_faults_ensemble(self, method='ensemble', use_gpu=True):
        """Detect faults using ensemble of advanced methods"""
        st.info("Starting advanced fault detection...")
        
        # Calculate attributes
        attributes = self.attribute_calculator.calculate_all_attributes()
        
        # Prepare data for ML models
        feature_cube = self._prepare_feature_cube(attributes)
        
        # Apply selected detection method
        if method == 'ensemble':
            faults = self._ensemble_detection(feature_cube, use_gpu)
        elif method == 'deep_learning':
            faults = self._deep_learning_detection(feature_cube, use_gpu)
        elif method == 'xgboost':
            faults = self._xgboost_detection(feature_cube)
        elif method == 'anomaly_detection':
            faults = self._anomaly_detection(feature_cube)
        elif method == 'traditional':
            faults = self._traditional_detection(feature_cube)
        else:
            faults = self._ensemble_detection(feature_cube, use_gpu)
        
        self.fault_probabilities = faults
        
        # Extract fault orientations
        self.fault_orientations = self._extract_fault_orientations(attributes, faults)
        
        # Build fracture network
        self.fracture_network = self._build_fracture_network(faults)
        
        st.success("Fault detection completed!")
        return faults, self.fault_orientations, self.fracture_network
    
    def _prepare_feature_cube(self, attributes):
        """Prepare feature cube from attributes"""
        feature_list = []
        
        # Select key attributes for fault detection
        key_attributes = ['coherence', 'dip', 'curvature', 'sobel', 
                         'variance', 'entropy', 'instantaneous_frequency']
        
        for attr_name in key_attributes:
            if attr_name in attributes:
                attr_data = attributes[attr_name]
                # Normalize attribute
                attr_min, attr_max = attr_data.min(), attr_data.max()
                if attr_max > attr_min:
                    attr_norm = (attr_data - attr_min) / (attr_max - attr_min)
                else:
                    attr_norm = np.zeros_like(attr_data)
                feature_list.append(attr_norm)
        
        # Add amplitude as base feature
        amplitude = attributes['amplitude']
        amp_min, amp_max = amplitude.min(), amplitude.max()
        if amp_max > amp_min:
            amp_norm = (amplitude - amp_min) / (amp_max - amp_min)
        else:
            amp_norm = np.zeros_like(amplitude)
        feature_list.append(amp_norm)
        
        # Stack features
        feature_cube = np.stack(feature_list, axis=-1)  # Shape: (I, X, T, features)
        return feature_cube
    
    def _ensemble_detection(self, feature_cube, use_gpu=True):
        """Ensemble detection combining multiple methods"""
        st.info("Running ensemble fault detection...")
        
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        results = []
        weights = []
        
        # Method 1: Deep Learning
        st.info("Method 1/4: Deep Learning Detection")
        dl_result = self._deep_learning_detection(feature_cube, use_gpu)
        results.append(dl_result)
        weights.append(0.4)
        
        # Method 2: XGBoost
        st.info("Method 2/4: XGBoost Detection")
        xgb_result = self._xgboost_detection(feature_cube)
        results.append(xgb_result)
        weights.append(0.3)
        
        # Method 3: Anomaly Detection
        st.info("Method 3/4: Anomaly Detection")
        anomaly_result = self._anomaly_detection(feature_cube)
        results.append(anomaly_result)
        weights.append(0.2)
        
        # Method 4: Traditional
        st.info("Method 4/4: Traditional Edge Detection")
        trad_result = self._traditional_detection(feature_cube)
        results.append(trad_result)
        weights.append(0.1)
        
        # Combine results
        ensemble_result = np.zeros_like(dl_result)
        total_weight = sum(weights)
        
        for result, weight in zip(results, weights):
            ensemble_result += result * (weight / total_weight)
        
        return ensemble_result
    
    def _deep_learning_detection(self, feature_cube, use_gpu=True):
        """Deep learning based fault detection"""
        st.info("Initializing deep learning model...")
        
        # Prepare data for 3D CNN
        feature_tensor = torch.FloatTensor(feature_cube).permute(3, 0, 1, 2).unsqueeze(0)  # Add batch dim
        
        # Initialize model
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = FaultDetectionModel(input_channels=feature_cube.shape[-1])
        model.to(device)
        model.eval()
        
        # Load or train model (simplified - in production would load pre-trained)
        # For demo, we'll use a simplified detection
        
        # Simplified CNN detection (actual implementation would train on labeled data)
        with torch.no_grad():
            feature_tensor = feature_tensor.to(device)
            
            # Use multiple convolutional filters for edge detection
            kernel_size = 5
            conv_layers = []
            
            for _ in range(4):
                conv_layer = nn.Conv3d(feature_cube.shape[-1], 16, kernel_size=kernel_size, padding=kernel_size//2)
                conv_layer.to(device)
                conv_layers.append(conv_layer)
            
            # Apply convolutional layers
            features = []
            for conv_layer in conv_layers:
                features.append(conv_layer(feature_tensor))
            
            # Combine features
            combined = torch.cat(features, dim=1)
            
            # Final detection layer
            detector = nn.Conv3d(64, 1, kernel_size=3, padding=1)
            detector.to(device)
            
            detection = detector(combined)
            detection = torch.sigmoid(detection)
            
            # Convert to numpy
            detection_np = detection.cpu().numpy()[0, 0]
        
        return detection_np
    
    def _xgboost_detection(self, feature_cube):
        """XGBoost based fault detection"""
        st.info("Running XGBoost detection...")
        
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        
        # Reshape for XGBoost
        X = feature_cube.reshape(-1, n_features)
        
        # Sample for training (in production, would use actual labels)
        n_samples_train = min(10000, len(X))
        indices = np.random.choice(len(X), n_samples_train, replace=False)
        X_train = X[indices]
        
        # Generate synthetic labels for demonstration
        # In production, these would come from labeled fault data
        y_train = self._generate_synthetic_labels(X_train)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Predict on all data (in batches for memory efficiency)
        batch_size = 10000
        predictions = np.zeros(len(X))
        
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_batch = X[i:end_idx]
            pred_batch = model.predict_proba(X_batch)[:, 1]
            predictions[i:end_idx] = pred_batch
        
        # Reshape back to original dimensions
        fault_prob = predictions.reshape(n_inlines, n_xlines, n_samples)
        
        return fault_prob
    
    def _generate_synthetic_labels(self, X):
        """Generate synthetic labels for training (for demo)"""
        # This is a simplified version - in production would use actual labeled data
        # Based on feature patterns that typically indicate faults
        
        # Combine features with weights that emphasize discontinuity features
        weights = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])[:X.shape[1]]
        
        # Calculate weighted score
        scores = np.dot(X, weights)
        
        # Normalize scores
        scores = (scores - scores.mean()) / scores.std()
        
        # Generate labels (faults are where score is high)
        labels = (scores > 1.5).astype(int)
        
        # Add some noise for realism
        noise = np.random.rand(len(labels)) > 0.95
        labels[noise] = 1 - labels[noise]
        
        return labels
    
    def _anomaly_detection(self, feature_cube):
        """Anomaly detection based fault detection"""
        st.info("Running anomaly detection...")
        
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        
        # Reshape for anomaly detection
        X = feature_cube.reshape(-1, n_features)
        
        # Sample for efficiency
        n_samples_anomaly = min(5000, len(X))
        indices = np.random.choice(len(X), n_samples_anomaly, replace=False)
        X_sample = X[indices]
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        
        iso_forest.fit(X_sample)
        
        # Predict anomalies in batches
        batch_size = 10000
        anomaly_scores = np.zeros(len(X))
        
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_batch = X[i:end_idx]
            scores_batch = iso_forest.decision_function(X_batch)
            anomaly_scores[i:end_idx] = scores_batch
        
        # Convert to probability (lower scores = more anomalous)
        anomaly_scores = -anomaly_scores  # Invert so higher = more anomalous
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Reshape back
        fault_prob = anomaly_scores.reshape(n_inlines, n_xlines, n_samples)
        
        return fault_prob
    
    def _traditional_detection(self, feature_cube):
        """Traditional edge detection methods"""
        st.info("Running traditional edge detection...")
        
        n_inlines, n_xlines, n_samples, n_features = feature_cube.shape
        
        # Use coherence and gradient features
        coherence_idx = 0  # Assuming coherence is first feature
        gradient_idx = 3   # Assuming gradient features
        
        coherence = feature_cube[:, :, :, coherence_idx]
        gradient = feature_cube[:, :, :, gradient_idx]
        
        # Combine with weights
        fault_prob = 0.6 * coherence + 0.4 * gradient
        
        # Apply thresholding and smoothing
        fault_prob = np.where(fault_prob > 0.7, fault_prob, 0)
        
        return fault_prob
    
    def _extract_fault_orientations(self, attributes, fault_probabilities):
        """Extract fault orientations from detected faults"""
        st.info("Extracting fault orientations...")
        
        # Use gradient information from attributes
        if 'dip' in attributes and 'azimuth' in attributes:
            dip = attributes['dip']
            azimuth = attributes['azimuth']
            
            # Only calculate orientations where fault probability is high
            fault_mask = fault_probabilities > 0.5
            
            orientations = np.zeros((*fault_probabilities.shape, 3))
            orientations[..., 0] = dip * fault_mask  # Dip
            orientations[..., 1] = azimuth * fault_mask  # Azimuth
            orientations[..., 2] = fault_probabilities  # Confidence
            
            return orientations
        
        return None
    
    def _build_fracture_network(self, fault_probabilities):
        """Build connected fracture network from fault probabilities"""
        st.info("Building fracture network...")
        
        # Threshold fault probabilities
        fault_binary = fault_probabilities > 0.5
        
        # Label connected components
        from scipy.ndimage import label
        labeled_array, num_features = label(fault_binary)
        
        # Calculate properties for each fracture
        fracture_properties = []
        for i in range(1, num_features + 1):
            fracture_mask = labeled_array == i
            if np.sum(fracture_mask) > 10:  # Minimum size threshold
                # Calculate center of mass
                indices = np.where(fracture_mask)
                center = [np.mean(idx) for idx in indices]
                
                # Calculate size
                size = np.sum(fracture_mask)
                
                # Calculate orientation (principal components)
                coords = np.array(indices).T
                if len(coords) > 2:
                    pca = PCA(n_components=3)
                    pca.fit(coords)
                    orientation = pca.components_[0]  # Primary orientation
                    
                    fracture_properties.append({
                        'id': i,
                        'center': center,
                        'size': size,
                        'orientation': orientation.tolist(),
                        'mask': fracture_mask
                    })
        
        st.info(f"Detected {len(fracture_properties)} fracture segments")
        return fracture_properties
    
    def visualize_faults(self, inline_idx=None, crossline_idx=None, time_slice=None):
        """Visualize detected faults"""
        if self.fault_probabilities is None:
            st.error("No fault detection results available. Run detection first.")
            return None
        
        n_inlines, n_xlines, n_samples = self.fault_probabilities.shape
        
        # Determine visualization type
        if inline_idx is not None:
            # Inline view
            inline_idx = min(max(inline_idx, 0), n_inlines - 1)
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
            aspect_ratio = n_samples / n_xlines
            
        elif crossline_idx is not None:
            # Crossline view
            crossline_idx = min(max(crossline_idx, 0), n_xlines - 1)
            data_slice = self.fault_probabilities[:, crossline_idx, :]
            title = f"Fault Detection - Crossline {crossline_idx}"
            aspect_ratio = n_samples / n_inlines
            
        elif time_slice is not None:
            # Time slice view
            time_slice = min(max(time_slice, 0), n_samples - 1)
            data_slice = self.fault_probabilities[:, :, time_slice]
            title = f"Fault Detection - Time Slice {time_slice}"
            aspect_ratio = n_xlines / n_inlines
            
        else:
            # Default to middle inline
            inline_idx = n_inlines // 2
            data_slice = self.fault_probabilities[inline_idx, :, :]
            title = f"Fault Detection - Inline {inline_idx}"
            aspect_ratio = n_samples / n_xlines
        
        # Create visualization
        fig = go.Figure()
        
        # Heatmap of fault probabilities
        fig.add_trace(go.Heatmap(
            z=data_slice.T if inline_idx is not None or crossline_idx is not None else data_slice,
            colorscale='hot',
            opacity=0.8,
            name='Fault Probability'
        ))
        
        # Overlay seismic amplitude for context
        if inline_idx is not None:
            seismic_slice = self.seismic_data[inline_idx, :, :]
        elif crossline_idx is not None:
            seismic_slice = self.seismic_data[:, crossline_idx, :]
        else:
            seismic_slice = self.seismic_data[:, :, time_slice]
        
        # Normalize seismic for contours
        seismic_norm = (seismic_slice - seismic_slice.min()) / (seismic_slice.max() - seismic_slice.min())
        
        fig.add_trace(go.Contour(
            z=seismic_norm.T if inline_idx is not None or crossline_idx is not None else seismic_norm,
            colorscale='gray',
            opacity=0.3,
            showscale=False,
            contours=dict(
                showlabels=False,
                coloring='lines'
            ),
            name='Seismic Amplitude'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Crossline" if inline_idx is not None else "Inline",
            yaxis_title="Time Sample" if inline_idx is not None or crossline_idx is not None else "Inline",
            width=800,
            height=600 * aspect_ratio,
            showlegend=True
        )
        
        return fig
    
    def export_fault_network(self, output_format='csv'):
        """Export fault network data"""
        if self.fracture_network is None:
            st.error("No fracture network available. Run detection first.")
            return None
        
        try:
            # Prepare data for export
            export_data = []
            for fracture in self.fracture_network:
                export_data.append({
                    'fracture_id': fracture['id'],
                    'center_inline': fracture['center'][0],
                    'center_crossline': fracture['center'][1],
                    'center_time': fracture['center'][2],
                    'size_voxels': fracture['size'],
                    'orientation_x': fracture['orientation'][0],
                    'orientation_y': fracture['orientation'][1],
                    'orientation_z': fracture['orientation'][2]
                })
            
            if output_format == 'csv':
                import pandas as pd
                df = pd.DataFrame(export_data)
                return df.to_csv(index=False)
            elif output_format == 'json':
                return json.dumps(export_data, indent=2)
            elif output_format == 'numpy':
                return np.array([list(f.values()) for f in export_data])
            else:
                return export_data
                
        except Exception as e:
            st.error(f"Error exporting fault network: {e}")
            return None

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

# Available colormaps - expanded to include all needed colormaps
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'hot', 'cool', 'jet', 'rainbow', 'turbo',
    'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
    'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG', 'RdGy',
    'Reds', 'Greens', 'Blues', 'Oranges', 'Purples'
]

# Wavelet types
WAVELET_TYPES = ['FFT', 'Morlet', 'Ricker', 'CWT']

# Fault detection methods
FAULT_DETECTION_METHODS = {
    'ensemble': 'Ensemble (Combined Methods)',
    'deep_learning': 'Deep Learning (3D CNN)',
    'xgboost': 'XGBoost (Gradient Boosting)',
    'anomaly_detection': 'Anomaly Detection',
    'traditional': 'Traditional Edge Detection'
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

def create_rgb_plotly_figure(rgb_data, title):
    """Create a Plotly figure for RGB data using go.Image"""
    if rgb_data.ndim == 3 and rgb_data.shape[2] == 3:
        rgb_uint8 = (rgb_data * 255).astype(np.uint8)
        
        fig = go.Figure()
        fig.add_trace(go.Image(z=rgb_uint8))
        
        fig.update_layout(
            title=title,
            xaxis_title="Position",
            yaxis_title="Position",
            width=600,
            height=500
        )
        
        return fig
    else:
        st.error(f"Invalid RGB data shape: {rgb_data.shape}. Expected (height, width, 3)")
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

def compute_average_spectrum(section_data, sample_rate, section_type):
    """Compute average frequency spectrum for a section"""
    try:
        if section_data is None or section_data.size == 0:
            return np.array([]), np.array([])
            
        if section_type == "Inline":
            n_traces = section_data.shape[1]
            avg_amplitude = None
            freqs_positive = None
            
            for j in range(n_traces):
                trace = section_data[:, j]
                freqs, amplitude = compute_frequency_spectrum(trace, sample_rate)
                if amplitude.size > 0:
                    if avg_amplitude is None:
                        avg_amplitude = amplitude
                        freqs_positive = freqs
                    else:
                        avg_amplitude += amplitude
            
            if avg_amplitude is not None:
                avg_amplitude /= n_traces
            return freqs_positive, avg_amplitude
        
        elif section_type == "Crossline":
            n_traces = section_data.shape[0]
            avg_amplitude = None
            freqs_positive = None
            
            for i in range(n_traces):
                trace = section_data[i, :]
                freqs, amplitude = compute_frequency_spectrum(trace, sample_rate)
                if amplitude.size > 0:
                    if avg_amplitude is None:
                        avg_amplitude = amplitude
                        freqs_positive = freqs
                    else:
                        avg_amplitude += amplitude
            
            if avg_amplitude is not None:
                avg_amplitude /= n_traces
            return freqs_positive, avg_amplitude
            
        else:
            return np.array([]), np.array([])
            
    except Exception as e:
        st.warning(f"Error computing average spectrum: {e}")
        return np.array([]), np.array([])

def safe_get_colormap_index(colormap_name, default_index=0):
    """Safely get the index of a colormap in the COLORMAPS list"""
    try:
        return COLORMAPS.index(colormap_name)
    except ValueError:
        return default_index

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
                                        st.sidebar.info("This is a numpy format with metadata header")
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
                        st.success("Enhanced file ready for download!")
                
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
                        
                        freqs_orig, amp_orig = compute_average_spectrum(original_inline, enhancer.sample_rate, "Inline")
                        if amp_orig is not None and len(amp_orig) > 0:
                            fig_spec_orig = go.Figure()
                            fig_spec_orig.add_trace(go.Scatter(x=freqs_orig, y=amp_orig, mode='lines', name='Original', line=dict(color='blue')))
                            fig_spec_orig.update_layout(
                                title=f"Frequency Spectrum - Original Inline {inline_idx}",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="Amplitude",
                                height=300
                            )
                            st.plotly_chart(fig_spec_orig, use_container_width=True)
                        else:
                            st.info("Frequency spectrum not available for this section type")
                    
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
                        
                        if display_type == "Amplitude":
                            freqs_enh, amp_enh = compute_average_spectrum(enhanced_inline, enhancer.sample_rate, "Inline")
                            if amp_enh is not None and len(amp_enh) > 0:
                                fig_spec_enh = go.Figure()
                                fig_spec_enh.add_trace(go.Scatter(x=freqs_orig, y=amp_orig, mode='lines', name='Original', line=dict(color='blue', dash='dash')))
                                fig_spec_enh.add_trace(go.Scatter(x=freqs_enh, y=amp_enh, mode='lines', name='Enhanced', line=dict(color='red')))
                                fig_spec_enh.update_layout(
                                    title=f"Frequency Spectrum - Enhanced Inline {inline_idx}",
                                    xaxis_title="Frequency (Hz)",
                                    yaxis_title="Amplitude",
                                    height=300
                                )
                                st.plotly_chart(fig_spec_enh, use_container_width=True)
                
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
                        
                        freqs_orig, amp_orig = compute_average_spectrum(original_xline, enhancer.sample_rate, "Crossline")
                        if amp_orig is not None and len(amp_orig) > 0:
                            fig_spec_orig = go.Figure()
                            fig_spec_orig.add_trace(go.Scatter(x=freqs_orig, y=amp_orig, mode='lines', name='Original', line=dict(color='blue')))
                            fig_spec_orig.update_layout(
                                title=f"Frequency Spectrum - Original Crossline {crossline_idx}",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="Amplitude",
                                height=300
                            )
                            st.plotly_chart(fig_spec_orig, use_container_width=True)
                        else:
                            st.info("Frequency spectrum not available for this section type")
                    
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
                        
                        if display_type == "Amplitude":
                            freqs_enh, amp_enh = compute_average_spectrum(enhanced_xline, enhancer.sample_rate, "Crossline")
                            if amp_enh is not None and len(amp_enh) > 0:
                                fig_spec_enh = go.Figure()
                                fig_spec_enh.add_trace(go.Scatter(x=freqs_orig, y=amp_orig, mode='lines', name='Original', line=dict(color='blue', dash='dash')))
                                fig_spec_enh.add_trace(go.Scatter(x=freqs_enh, y=amp_enh, mode='lines', name='Enhanced', line=dict(color='red')))
                                fig_spec_enh.update_layout(
                                    title=f"Frequency Spectrum - Enhanced Crossline {crossline_idx}",
                                    xaxis_title="Frequency (Hz)",
                                    yaxis_title="Amplitude",
                                    height=300
                                )
                                st.plotly_chart(fig_spec_enh, use_container_width=True)
                
                else:
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
                        
                        st.info("Frequency spectrum analysis requires full seismic traces (use Inline or Crossline view)")
                    
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
        
        st.header("About This Tool")
        st.markdown("""
        This tool enhances 3D seismic data bandwidth using **spectral blueing** techniques.
        
        ### How to Use:
        1. Upload a SEG-Y file
        2. Adjust processing parameters
        3. Click "Process Seismic Data"
        4. Generate and download the enhanced file
        
        ### Output Formats:
        - **SEG-Y format**: Standard seismic format (preferred)
        - **Numpy format**: With metadata header (fallback)
        
        Both formats contain the enhanced seismic data ready for analysis.
        """)

def display_spectral_decomposition_tab(enhancer):
    """Display the spectral decomposition tab"""
    st.header("🔬 Spectral Decomposition Analysis")
    
    if enhancer.original_data is None:
        st.info("Please load a SEG-Y file in the Bandwidth Enhancement tab first.")
        return
    
    st.sidebar.header("🌊 Wavelet Settings")
    st.sidebar.subheader("Wavelet Type Selection")
    wavelet_type = st.sidebar.selectbox(
        "Wavelet Type", 
        WAVELET_TYPES, 
        index=0,
        help="Choose the wavelet transform method for spectral decomposition",
        key="sd_wavelet_type"
    )
    
    if wavelet_type == 'Morlet':
        st.sidebar.subheader("Morlet Wavelet Parameters")
        morlet_cycles = st.sidebar.slider("Number of Cycles", 4, 10, 6, 
                                        help="Number of cycles in the Morlet wavelet",
                                        key="sd_morlet_cycles")
        wavelet_params = {'cycles': morlet_cycles}
        
    elif wavelet_type == 'CWT':
        st.sidebar.subheader("CWT Parameters")
        cwt_scales = st.sidebar.slider("Scales Factor", 0.5, 3.0, 1.0, 0.1,
                                     help="Scale factor for Continuous Wavelet Transform",
                                     key="sd_cwt_scales")
        wavelet_params = {'scales_factor': cwt_scales}
        
    else:
        wavelet_params = {}
    
    st.sidebar.header("🎨 Spectral Decomposition Settings")
    st.sidebar.subheader("Colormap Selection")
    spectral_colormap = st.sidebar.selectbox("Spectral Colormap", COLORMAPS, index=0, key="sd_cmap")
    red_component_cmap = st.sidebar.selectbox("Red Component Colormap", COLORMAPS, 
                                            index=safe_get_colormap_index('Reds', 22), key="sd_red_cmap")
    green_component_cmap = st.sidebar.selectbox("Green Component Colormap", COLORMAPS, 
                                              index=safe_get_colormap_index('Greens', 23), key="sd_green_cmap")
    blue_component_cmap = st.sidebar.selectbox("Blue Component Colormap", COLORMAPS, 
                                             index=safe_get_colormap_index('Blues', 24), key="sd_blue_cmap")
    
    col1, col2 = st.columns(2)
    with col1:
        section_type = st.selectbox("Select Section Type", 
                                  ["Time Slice", "Inline", "Crossline"],
                                  key="sd_section_type")
    
    n_inlines, n_xlines, n_samples = enhancer.original_data.shape
    
    inline_idx = n_inlines // 2 if n_inlines > 0 else 0
    crossline_idx = n_xlines // 2 if n_xlines > 0 else 0
    time_slice = n_samples // 2 if n_samples > 0 else 0
    actual_time = time_slice
    
    if section_type == "Time Slice":
        with col2:
            time_slice = st.slider("Time Slice", 0, n_samples-1, time_slice, key="sd_time")
        if enhancer.geometry and 'samples' in enhancer.geometry and len(enhancer.geometry['samples']) > time_slice:
            actual_time = enhancer.geometry['samples'][time_slice]
        st.info(f"Selected time: {actual_time} ms")
    elif section_type == "Inline":
        with col2:
            inline_idx = st.slider("Inline", 0, n_inlines-1, inline_idx, key="sd_inline")
        st.info(f"Selected inline: {inline_idx}")
    else:
        with col2:
            crossline_idx = st.slider("Crossline", 0, n_xlines-1, crossline_idx, key="sd_crossline")
        st.info(f"Selected crossline: {crossline_idx}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_freq = st.slider("Minimum Frequency (Hz)", 5, 50, 10, key="sd_min_freq")
    with col2:
        max_freq = st.slider("Maximum Frequency (Hz)", 60, 200, 100, key="sd_max_freq")
    with col3:
        num_frequencies = st.slider("Number of Frequencies", 5, 20, 10, key="sd_num_freq")
    
    frequencies = np.linspace(min_freq, max_freq, num_frequencies).astype(int)
    
    if st.button("Compute Spectral Decomposition", type="primary", key="sd_compute"):
        with st.spinner(f"Computing spectral decomposition using {wavelet_type}... This may take a while for large datasets."):
            spectral_analyzer = SpectralDecomposition(enhancer.original_data, enhancer.sample_rate)
            frequency_data = spectral_analyzer.compute_spectral_decomposition(
                frequencies, 
                wavelet_type=wavelet_type,
                **wavelet_params
            )
            
            st.session_state.spectral_analyzer = spectral_analyzer
            st.session_state.frequency_data = frequency_data
            st.session_state.frequencies = frequencies
            st.session_state.wavelet_type = wavelet_type
    
    if 'spectral_analyzer' in st.session_state:
        spectral_analyzer = st.session_state.spectral_analyzer
        frequencies = st.session_state.frequencies
        current_wavelet = st.session_state.get('wavelet_type', 'FFT')
        
        st.info(f"Using {current_wavelet} wavelet for spectral decomposition")
        
        st.subheader("📊 Frequency Explorer")
        selected_freq_idx = st.selectbox("Select Frequency", range(len(frequencies)), 
                                       format_func=lambda i: f"{frequencies[i]} Hz",
                                       key="sd_freq_select")
        
        if section_type == "Time Slice":
            freq_data = spectral_analyzer.get_frequency_slice(selected_freq_idx, time_slice)
            title = f"Frequency Slice: {frequencies[selected_freq_idx]} Hz at {actual_time} ms ({current_wavelet})"
            display_data = freq_data
        elif section_type == "Inline":
            freq_data = spectral_analyzer.get_frequency_inline(selected_freq_idx, inline_idx)
            title = f"Frequency Inline: {frequencies[selected_freq_idx]} Hz at Inline {inline_idx} ({current_wavelet})"
            display_data = freq_data.T
        else:
            freq_data = spectral_analyzer.get_frequency_crossline(selected_freq_idx, crossline_idx)
            title = f"Frequency Crossline: {frequencies[selected_freq_idx]} Hz at Crossline {crossline_idx} ({current_wavelet})"
            display_data = freq_data.T
        
        fig = px.imshow(display_data, 
                       title=title,
                       color_continuous_scale=spectral_colormap,
                       aspect='auto')
        fig.update_layout(coloraxis_colorbar=dict(title="Amplitude"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎨 RGB Frequency Blending")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_freq_idx = st.selectbox("Low Frequency (Red)", range(len(frequencies)), 
                                      index=0, format_func=lambda i: f"{frequencies[i]} Hz",
                                      key="sd_low_freq")
        with col2:
            mid_freq_idx = st.selectbox("Mid Frequency (Green)", range(len(frequencies)), 
                                      index=len(frequencies)//2, format_func=lambda i: f"{frequencies[i]} Hz",
                                      key="sd_mid_freq")
        with col3:
            high_freq_idx = st.selectbox("High Frequency (Blue)", range(len(frequencies)), 
                                       index=len(frequencies)-1, format_func=lambda i: f"{frequencies[i]} Hz",
                                       key="sd_high_freq")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            red_weight = st.slider("Red Weight", 0.0, 2.0, 1.0, 0.1, key="sd_red_weight")
        with col2:
            green_weight = st.slider("Green Weight", 0.0, 2.0, 1.0, 0.1, key="sd_green_weight")
        with col3:
            blue_weight = st.slider("Blue Weight", 0.0, 2.0, 1.0, 0.1, key="sd_blue_weight")
        
        if st.button("Generate RGB Blend", key="sd_rgb_blend"):
            with st.spinner("Creating RGB frequency blend..."):
                if section_type == "Time Slice":
                    rgb_blend = spectral_analyzer.create_rgb_blend(
                        low_freq_idx, mid_freq_idx, high_freq_idx, 
                        time_slice, 
                        weights=(red_weight, green_weight, blue_weight)
                    )
                    title_suffix = f"at Time {actual_time} ms"
                    display_rgb = rgb_blend
                elif section_type == "Inline":
                    rgb_blend = spectral_analyzer.create_rgb_inline(
                        low_freq_idx, mid_freq_idx, high_freq_idx,
                        inline_idx,
                        weights=(red_weight, green_weight, blue_weight)
                    )
                    title_suffix = f"at Inline {inline_idx}"
                    display_rgb = rgb_blend.transpose(1, 0, 2)
                else:
                    rgb_blend = spectral_analyzer.create_rgb_crossline(
                        low_freq_idx, mid_freq_idx, high_freq_idx,
                        crossline_idx,
                        weights=(red_weight, green_weight, blue_weight)
                    )
                    title_suffix = f"at Crossline {crossline_idx}"
                    display_rgb = rgb_blend.transpose(1, 0, 2)
                
                fig_rgb = create_rgb_plotly_figure(
                    display_rgb, 
                    f"RGB Frequency Blend {title_suffix} ({current_wavelet})\n"
                    f"Low (R): {frequencies[low_freq_idx]}Hz, "
                    f"Mid (G): {frequencies[mid_freq_idx]}Hz, "
                    f"High (B): {frequencies[high_freq_idx]}Hz"
                )
                if fig_rgb:
                    st.plotly_chart(fig_rgb, use_container_width=True)
                
                st.subheader("Individual Frequency Components")
                col1, col2, col3 = st.columns(3)
                
                def get_component_data(freq_idx, section_type, time_slice, inline_idx, crossline_idx):
                    if section_type == "Time Slice":
                        return spectral_analyzer.get_frequency_slice(freq_idx, time_slice)
                    elif section_type == "Inline":
                        return spectral_analyzer.get_frequency_inline(freq_idx, inline_idx)
                    else:
                        return spectral_analyzer.get_frequency_crossline(freq_idx, crossline_idx)
                
                def get_display_data(data, section_type):
                    if section_type == "Time Slice":
                        return data
                    else:
                        return data.T
                
                with col1:
                    low_component = get_component_data(low_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    display_low = get_display_data(low_component, section_type)
                    fig_low = px.imshow(display_low, 
                                      title=f"Low Freq: {frequencies[low_freq_idx]} Hz",
                                      color_continuous_scale=red_component_cmap,
                                      aspect='auto')
                    st.plotly_chart(fig_low, use_container_width=True)
                
                with col2:
                    mid_component = get_component_data(mid_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    display_mid = get_display_data(mid_component, section_type)
                    fig_mid = px.imshow(display_mid, 
                                      title=f"Mid Freq: {frequencies[mid_freq_idx]} Hz",
                                      color_continuous_scale=green_component_cmap,
                                      aspect='auto')
                    st.plotly_chart(fig_mid, use_container_width=True)
                
                with col3:
                    high_component = get_component_data(high_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    display_high = get_display_data(high_component, section_type)
                    fig_high = px.imshow(display_high, 
                                       title=f"High Freq: {frequencies[high_freq_idx]} Hz",
                                       color_continuous_scale=blue_component_cmap,
                                       aspect='auto')
                    st.plotly_chart(fig_high, use_container_width=True)

def display_fault_detection_tab(enhancer):
    """Display the advanced fault and fracture detection tab"""
    st.header("🔍 AI-Powered Fault & Fracture Detection")
    
    if enhancer.enhanced_data is None:
        st.info("Please process seismic data in the Bandwidth Enhancement tab first.")
        return
    
    st.sidebar.header("⚙️ Fault Detection Parameters")
    
    # Method selection
    st.sidebar.subheader("Detection Method")
    method = st.sidebar.selectbox(
        "Select Detection Method",
        options=list(FAULT_DETECTION_METHODS.keys()),
        format_func=lambda x: FAULT_DETECTION_METHODS[x],
        key="fd_method"
    )
    
    st.sidebar.subheader("Detection Sensitivity")
    sensitivity = st.sidebar.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values detect more subtle faults but may increase false positives",
        key="fd_sensitivity"
    )
    
    # GPU acceleration
    use_gpu = st.sidebar.checkbox(
        "Use GPU Acceleration (if available)",
        value=True,
        help="Use GPU for faster deep learning computations",
        key="fd_use_gpu"
    )
    
    st.sidebar.subheader("Post-processing")
    apply_smoothing = st.sidebar.checkbox(
        "Apply Smoothing",
        value=True,
        help="Apply Gaussian smoothing to detection results",
        key="fd_smoothing"
    )
    
    if apply_smoothing:
        smoothing_sigma = st.sidebar.slider(
            "Smoothing Sigma",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            key="fd_sigma"
        )
    
    # Visualization settings
    st.sidebar.header("🎨 Visualization Settings")
    fault_colormap = st.sidebar.selectbox(
        "Fault Probability Colormap",
        COLORMAPS,
        index=safe_get_colormap_index('hot', 6),
        key="fd_cmap"
    )
    
    show_seismic_overlay = st.sidebar.checkbox(
        "Show Seismic Overlay",
        value=True,
        help="Overlay seismic amplitude for context",
        key="fd_overlay"
    )
    
    # Main detection section
    st.markdown("""
    ### Advanced Fault Detection System
    This system uses state-of-the-art machine learning and AI techniques to detect 
    faults and fractures in 3D seismic data with unprecedented accuracy.
    
    **Key Features:**
    - Ensemble learning combining multiple detection methods
    - Deep learning with 3D convolutional neural networks
    - Advanced seismic attribute analysis
    - Automated fracture network extraction
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Run Fault Detection", type="primary", use_container_width=True):
            with st.spinner(f"Running {FAULT_DETECTION_METHODS[method]} detection..."):
                # Initialize fault detector
                fault_detector = AdvancedFaultDetector(
                    enhancer.enhanced_data,
                    enhancer.sample_rate
                )
                
                # Run detection
                faults, orientations, fracture_network = fault_detector.detect_faults_ensemble(
                    method=method,
                    use_gpu=use_gpu
                )
                
                # Store in session state
                st.session_state.fault_detector = fault_detector
                st.session_state.faults = faults
                st.session_state.fracture_network = fracture_network
                st.session_state.detection_completed = True
                
                st.success("✅ Fault detection completed!")
    
    with col2:
        if st.button("📊 View Detection Statistics", type="secondary", use_container_width=True):
            if 'fault_detector' in st.session_state:
                st.info("Detection statistics will be displayed below")
            else:
                st.warning("Please run fault detection first")
    
    # Display results if available
    if st.session_state.get('detection_completed', False):
        fault_detector = st.session_state.fault_detector
        faults = st.session_state.faults
        fracture_network = st.session_state.fracture_network
        
        st.subheader("📈 Detection Results")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_voxels = np.prod(faults.shape)
            fault_voxels = np.sum(faults > 0.5)
            fault_percentage = (fault_voxels / total_voxels) * 100
            st.metric("Fault Probability > 50%", f"{fault_percentage:.2f}%")
        
        with col2:
            avg_probability = np.mean(faults)
            st.metric("Average Probability", f"{avg_probability:.3f}")
        
        with col3:
            max_probability = np.max(faults)
            st.metric("Maximum Probability", f"{max_probability:.3f}")
        
        with col4:
            if fracture_network:
                st.metric("Fracture Segments", len(fracture_network))
            else:
                st.metric("Fracture Segments", "0")
        
        # Interactive visualization
        st.subheader("📊 Interactive Visualization")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            view_type = st.selectbox(
                "View Type",
                ["Inline", "Crossline", "Time Slice", "3D Volume"],
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
            
            fig = fault_detector.visualize_faults(inline_idx=inline_idx)
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
            
            fig = fault_detector.visualize_faults(crossline_idx=crossline_idx)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        elif view_type == "Time Slice":
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
        
        else:  # 3D Volume
            st.info("3D volume visualization requires advanced rendering. Showing maximum intensity projection instead.")
            
            # Create maximum intensity projection
            mip_inline = np.max(faults, axis=0)
            mip_crossline = np.max(faults, axis=1)
            mip_time = np.max(faults, axis=2)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1 = px.imshow(
                    mip_inline.T,
                    title="Maximum Intensity Projection (Inline)",
                    color_continuous_scale=fault_colormap,
                    aspect='auto'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.imshow(
                    mip_crossline.T,
                    title="Maximum Intensity Projection (Crossline)",
                    color_continuous_scale=fault_colormap,
                    aspect='auto'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                fig3 = px.imshow(
                    mip_time,
                    title="Maximum Intensity Projection (Time)",
                    color_continuous_scale=fault_colormap,
                    aspect='auto'
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        # Fracture network analysis
        if fracture_network:
            st.subheader("🔗 Fracture Network Analysis")
            
            # Create summary table
            import pandas as pd
            
            fracture_data = []
            for fracture in fracture_network[:10]:  # Show first 10
                fracture_data.append({
                    'ID': fracture['id'],
                    'Size (voxels)': fracture['size'],
                    'Center Inline': f"{fracture['center'][0]:.1f}",
                    'Center Crossline': f"{fracture['center'][1]:.1f}",
                    'Center Time': f"{fracture['center'][2]:.1f}"
                })
            
            if fracture_data:
                df = pd.DataFrame(fracture_data)
                st.dataframe(df, use_container_width=True)
                
                # Size distribution
                sizes = [f['size'] for f in fracture_network]
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=sizes,
                    nbinsx=20,
                    name='Fracture Sizes',
                    marker_color='orange'
                ))
                
                fig_dist.update_layout(
                    title="Fracture Size Distribution",
                    xaxis_title="Size (voxels)",
                    yaxis_title="Count",
                    bargap=0.1
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Fault Probabilities", use_container_width=True):
                # Create numpy file with fault probabilities
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, f"fault_probabilities_{uuid.uuid4()[:8]}.npy")
                np.save(output_file, faults)
                
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download Fault Probabilities",
                        data=f,
                        file_name="fault_probabilities.npy",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        with col2:
            if fracture_network:
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "JSON", "GeoJSON"],
                    key="fd_export_format"
                )
                
                export_data = fault_detector.export_fault_network(
                    output_format=export_format.lower()
                )
                
                if export_data is not None:
                    st.download_button(
                        label=f"📥 Download Fracture Network ({export_format})",
                        data=export_data,
                        file_name=f"fracture_network.{export_format.lower()}",
                        mime="text/plain",
                        use_container_width=True
                    )
        
        with col3:
            if st.button("Generate Detection Report", use_container_width=True):
                # Create comprehensive report
                report = {
                    "detection_method": FAULT_DETECTION_METHODS[method],
                    "detection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "data_shape": faults.shape,
                    "statistics": {
                        "total_voxels": int(total_voxels),
                        "fault_voxels": int(fault_voxels),
                        "fault_percentage": float(fault_percentage),
                        "average_probability": float(avg_probability),
                        "maximum_probability": float(max_probability)
                    },
                    "fracture_network_summary": {
                        "total_fractures": len(fracture_network) if fracture_network else 0,
                        "largest_fracture": int(max(sizes)) if fracture_network else 0,
                        "average_size": float(np.mean(sizes)) if fracture_network else 0
                    }
                }
                
                report_json = json.dumps(report, indent=2)
                
                st.download_button(
                    label="📥 Download Detection Report",
                    data=report_json,
                    file_name="fault_detection_report.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    else:
        # Show example/demo if no detection run yet
        st.info("""
        ### Ready to Detect Faults?
        
        This advanced fault detection system uses cutting-edge AI techniques:
        
        **Available Methods:**
        1. **Ensemble Detection** - Combines multiple methods for highest accuracy
        2. **Deep Learning** - 3D convolutional neural networks
        3. **XGBoost** - Gradient boosting with feature importance
        4. **Anomaly Detection** - Statistical methods for outlier detection
        5. **Traditional** - Edge detection and coherence-based methods
        
        **Benefits:**
        - Higher accuracy than commercial software
        - Automated fracture network extraction
        - Quantitative fault probability maps
        - Exportable results for further analysis
        
        Click **"Run Fault Detection"** to begin!
        """)
        
        # Show example image
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://via.placeholder.com/600x400/333/FFFFFF?text=Fault+Detection+Example", 
                    caption="Example Fault Detection Output")
        
        with col2:
            st.image("https://via.placeholder.com/600x400/555/FFFFFF?text=Fracture+Network+Visualization", 
                    caption="Example Fracture Network")

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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Bandwidth Enhancement", 
        "🔬 Spectral Decomposition",
        "🤖 AI Fault Detection"
    ])
    
    with tab1:
        display_bandwidth_enhancement_tab(st.session_state.enhancer)
    
    with tab2:
        display_spectral_decomposition_tab(st.session_state.enhancer)
    
    with tab3:
        display_fault_detection_tab(st.session_state.enhancer)

if __name__ == "__main__":
    main()
