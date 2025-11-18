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
        
    def compute_spectral_decomposition(self, frequencies, window_length=100):
        """Compute spectral decomposition for given frequencies"""
        st.info(f"Computing spectral decomposition for {len(frequencies)} frequencies...")
        
        n_inlines, n_xlines, n_samples = self.seismic_data.shape
        n_freqs = len(frequencies)
        
        # Initialize frequency data array
        self.frequency_data = np.zeros((n_inlines, n_xlines, n_samples, n_freqs))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for freq_idx, target_freq in enumerate(frequencies):
            status_text.text(f"Processing frequency {target_freq} Hz ({freq_idx+1}/{n_freqs})")
            
            for i in range(n_inlines):
                for j in range(n_xlines):
                    trace = self.seismic_data[i, j, :]
                    # Apply Gaussian filter centered at target frequency
                    filtered_trace = self.gaussian_spectral_filter(trace, target_freq, window_length)
                    self.frequency_data[i, j, :, freq_idx] = np.abs(filtered_trace)
            
            progress_bar.progress((freq_idx + 1) / n_freqs)
        
        progress_bar.empty()
        status_text.empty()
        st.success("Spectral decomposition completed!")
        return self.frequency_data
    
    def gaussian_spectral_filter(self, trace, center_freq, window_length):
        """Apply Gaussian spectral filter to extract specific frequency component"""
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

def display_bandwidth_enhancement_tab(enhancer):
    """Display the bandwidth enhancement tab with inline/crossline comparison"""
    st.title("🌊 3D Seismic Bandwidth Enhancement Tool")
    
    # Sidebar
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format",
        key="bw_uploader"
    )
    
    st.sidebar.header("⚙️ Processing Parameters")
    
    # Processing presets
    st.sidebar.subheader("Processing Presets")
    preset = st.sidebar.selectbox(
        "Choose Processing Preset",
        options=list(PROCESSING_PRESETS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="bw_preset"
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
    target_freq = st.sidebar.slider("Target Frequency (Hz)", 30, 120, target_freq, key="bw_target_freq")
    enhancement_factor = st.sidebar.slider("Enhancement Factor", 1.0, 3.0, enhancement_factor, 0.1, key="bw_enhancement")
    low_freq_boost = st.sidebar.slider("Low Frequency Boost", 1.0, 2.0, low_freq_boost, 0.1, key="bw_low_boost")
    mid_freq_start = st.sidebar.slider("Mid Frequency Start (Hz)", 10, 50, mid_freq_start, key="bw_mid_start")
    
    st.sidebar.subheader("Bandpass Filter")
    lowcut = st.sidebar.slider("Low Cut (Hz)", 1, 50, lowcut, key="bw_lowcut")
    highcut = st.sidebar.slider("High Cut (Hz)", 60, 200, highcut, key="bw_highcut")
    filter_order = st.sidebar.slider("Filter Order", 2, 6, filter_order, key="bw_filter_order")
    
    # Main processing
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

            # Download section
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
                
                # Display results with interactive inline/crossline comparison
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
                
                # Interactive inline/crossline comparison
                st.header("📊 Interactive Data Comparison")
                
                # Section selection
                col1, col2, col3 = st.columns(3)
                with col1:
                    section_type = st.selectbox("Select Section Type", 
                                              ["Inline", "Crossline", "Time Slice"],
                                              key="bw_section_type")
                
                n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                
                if section_type == "Inline":
                    with col2:
                        inline_idx = st.slider("Inline", 0, n_inlines-1, n_inlines//2, key="bw_inline")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_inline_display")
                    
                    # Display inline comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        original_inline = enhancer.original_data[inline_idx, :, :]
                        fig_orig = px.imshow(original_inline.T, 
                                           title=f"Original Inline {inline_idx}",
                                           color_continuous_scale='gray',
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_inline = enhancer.enhanced_data[inline_idx, :, :]
                            fig_enh = px.imshow(enhanced_inline.T, 
                                              title=f"Enhanced Inline {inline_idx}",
                                              color_continuous_scale='gray',
                                              aspect='auto')
                        else:
                            diff_inline = enhancer.enhanced_data[inline_idx, :, :] - enhancer.original_data[inline_idx, :, :]
                            fig_enh = px.imshow(diff_inline.T, 
                                              title=f"Difference Inline {inline_idx}",
                                              color_continuous_scale='RdBu',
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                elif section_type == "Crossline":
                    with col2:
                        crossline_idx = st.slider("Crossline", 0, n_xlines-1, n_xlines//2, key="bw_crossline")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_xline_display")
                    
                    # Display crossline comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        original_xline = enhancer.original_data[:, crossline_idx, :]
                        fig_orig = px.imshow(original_xline.T, 
                                           title=f"Original Crossline {crossline_idx}",
                                           color_continuous_scale='gray',
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_xline = enhancer.enhanced_data[:, crossline_idx, :]
                            fig_enh = px.imshow(enhanced_xline.T, 
                                              title=f"Enhanced Crossline {crossline_idx}",
                                              color_continuous_scale='gray',
                                              aspect='auto')
                        else:
                            diff_xline = enhancer.enhanced_data[:, crossline_idx, :] - enhancer.original_data[:, crossline_idx, :]
                            fig_enh = px.imshow(diff_xline.T, 
                                              title=f"Difference Crossline {crossline_idx}",
                                              color_continuous_scale='RdBu',
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
                
                else:  # Time Slice
                    with col2:
                        time_slice = st.slider("Time Slice", 0, n_samples-1, n_samples//2, key="bw_time")
                    with col3:
                        display_type = st.selectbox("Display Type", ["Amplitude", "Difference"], key="bw_time_display")
                    
                    actual_time = enhancer.geometry['samples'][time_slice]
                    
                    # Display time slice comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        original_slice = enhancer.original_data[:, :, time_slice]
                        fig_orig = px.imshow(original_slice, 
                                           title=f"Original Time Slice {actual_time} ms",
                                           color_continuous_scale='gray',
                                           aspect='auto')
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        if display_type == "Amplitude":
                            enhanced_slice = enhancer.enhanced_data[:, :, time_slice]
                            fig_enh = px.imshow(enhanced_slice, 
                                              title=f"Enhanced Time Slice {actual_time} ms",
                                              color_continuous_scale='gray',
                                              aspect='auto')
                        else:
                            diff_slice = enhancer.enhanced_data[:, :, time_slice] - enhancer.original_data[:, :, time_slice]
                            fig_enh = px.imshow(diff_slice, 
                                              title=f"Difference Time Slice {actual_time} ms",
                                              color_continuous_scale='RdBu',
                                              aspect='auto')
                        st.plotly_chart(fig_enh, use_container_width=True)
        
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
        4. Generate and download the enhanced file
        
        ### Output Formats:
        - **SEG-Y format**: Standard seismic format (preferred)
        - **Numpy format**: With metadata header (fallback)
        
        Both formats contain the enhanced seismic data ready for analysis.
        """)

def display_spectral_decomposition_tab(enhancer):
    """Display the spectral decomposition tab with inline/crossline/slice selection"""
    st.header("🔬 Spectral Decomposition Analysis")
    
    if enhancer.original_data is None:
        st.info("Please load a SEG-Y file in the Bandwidth Enhancement tab first.")
        return
    
    # Section type selection
    col1, col2 = st.columns(2)
    with col1:
        section_type = st.selectbox("Select Section Type", 
                                  ["Time Slice", "Inline", "Crossline"],
                                  key="sd_section_type")
    
    n_inlines, n_xlines, n_samples = enhancer.original_data.shape
    
    if section_type == "Time Slice":
        with col2:
            time_slice = st.slider("Time Slice", 0, n_samples-1, n_samples//2, key="sd_time")
        actual_time = enhancer.geometry['samples'][time_slice]
        st.info(f"Selected time: {actual_time} ms")
    elif section_type == "Inline":
        with col2:
            inline_idx = st.slider("Inline", 0, n_inlines-1, n_inlines//2, key="sd_inline")
        st.info(f"Selected inline: {inline_idx}")
    else:  # Crossline
        with col2:
            crossline_idx = st.slider("Crossline", 0, n_xlines-1, n_xlines//2, key="sd_crossline")
        st.info(f"Selected crossline: {crossline_idx}")
    
    # Frequency range selection
    col1, col2, col3 = st.columns(3)
    with col1:
        min_freq = st.slider("Minimum Frequency (Hz)", 5, 50, 10, key="sd_min_freq")
    with col2:
        max_freq = st.slider("Maximum Frequency (Hz)", 60, 200, 100, key="sd_max_freq")
    with col3:
        num_frequencies = st.slider("Number of Frequencies", 5, 20, 10, key="sd_num_freq")
    
    frequencies = np.linspace(min_freq, max_freq, num_frequencies).astype(int)
    
    if st.button("Compute Spectral Decomposition", type="primary", key="sd_compute"):
        with st.spinner("Computing spectral decomposition... This may take a while for large datasets."):
            spectral_analyzer = SpectralDecomposition(enhancer.original_data, enhancer.sample_rate)
            frequency_data = spectral_analyzer.compute_spectral_decomposition(frequencies)
            
            # Store in session state
            st.session_state.spectral_analyzer = spectral_analyzer
            st.session_state.frequency_data = frequency_data
            st.session_state.frequencies = frequencies
    
    if 'spectral_analyzer' in st.session_state:
        spectral_analyzer = st.session_state.spectral_analyzer
        frequencies = st.session_state.frequencies
        
        # Frequency explorer
        st.subheader("📊 Frequency Explorer")
        selected_freq_idx = st.selectbox("Select Frequency", range(len(frequencies)), 
                                       format_func=lambda i: f"{frequencies[i]} Hz",
                                       key="sd_freq_select")
        
        # Display frequency section based on type
        if section_type == "Time Slice":
            freq_data = spectral_analyzer.get_frequency_slice(selected_freq_idx, time_slice)
            title = f"Frequency Slice: {frequencies[selected_freq_idx]} Hz at {actual_time} ms"
        elif section_type == "Inline":
            freq_data = spectral_analyzer.get_frequency_inline(selected_freq_idx, inline_idx)
            title = f"Frequency Inline: {frequencies[selected_freq_idx]} Hz at Inline {inline_idx}"
        else:  # Crossline
            freq_data = spectral_analyzer.get_frequency_crossline(selected_freq_idx, crossline_idx)
            title = f"Frequency Crossline: {frequencies[selected_freq_idx]} Hz at Crossline {crossline_idx}"
        
        fig = px.imshow(freq_data.T if section_type != "Time Slice" else freq_data, 
                       title=title,
                       color_continuous_scale='viridis',
                       aspect='auto')
        fig.update_layout(coloraxis_colorbar=dict(title="Amplitude"))
        st.plotly_chart(fig, use_container_width=True)
        
        # RGB Blending
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
        
        # Weight controls for RGB blending
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
                elif section_type == "Inline":
                    rgb_blend = spectral_analyzer.create_rgb_inline(
                        low_freq_idx, mid_freq_idx, high_freq_idx,
                        inline_idx,
                        weights=(red_weight, green_weight, blue_weight)
                    )
                    title_suffix = f"at Inline {inline_idx}"
                else:  # Crossline
                    rgb_blend = spectral_analyzer.create_rgb_crossline(
                        low_freq_idx, mid_freq_idx, high_freq_idx,
                        crossline_idx,
                        weights=(red_weight, green_weight, blue_weight)
                    )
                    title_suffix = f"at Crossline {crossline_idx}"
                
                # Display RGB blend
                fig_rgb = px.imshow(rgb_blend.T if section_type != "Time Slice" else rgb_blend, 
                                  title=f"RGB Frequency Blend {title_suffix}\n"
                                        f"Low (R): {frequencies[low_freq_idx]}Hz, "
                                        f"Mid (G): {frequencies[mid_freq_idx]}Hz, "
                                        f"High (B): {frequencies[high_freq_idx]}Hz",
                                  aspect='auto')
                st.plotly_chart(fig_rgb, use_container_width=True)
                
                # Display individual components
                st.subheader("Individual Frequency Components")
                col1, col2, col3 = st.columns(3)
                
                def get_component_data(freq_idx, section_type, time_slice, inline_idx, crossline_idx):
                    if section_type == "Time Slice":
                        return spectral_analyzer.get_frequency_slice(freq_idx, time_slice)
                    elif section_type == "Inline":
                        return spectral_analyzer.get_frequency_inline(freq_idx, inline_idx)
                    else:
                        return spectral_analyzer.get_frequency_crossline(freq_idx, crossline_idx)
                
                with col1:
                    low_component = get_component_data(low_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    fig_low = px.imshow(low_component.T if section_type != "Time Slice" else low_component, 
                                      title=f"Low Freq: {frequencies[low_freq_idx]} Hz",
                                      color_continuous_scale='Reds',
                                      aspect='auto')
                    st.plotly_chart(fig_low, use_container_width=True)
                
                with col2:
                    mid_component = get_component_data(mid_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    fig_mid = px.imshow(mid_component.T if section_type != "Time Slice" else mid_component, 
                                      title=f"Mid Freq: {frequencies[mid_freq_idx]} Hz",
                                      color_continuous_scale='Greens',
                                      aspect='auto')
                    st.plotly_chart(fig_mid, use_container_width=True)
                
                with col3:
                    high_component = get_component_data(high_freq_idx, section_type, time_slice, inline_idx, crossline_idx)
                    fig_high = px.imshow(high_component.T if section_type != "Time Slice" else high_component, 
                                       title=f"High Freq: {frequencies[high_freq_idx]} Hz",
                                       color_continuous_scale='Blues',
                                       aspect='auto')
                    st.plotly_chart(fig_high, use_container_width=True)

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
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Bandwidth Enhancement", "🔬 Spectral Decomposition"])
    
    with tab1:
        display_bandwidth_enhancement_tab(st.session_state.enhancer)

    with tab2:
        display_spectral_decomposition_tab(st.session_state.enhancer)

if __name__ == "__main__":
    main()
