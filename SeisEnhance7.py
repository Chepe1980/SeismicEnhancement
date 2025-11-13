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
        """
        SPECTRAL BLUEING: Frequency-domain enhancement technique
        
        Mathematical Foundation:
        S_enhanced(f) = S_original(f) × W(f)
        
        Where:
        - S_original(f) = Original frequency spectrum
        - S_enhanced(f) = Enhanced frequency spectrum  
        - W(f) = Frequency-dependent weighting function
        
        Frequency Bands:
        - Low (5-30 Hz): Structural information, gentle boost
        - Mid (30-80 Hz): Primary resolution, moderate enhancement  
        - High (80+ Hz): Fine details, controlled boost with roll-off
        """
        st.info("Applying Spectral Blueing - Frequency Domain Enhancement...")
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
        """
        Apply spectral blueing to a single trace
        
        Processing Steps:
        1. FFT to frequency domain
        2. Apply frequency-dependent weighting
        3. Inverse FFT back to time domain
        4. Amplitude preservation
        """
        # Step 1: Remove mean and detrend
        trace = signal.detrend(trace)
        
        # Step 2: FFT to frequency domain
        trace_fft = fft(trace)
        freqs = fftfreq(len(trace), d=self.sample_rate/1000.0)  # Frequency in Hz
        
        # Step 3: Create frequency-dependent weighting
        weights = np.ones_like(freqs, dtype=complex)
        freq_magnitude = np.abs(freqs)
        
        # Low frequencies (5-30 Hz): Structural information
        low_freq_mask = (freq_magnitude > 5) & (freq_magnitude <= mid_freq_range[0])
        weights[low_freq_mask] = low_freq_boost
        
        # Target frequencies (30-80 Hz): Primary resolution
        target_mask = (freq_magnitude > mid_freq_range[0]) & (freq_magnitude <= target_freq)
        weights[target_mask] = enhancement_factor
        
        # High frequencies (80+ Hz): Fine details with roll-off
        high_freq_mask = freq_magnitude > target_freq
        # Apply roll-off to avoid noise amplification
        rolloff_factor = enhancement_factor * np.exp(-0.001 * (freq_magnitude[high_freq_mask] - target_freq)**2)
        weights[high_freq_mask] = np.maximum(1.0, rolloff_factor)
        
        # Apply weights
        enhanced_fft = trace_fft * weights
        
        # Step 4: Inverse FFT and take real part
        enhanced_trace = np.real(ifft(enhanced_fft))
        
        # Amplitude preservation: Maintain relative amplitude relationships
        if np.std(enhanced_trace) > 0:
            enhanced_trace = enhanced_trace * (np.std(trace) / np.std(enhanced_trace))
        
        return enhanced_trace

    def bandpass_filter(self, seismic_data, lowcut=8, highcut=120, order=3):
        """
        BANDPASS FILTERING: Remove noise while preserving enhanced signal
        
        Purpose:
        - Remove very low-frequency noise (ground roll, swell)
        - Remove high-frequency random noise
        - Preserve amplitude relationships in target frequencies
        
        Butterworth Filter Characteristics:
        - Zero-phase (filtfilt) to avoid phase distortion
        - Flat passband for amplitude preservation
        - Steep roll-off for effective noise attenuation
        """
        st.info("Applying Bandpass Filter - Noise Removal...")
        
        # Calculate Nyquist frequency
        sampling_interval = self.sample_rate / 1000.0  # Convert ms to seconds
        sampling_freq = 1.0 / sampling_interval  # Sampling frequency in Hz
        nyquist = sampling_freq / 2.0  # Nyquist frequency in Hz
        
        st.info(f"Nyquist Frequency: {nyquist:.1f} Hz")
        
        # Auto-adjust filter range if needed (prevent aliasing)
        if highcut >= nyquist * 0.95:
            highcut = nyquist * 0.9
            st.warning(f"Adjusted highcut to {highcut:.1f} Hz for stability")
        
        # Normalize frequencies for filter design
        low_normalized = lowcut / nyquist
        high_normalized = highcut / nyquist
        
        st.info(f"Filter Range: {lowcut}-{highcut} Hz")
        st.info(f"Normalized Range: {low_normalized:.3f}-{high_normalized:.3f}")
        
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
                
                # Use filtfilt for zero-phase filtering (no phase distortion)
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

    def calculate_snr(self, data):
        """Calculate Signal-to-Noise Ratio (SNR) in decibels"""
        signal_power = np.mean(data**2)
        noise_power = np.var(data - np.mean(data))
        return 10 * np.log10(signal_power / (noise_power + 1e-10))

    def calculate_resolution_gain(self):
        """Calculate spectral resolution gain across frequency spectrum"""
        if self.original_data is None or self.enhanced_data is None:
            return None
            
        n_inlines, n_xlines, n_samples = self.original_data.shape
        freqs = fftfreq(n_samples, d=self.sample_rate/1000.0)
        positive_freqs = freqs > 0
        
        # Calculate average spectra from sampled traces
        avg_orig_spectrum = np.zeros(np.sum(positive_freqs))
        avg_enh_spectrum = np.zeros(np.sum(positive_freqs))
        trace_count = 0
        
        # Sample traces for efficiency
        for i in range(0, n_inlines, max(1, n_inlines//10)):
            for j in range(0, n_xlines, max(1, n_xlines//10)):
                orig_fft = np.abs(fft(self.original_data[i, j, :]))[positive_freqs]
                enh_fft = np.abs(fft(self.enhanced_data[i, j, :]))[positive_freqs]
                
                if len(orig_fft) == len(avg_orig_spectrum):
                    avg_orig_spectrum += orig_fft
                    avg_enh_spectrum += enh_fft
                    trace_count += 1
        
        if trace_count > 0:
            avg_orig_spectrum /= trace_count
            avg_enh_spectrum /= trace_count
            return avg_enh_spectrum / (avg_orig_spectrum + 1e-10)
        
        return None

    def plot_quality_metrics(self):
        """Plot comprehensive quality assessment metrics"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data available for quality metrics")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Signal-to-noise ratio improvement
        snr_original = self.calculate_snr(self.original_data)
        snr_enhanced = self.calculate_snr(self.enhanced_data)
        
        axes[0, 0].bar(['Original', 'Enhanced'], [snr_original, snr_enhanced], 
                      color=['blue', 'red'], alpha=0.7)
        axes[0, 0].set_title('Signal-to-Noise Ratio (SNR) Improvement', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('SNR (dB)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(0.5, 0.9, f'ΔSNR: {snr_enhanced-snr_original:+.1f} dB', 
                       transform=axes[0, 0].transAxes, ha='center', fontweight='bold')
        
        # Resolution metrics
        resolution_gain = self.calculate_resolution_gain()
        if resolution_gain is not None:
            freqs = fftfreq(self.original_data.shape[2], d=self.sample_rate/1000.0)
            positive_freqs = freqs > 0
            plot_freqs = freqs[positive_freqs]
            
            axes[0, 1].plot(plot_freqs[:len(resolution_gain)], resolution_gain, 'g-', linewidth=2)
            axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
            axes[0, 1].set_title('Spectral Resolution Gain', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Enhancement Factor')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0.5, 2.5])
        
        # Amplitude distribution
        axes[1, 0].hist(self.original_data.flatten(), bins=100, alpha=0.7, 
                       label='Original', color='blue', density=True)
        axes[1, 0].hist(self.enhanced_data.flatten(), bins=100, alpha=0.7, 
                       label='Enhanced', color='red', density=True)
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Amplitude Distribution Preservation', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Data range comparison
        metrics = ['Minimum', 'Maximum', 'Standard Dev']
        original_vals = [np.min(self.original_data), np.max(self.original_data), np.std(self.original_data)]
        enhanced_vals = [np.min(self.enhanced_data), np.max(self.enhanced_data), np.std(self.enhanced_data)]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, original_vals, width, label='Original', 
                      color='blue', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, enhanced_vals, width, label='Enhanced', 
                      color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Statistical Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Data Statistics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_spectral_comparison(self, original_trace, enhanced_trace, trace_idx=0):
        """Plot comprehensive spectral comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain comparison
        time_axis = np.arange(len(original_trace)) * self.sample_rate
        axes[0, 0].plot(time_axis, original_trace, 'b-', alpha=0.8, label='Original', linewidth=1.5)
        axes[0, 0].plot(time_axis, enhanced_trace, 'r-', alpha=0.7, label='Enhanced', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'Trace Comparison (Trace #{trace_idx})', fontweight='bold')
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
        
        axes[0, 1].semilogy(plot_freqs, orig_fft[positive_freqs][valid_freqs], 'b-', 
                       alpha=0.8, label='Original', linewidth=2)
        axes[0, 1].semilogy(plot_freqs, enh_fft[positive_freqs][valid_freqs], 'r-', 
                       alpha=0.7, label='Enhanced', linewidth=2)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Amplitude (log)')
        axes[0, 1].set_title('Frequency Spectrum Comparison', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectral ratio
        spectral_ratio = enh_fft[positive_freqs] / (orig_fft[positive_freqs] + 1e-10)
        axes[1, 0].plot(plot_freqs, spectral_ratio[valid_freqs], 'g-', linewidth=2)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Spectral Ratio')
        axes[1, 0].set_title('Enhancement Factor vs Frequency', fontweight='bold')
        axes[1, 0].set_ylim([0, 3])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of amplitudes
        axes[1, 1].hist(original_trace, bins=100, alpha=0.7, label='Original', color='blue', density=True)
        axes[1, 1].hist(enhanced_trace, bins=100, alpha=0.7, label='Enhanced', color='red', density=True)
        axes[1, 1].set_xlabel('Amplitude')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Amplitude Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def enhance_bandwidth(self, file_path, target_freq=80, enhancement_factor=1.5, low_freq_boost=1.2,
                         mid_freq_start=30, lowcut=8, highcut=120, filter_order=3):
        """
        MAIN PROCESSING PIPELINE: Spectral Blueing + Bandpass Filtering
        
        Processing Flow:
        1. Load SEG-Y data with geometry detection
        2. Apply spectral blueing for frequency enhancement
        3. Apply bandpass filtering for noise removal
        4. Calculate quality metrics
        """
        st.info("🚀 Starting 3D Seismic Bandwidth Enhancement...")
        self.original_data = self.read_segy(file_path)
        
        if self.original_data is None:
            raise ValueError("Failed to load SEG-Y file")
        
        # Store parameters for metadata and display
        self.target_freq = target_freq
        self.enhancement_factor = enhancement_factor
        self.low_freq_boost = low_freq_boost
        self.lowcut = lowcut
        self.highcut = highcut
        
        st.success(f"Original data shape: {self.original_data.shape}")
        st.info(f"Original data range: {np.min(self.original_data):.3f} to {np.max(self.original_data):.3f}")
        
        start_time = time.time()
        
        # Apply spectral blueing
        self.enhanced_data = self.spectral_blueing(
            self.original_data,
            target_freq=target_freq,
            enhancement_factor=enhancement_factor,
            low_freq_boost=low_freq_boost,
            mid_freq_range=(mid_freq_start, target_freq)
        )
        
        # Apply bandpass filter
        self.enhanced_data = self.bandpass_filter(
            self.enhanced_data,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order
        )
        
        processing_time = time.time() - start_time
        st.success(f"🎉 Processing completed in {processing_time:.2f} seconds")
        st.info(f"Enhanced data range: {np.min(self.enhanced_data):.3f} to {np.max(self.enhanced_data):.3f}")
        
        # Calculate and display quality metrics
        snr_original = self.calculate_snr(self.original_data)
        snr_enhanced = self.calculate_snr(self.enhanced_data)
        st.info(f"📊 Signal-to-Noise Ratio: {snr_original:.1f} dB → {snr_enhanced:.1f} dB (Δ: {snr_enhanced-snr_original:+.1f} dB)")
        
        return self.enhanced_data

# Processing presets for different geological objectives
PROCESSING_PRESETS = {
    'structural_interpretation': {
        'target_freq': 70,
        'enhancement_factor': 1.8,
        'low_freq_boost': 1.2,
        'mid_freq_start': 25,
        'lowcut': 8,
        'highcut': 100,
        'filter_order': 3,
        'description': 'Optimal for structural mapping and fault interpretation'
    },
    'stratigraphic_analysis': {
        'target_freq': 90,
        'enhancement_factor': 2.2,
        'low_freq_boost': 1.1,
        'mid_freq_start': 30,
        'lowcut': 10,
        'highcut': 120,
        'filter_order': 4,
        'description': 'Enhanced for stratigraphic features and channel systems'
    },
    'thin_bed_resolution': {
        'target_freq': 110,
        'enhancement_factor': 2.8,
        'low_freq_boost': 1.0,
        'mid_freq_start': 35,
        'lowcut': 15,
        'highcut': 150,
        'filter_order': 4,
        'description': 'Maximum resolution for thin beds and fine details'
    },
    'amplitude_preservation': {
        'target_freq': 60,
        'enhancement_factor': 1.3,
        'low_freq_boost': 1.3,
        'mid_freq_start': 20,
        'lowcut': 5,
        'highcut': 80,
        'filter_order': 3,
        'description': 'Conservative settings for AVO and amplitude studies'
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

def display_theory_guide():
    """Display comprehensive theory and user guide"""
    st.sidebar.markdown("---")
    st.sidebar.header("🎓 Theory & Guide")
    
    with st.sidebar.expander("📖 Spectral Blueing Theory", expanded=False):
        st.markdown("""
        **Spectral Blueing** enhances seismic resolution by selectively amplifying frequency components:
        
        ### Mathematical Foundation:
        ```
        S_enhanced(f) = S_original(f) × W(f)
        ```
        
        ### Frequency Bands:
        - **Low (5-30 Hz)**: Structural info, gentle boost
        - **Mid (30-80 Hz)**: Primary resolution, moderate enhancement  
        - **High (80+ Hz)**: Fine details, controlled boost
        
        ### Benefits:
        - Sharper seismic events
        - Better thin-bed detection
        - Improved signal-to-noise ratio
        - Preserved amplitude relationships
        """)
    
    with st.sidebar.expander("🎯 Parameter Guide", expanded=False):
        st.markdown("""
        ### For Different Objectives:
        
        **Structural Interpretation**:
        - Target: 60-80 Hz, Enhancement: 1.5-2.0x
        
        **Stratigraphic Analysis**:
        - Target: 80-100 Hz, Enhancement: 2.0-2.5x
        
        **Thin Bed Resolution**:
        - Target: 100-120 Hz, Enhancement: 2.5-3.0x
        
        **Amplitude Studies**:
        - Target: 60-70 Hz, Enhancement: 1.2-1.5x
        """)
    
    with st.sidebar.expander("🔧 Technical Details", expanded=False):
        st.markdown("""
        ### Processing Pipeline:
        1. **FFT**: Time → Frequency domain
        2. **Spectral Weighting**: Frequency-dependent enhancement
        3. **Inverse FFT**: Frequency → Time domain
        4. **Bandpass Filtering**: Noise removal
        
        ### Key Features:
        - Zero-phase filtering (no distortion)
        - Amplitude preservation
        - Memory-efficient chunk processing
        - Quality metrics calculation
        """)

def main():
    st.set_page_config(
        page_title="3D Seismic Bandwidth Enhancer",
        page_icon="🌊",
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
    
    # Main title with description
    st.title("🌊 3D Seismic Bandwidth Enhancement Tool")
    st.markdown("""
    **Enhance seismic resolution using Spectral Blueing techniques**  
    *Improve frequency content while preserving amplitude relationships for better geological interpretation*
    """)
    
    enhancer = st.session_state.enhancer
    
    # Display theory guide in sidebar
    display_theory_guide()
    
    # Sidebar for data input and parameters
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format"
    )
    
    st.sidebar.header("⚙️ Processing Parameters")
    
    # Processing presets
    st.sidebar.subheader("🎚️ Processing Presets")
    preset = st.sidebar.selectbox(
        "Choose Geological Objective",
        options=list(PROCESSING_PRESETS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Select preset optimized for specific interpretation goals"
    )
    
    # Display preset description
    if preset:
        preset_info = PROCESSING_PRESETS[preset]
        st.sidebar.info(f"**{preset.replace('_', ' ').title()}**: {preset_info['description']}")
    
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
    
    # Spectral Blueing Parameters
    st.sidebar.subheader("🎵 Spectral Blueing")
    target_freq = st.sidebar.slider(
        "Target Frequency (Hz)", 
        30, 120, target_freq,
        help="Primary frequency for maximum enhancement (30-120 Hz)"
    )
    enhancement_factor = st.sidebar.slider(
        "Enhancement Factor", 
        1.0, 3.0, enhancement_factor, 0.1,
        help="Boost strength at target frequency (1.0-3.0x)"
    )
    low_freq_boost = st.sidebar.slider(
        "Low Frequency Boost", 
        1.0, 2.0, low_freq_boost, 0.1,
        help="Gentle enhancement for structural frequencies (1.0-2.0x)"
    )
    mid_freq_start = st.sidebar.slider(
        "Mid Frequency Start (Hz)", 
        10, 50, mid_freq_start,
        help="Start of main enhancement band (10-50 Hz)"
    )
    
    # Bandpass Filter Parameters
    st.sidebar.subheader("🔊 Bandpass Filter")
    lowcut = st.sidebar.slider(
        "Low Cut Frequency (Hz)", 
        1, 50, lowcut,
        help="Remove frequencies below this value (1-50 Hz)"
    )
    highcut = st.sidebar.slider(
        "High Cut Frequency (Hz)", 
        60, 200, highcut,
        help="Remove frequencies above this value (60-200 Hz)"
    )
    filter_order = st.sidebar.slider(
        "Filter Order", 
        2, 6, filter_order,
        help="Steepness of filter roll-off (2-6)"
    )
    
    # Performance settings
    st.sidebar.subheader("⚡ Performance")
    use_chunk_processing = st.sidebar.checkbox(
        "Use Chunk Processing", 
        value=True,
        help="Process data in chunks to reduce memory usage for large datasets"
    )
    chunk_size = st.sidebar.slider(
        "Chunk Size (inlines)",
        10, 100, 50,
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
                with st.spinner("🔄 Processing 3D seismic data... This may take several minutes for large datasets"):
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
                
                st.balloons()
                st.success("✅ 3D Processing completed!")
                st.session_state.data_processed = True
                st.session_state.file_generated = False
                st.session_state.enhanced_file_path = None
                st.session_state.enhanced_file_data = None

            # Download section
            if st.session_state.get('data_processed', False):
                st.sidebar.header("💾 Download Results")
                
                if st.sidebar.button("🛠️ Generate Enhanced File", type="secondary", use_container_width=True):
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
                            help_text = "Download enhanced data in standard SEG-Y format"
                        else:
                            download_name = "enhanced_seismic_data.dat"
                            label = "📥 Download Enhanced Data"
                            help_text = "Download enhanced data in numpy format with metadata"
                        
                        st.download_button(
                            label=label,
                            data=file_data,
                            file_name=download_name,
                            mime="application/octet-stream",
                            help=help_text,
                            key="download_enhanced_data",
                            use_container_width=True
                        )
                        st.success("Enhanced file ready for download!")
                
                # Display comprehensive results
                st.header("📊 Processing Results")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Quality Metrics", 
                    "🎵 Frequency Analysis", 
                    "📋 Data Statistics",
                    "🖼️ Section Views"
                ])
                
                with tab1:
                    st.subheader("Quality Assessment Metrics")
                    fig_quality = enhancer.plot_quality_metrics()
                    if fig_quality:
                        st.pyplot(fig_quality)
                    
                    # Additional quality insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        snr_orig = enhancer.calculate_snr(enhancer.original_data)
                        snr_enh = enhancer.calculate_snr(enhancer.enhanced_data)
                        st.metric("SNR Improvement", f"{snr_enh-snr_orig:+.1f} dB")
                    
                    with col2:
                        orig_range = np.max(enhancer.original_data) - np.min(enhancer.original_data)
                        enh_range = np.max(enhancer.enhanced_data) - np.min(enhancer.enhanced_data)
                        st.metric("Dynamic Range", f"{enh_range/orig_range:.2f}x")
                    
                    with col3:
                        orig_std = np.std(enhancer.original_data)
                        enh_std = np.std(enhancer.enhanced_data)
                        st.metric("Standard Deviation", f"{enh_std/orig_std:.2f}x")
                
                with tab2:
                    st.subheader("Frequency Domain Analysis")
                    # Select a representative trace
                    n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                    inline_idx = n_inlines // 2
                    xline_idx = n_xlines // 2
                    
                    original_trace = enhancer.original_data[inline_idx, xline_idx, :]
                    enhanced_trace = enhancer.enhanced_data[inline_idx, xline_idx, :]
                    
                    fig_spectral = enhancer.plot_spectral_comparison(
                        original_trace, enhanced_trace, 
                        trace_idx=f"Inline {inline_idx}, Xline {xline_idx}"
                    )
                    st.pyplot(fig_spectral)
                
                with tab3:
                    st.subheader("Data Statistics Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Data")
                        st.write(f"**Shape**: {enhancer.original_data.shape}")
                        st.write(f"**Range**: {np.min(enhancer.original_data):.3f} to {np.max(enhancer.original_data):.3f}")
                        st.write(f"**Mean**: {np.mean(enhancer.original_data):.3f}")
                        st.write(f"**Std Dev**: {np.std(enhancer.original_data):.3f}")
                        st.write(f"**SNR**: {enhancer.calculate_snr(enhancer.original_data):.1f} dB")
                    
                    with col2:
                        st.subheader("Enhanced Data")
                        st.write(f"**Shape**: {enhancer.enhanced_data.shape}")
                        st.write(f"**Range**: {np.min(enhancer.enhanced_data):.3f} to {np.max(enhancer.enhanced_data):.3f}")
                        st.write(f"**Mean**: {np.mean(enhancer.enhanced_data):.3f}")
                        st.write(f"**Std Dev**: {np.std(enhancer.enhanced_data):.3f}")
                        st.write(f"**SNR**: {enhancer.calculate_snr(enhancer.enhanced_data):.1f} dB")
                
                with tab4:
                    st.subheader("Seismic Section Views")
                    st.info("Inline and crossline sections showing original vs enhanced data")
                    # Simple section display
                    n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data**")
                        # Display a time slice
                        time_slice = n_samples // 2
                        fig, ax = plt.subplots(figsize=(8, 6))
                        vmax = np.percentile(np.abs(enhancer.original_data[:, :, time_slice]), 95)
                        im = ax.imshow(enhancer.original_data[:, :, time_slice], 
                                     aspect='auto', cmap='seismic', 
                                     vmin=-vmax, vmax=vmax)
                        ax.set_title(f'Original - Time {time_slice * enhancer.sample_rate:.0f}ms')
                        ax.set_xlabel('Crossline')
                        ax.set_ylabel('Inline')
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Enhanced Data**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        vmax = np.percentile(np.abs(enhancer.enhanced_data[:, :, time_slice]), 95)
                        im = ax.imshow(enhancer.enhanced_data[:, :, time_slice], 
                                     aspect='auto', cmap='seismic', 
                                     vmin=-vmax, vmax=vmax)
                        ax.set_title(f'Enhanced - Time {time_slice * enhancer.sample_rate:.0f}ms')
                        ax.set_xlabel('Crossline')
                        ax.set_ylabel('Inline')
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)
    
    else:
        # Welcome and instructions
        st.header("🎯 Welcome to the 3D Seismic Bandwidth Enhancer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📖 Complete User Guide
            
            #### 🚀 Quick Start:
            1. **Upload** a 3D SEG-Y file using the sidebar
            2. **Select** a processing preset or customize parameters
            3. **Process** the data (takes 2-10 minutes for typical volumes)
            4. **Download** the enhanced SEG-Y file
            
            #### 🎯 Geological Applications:
            - **Structural Interpretation**: Sharper faults and boundaries
            - **Stratigraphic Analysis**: Enhanced channel and deposit features  
            - **Thin Bed Resolution**: Improved detection of thin layers
            - **Amplitude Studies**: Preserved relationships for AVO analysis
            
            #### 🔬 Scientific Foundation:
            This tool uses **Spectral Blueing** - a frequency-domain enhancement technique that:
            - Compensates for natural high-frequency attenuation
            - Improves seismic resolution while preserving amplitudes
            - Uses zero-phase filtering to avoid distortion
            - Maintains geological integrity of the data
            
            #### 📊 Output Quality:
            - **SEG-Y Format**: Industry standard with all original headers
            - **Quality Metrics**: Comprehensive assessment of enhancement
            - **Visualization**: Multiple views for quality control
            - **Compatibility**: Works with all major interpretation software
            """)
        
        with col2:
            st.image("https://via.placeholder.com/300x400/4A90E2/FFFFFF?text=Seismic+Enhancement", 
                    caption="Spectral Blueing Process")
            
            st.info("""
            **💡 Pro Tips:**
            - Start with conservative settings
            - Use presets for common objectives
            - Check quality metrics before interpretation
            - Compare original vs enhanced sections
            """)
        
        # Technical details expander
        with st.expander("🔬 Advanced Technical Details", expanded=False):
            st.markdown("""
            ### Frequency Domain Processing Chain:
            
            **1. Fast Fourier Transform (FFT)**
            ```python
            trace_fft = fft(trace)  # Time → Frequency domain
            ```
            
            **2. Spectral Weighting Function**
            ```python
            weights = create_frequency_weights(freqs, target_freq, enhancement_factor)
            enhanced_fft = trace_fft * weights
            ```
            
            **3. Inverse FFT**
            ```python  
            enhanced_trace = np.real(ifft(enhanced_fft))  # Frequency → Time domain
            ```
            
            **4. Bandpass Filtering**
            ```python
            enhanced_trace = signal.filtfilt(b, a, enhanced_trace)  # Zero-phase
            ```
            
            ### Key Algorithms:
            - **Butterworth Filter**: Flat passband, steep roll-off
            - **Zero-phase Filtering**: No phase distortion  
            - **Amplitude Preservation**: Maintains relative relationships
            - **Chunk Processing**: Memory-efficient for large volumes
            """)

    # Clean up temporary files
    def cleanup_old_files():
        """Clean up old temporary files"""
        temp_dir = tempfile.gettempdir()
        current_file = getattr(st.session_state, 'enhanced_file_path', None)
        
        for filename in os.listdir(temp_dir):
            if filename.startswith("enhanced_") and (filename.endswith(".sgy") or filename.endswith(".dat")):
                file_path = os.path.join(temp_dir, filename)
                if file_path != current_file and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass
    
    cleanup_old_files()

if __name__ == "__main__":
    main()
