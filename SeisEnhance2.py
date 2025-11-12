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

class SeismicBandwidthEnhancer:
    def __init__(self):
        self.original_data = None
        self.enhanced_data = None
        self.sample_rate = 4.0  # Default 4ms, adjust if needed
        self.geometry = None  # Store geometry information
        self.original_segyfile = None  # Store original segy file reference
        
    def read_segy_3d(self, filename):
        """Read 3D SEG-Y file and return seismic data as numpy array"""
        try:
            with segyio.open(filename, "r") as segyfile:
                self.original_segyfile = segyfile  # Store for later use
                
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

    def read_segy_alternative(self, filename):
        """Alternative method to read SEG-Y files"""
        with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
            n_traces = segyfile.tracecount
            n_samples = segyfile.samples.size
            
            # Try to determine if it's 3D by checking trace headers
            try:
                ilines = []
                xlines = []
                for i in range(min(n_traces, 1000)):  # Sample first 1000 traces
                    ilines.append(segyfile.header[i][segyio.TraceField.INLINE_3D])
                    xlines.append(segyfile.header[i][segyio.TraceField.CROSSLINE_3D])
                
                unique_ilines = np.unique(ilines)
                unique_xlines = np.unique(xlines)
                
                if len(unique_ilines) > 1 and len(unique_xlines) > 1:
                    # Likely 3D data
                    st.info("3D data detected via trace headers")
                    # Create approximate 3D structure
                    data = np.zeros((len(unique_ilines), len(unique_xlines), n_samples))
                    # This is simplified - in practice you'd need proper mapping
                    for i in range(n_traces):
                        inline_idx = np.where(unique_ilines == ilines[i])[0][0]
                        xline_idx = np.where(unique_xlines == xlines[i])[0][0]
                        data[inline_idx, xline_idx, :] = segyfile.trace[i]
                else:
                    # Treat as 2D
                    data = np.zeros((1, n_traces, n_samples))
                    for i in range(n_traces):
                        data[0, i, :] = segyfile.trace[i]
                        
            except:
                # Fallback to 2D
                data = np.zeros((1, n_traces, n_samples))
                for i in range(n_traces):
                    data[0, i, :] = segyfile.trace[i]
            
            st.info(f"Alternative read - Shape: {data.shape}")
            return data

    def write_segy(self, data, original_filename, output_filename):
        """Write enhanced data back to SEG-Y file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
            
            # Copy the original SEG-Y structure and write new data
            with segyio.open(original_filename, "r") as src:
                # Create spec based on original file
                spec = segyio.spec()
                spec.format = src.format
                spec.samples = src.samples
                spec.tracecount = src.tracecount
                
                # For 3D data, preserve geometry
                try:
                    if hasattr(src, 'ilines') and src.ilines is not None:
                        spec.ilines = src.ilines
                    if hasattr(src, 'xlines') and src.xlines is not None:
                        spec.xlines = src.xlines
                except:
                    pass  # If geometry doesn't exist, continue without it
                
                with segyio.open(output_filename, "w", spec) as dst:
                    # Copy textual headers
                    dst.text[0] = src.text[0]
                    if hasattr(src, 'text') and len(src.text) > 1:
                        for i in range(1, len(src.text)):
                            dst.text[i] = src.text[i]
                    
                    # Copy binary header
                    dst.bin = src.bin
                    
                    # Copy trace headers and write new traces
                    for i in range(src.tracecount):
                        dst.header[i] = src.header[i]
                        
                        # Map the enhanced data back to trace order
                        if data.shape[0] == 1:  # 2D data
                            if i < data.shape[1]:
                                dst.trace[i] = data[0, i, :]
                            else:
                                # If trace count doesn't match, use first trace pattern
                                dst.trace[i] = data[0, 0, :]
                        else:  # 3D data
                            # For 3D data, we need to map back to original geometry
                            inline_idx, xline_idx = self._get_trace_position(i, src)
                            if (inline_idx < data.shape[0] and 
                                xline_idx < data.shape[1] and 
                                inline_idx >= 0 and xline_idx >= 0):
                                dst.trace[i] = data[inline_idx, xline_idx, :]
                            else:
                                # Fallback: use first trace pattern
                                dst.trace[i] = data[0, 0, :]
                        
            st.success(f"Enhanced SEG-Y file saved: {output_filename}")
            return True
            
        except Exception as e:
            st.error(f"Error writing SEG-Y file: {e}")
            # Try alternative writing method
            return self.write_segy_alternative(data, original_filename, output_filename)

    def _get_trace_position(self, trace_idx, segyfile):
        """Get inline and crossline position for a trace index"""
        try:
            # For 3D data with proper geometry
            if hasattr(segyfile, 'ilines') and hasattr(segyfile, 'xlines'):
                if segyfile.ilines is not None and segyfile.xlines is not None:
                    n_xlines = len(segyfile.xlines)
                    inline_idx = trace_idx // n_xlines
                    xline_idx = trace_idx % n_xlines
                    return inline_idx, xline_idx
        except:
            pass
        
        # Fallback for 2D or improperly structured data
        return 0, trace_idx

    def write_segy_alternative(self, data, original_filename, output_filename):
        """Alternative SEG-Y writing method - simpler approach"""
        try:
            with segyio.open(original_filename, "r", ignore_geometry=True) as src:
                # Get basic specifications from source
                n_traces = src.tracecount
                n_samples = len(src.samples)
                
                # Create output spec
                spec = segyio.spec()
                spec.format = src.format
                spec.samples = src.samples
                spec.tracecount = n_traces
                
                with segyio.open(output_filename, "w", spec) as dst:
                    # Copy headers
                    dst.text[0] = src.text[0]
                    dst.bin = src.bin
                    
                    # Write traces
                    for i in range(n_traces):
                        # Copy trace header
                        dst.header[i] = src.header[i]
                        
                        # Write trace data
                        if data.shape[0] == 1:  # 2D data
                            if i < data.shape[1]:
                                dst.trace[i] = data[0, i, :]
                            else:
                                dst.trace[i] = np.zeros(n_samples)  # Fallback
                        else:  # 3D data
                            # Flatten 3D data and take in order
                            flat_data = data.reshape(-1, data.shape[2])
                            if i < flat_data.shape[0]:
                                dst.trace[i] = flat_data[i, :]
                            else:
                                dst.trace[i] = np.zeros(n_samples)  # Fallback
                            
            st.success(f"Enhanced SEG-Y file saved (alternative method): {output_filename}")
            return True
        except Exception as e:
            st.error(f"Alternative writing failed: {e}")
            return False

    def create_downloadable_segy(self, original_filename, output_filename):
        """Create enhanced SEG-Y file and return the file path for download"""
        if self.enhanced_data is None:
            st.error("No enhanced data available. Please process the data first.")
            return None
        
        try:
            # Create temporary file for download
            temp_dir = tempfile.gettempdir()
            download_filename = os.path.join(temp_dir, output_filename)
            
            # Write the enhanced data to SEG-Y file
            st.info("Creating enhanced SEG-Y file for download...")
            success = self.write_segy(self.enhanced_data, original_filename, download_filename)
            
            if success:
                # Verify the file was created
                if os.path.exists(download_filename) and os.path.getsize(download_filename) > 0:
                    st.success("Enhanced SEG-Y file created successfully!")
                    return download_filename
                else:
                    st.error("Enhanced SEG-Y file was not created properly")
                    return None
            else:
                return None
                
        except Exception as e:
            st.error(f"Error creating downloadable SEG-Y: {e}")
            return None

    # [Keep all the other methods exactly the same as before: spectral_blueing, bandpass_filter, 
    # plot_spectral_comparison, plot_seismic_section, plot_3d_volume, plot_time_slice_comparison, 
    # plot_frequency_analysis, enhance_bandwidth - they remain unchanged]

    def spectral_blueing(self, seismic_data, target_freq=80, enhancement_factor=1.5,
                        low_freq_boost=1.2, mid_freq_range=(30, 80)):
        """Spectral blueing to enhance high frequencies - unchanged"""
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
        """Apply bandpass filter - unchanged"""
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
        """Plot comparison between original and enhanced spectra - unchanged"""
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

    def plot_seismic_section(self, inline_idx=None, xline_idx=None):
        """Plot a section of original vs enhanced seismic - unchanged"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
        
        n_inlines, n_xlines, n_samples = self.original_data.shape
        
        # Set default indices if not provided
        if inline_idx is None:
            inline_idx = n_inlines // 2
        if xline_idx is None:
            xline_idx = n_xlines // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Inline section
        vmax_inline = np.percentile(np.abs(self.original_data[inline_idx, :, :]), 95)
        axes[0, 0].imshow(self.original_data[inline_idx, :, :].T, 
                         aspect='auto', cmap='seismic', vmin=-vmax_inline, vmax=vmax_inline)
        axes[0, 0].set_title(f'Original - Inline {inline_idx}')
        axes[0, 0].set_xlabel('Crossline')
        axes[0, 0].set_ylabel('Time Sample')
        
        axes[1, 0].imshow(self.enhanced_data[inline_idx, :, :].T, 
                         aspect='auto', cmap='seismic', vmin=-vmax_inline, vmax=vmax_inline)
        axes[1, 0].set_title(f'Enhanced - Inline {inline_idx}')
        axes[1, 0].set_xlabel('Crossline')
        axes[1, 0].set_ylabel('Time Sample')
        
        # Crossline section
        vmax_xline = np.percentile(np.abs(self.original_data[:, xline_idx, :]), 95)
        axes[0, 1].imshow(self.original_data[:, xline_idx, :].T, 
                         aspect='auto', cmap='seismic', vmin=-vmax_xline, vmax=vmax_xline)
        axes[0, 1].set_title(f'Original - Crossline {xline_idx}')
        axes[0, 1].set_xlabel('Inline')
        axes[0, 1].set_ylabel('Time Sample')
        
        axes[1, 1].imshow(self.enhanced_data[:, xline_idx, :].T, 
                         aspect='auto', cmap='seismic', vmin=-vmax_xline, vmax=vmax_xline)
        axes[1, 1].set_title(f'Enhanced - Crossline {xline_idx}')
        axes[1, 1].set_xlabel('Inline')
        axes[1, 1].set_ylabel('Time Sample')
        
        # Time slice
        time_slice_idx = n_samples // 2
        vmax_time = np.percentile(np.abs(self.original_data[:, :, time_slice_idx]), 95)
        axes[0, 2].imshow(self.original_data[:, :, time_slice_idx], 
                         aspect='auto', cmap='seismic', vmin=-vmax_time, vmax=vmax_time)
        axes[0, 2].set_title(f'Original - Time {time_slice_idx * self.sample_rate:.0f}ms')
        axes[0, 2].set_xlabel('Crossline')
        axes[0, 2].set_ylabel('Inline')
        
        axes[1, 2].imshow(self.enhanced_data[:, :, time_slice_idx], 
                         aspect='auto', cmap='seismic', vmin=-vmax_time, vmax=vmax_time)
        axes[1, 2].set_title(f'Enhanced - Time {time_slice_idx * self.sample_rate:.0f}ms')
        axes[1, 2].set_xlabel('Crossline')
        axes[1, 2].set_ylabel('Inline')
        
        plt.tight_layout()
        return fig

    def plot_3d_volume(self, data_type='original', max_voxels=50000):
        """Create interactive 3D volume plot using Plotly - unchanged"""
        if self.original_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
        
        data = self.original_data if data_type == 'original' else self.enhanced_data
        
        # Downsample data if too large for performance
        n_inlines, n_xlines, n_samples = data.shape
        total_voxels = n_inlines * n_xlines * n_samples
        
        if total_voxels > max_voxels:
            downsample_factor = int(np.ceil((total_voxels / max_voxels) ** (1/3)))
            data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
            st.info(f"Downsampled data for 3D visualization (factor: {downsample_factor})")
        
        # Create coordinates
        n_inlines, n_xlines, n_samples = data.shape
        ilines = np.arange(n_inlines)
        xlines = np.arange(n_xlines)
        times = np.arange(n_samples) * self.sample_rate
        
        # Create meshgrid
        I, X, T = np.meshgrid(ilines, xlines, times, indexing='ij')
        
        # Flatten arrays
        I_flat = I.flatten()
        X_flat = X.flatten()
        T_flat = T.flatten()
        values_flat = data.flatten()
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Volume(
            x=I_flat,
            y=X_flat,
            z=T_flat,
            value=values_flat,
            isomin=np.percentile(values_flat, 10),
            isomax=np.percentile(values_flat, 90),
            opacity=0.1,
            surface_count=20,
            colorscale='RdBu_r',
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        
        fig.update_layout(
            title=f'3D Seismic Volume - {data_type.capitalize()}',
            scene=dict(
                xaxis_title='Inline',
                yaxis_title='Crossline',
                zaxis_title='Time (ms)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig

    def plot_time_slice_comparison(self, time_slice_idx=None):
        """Plot time slice comparison using Plotly - unchanged"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
        
        n_inlines, n_xlines, n_samples = self.original_data.shape
        
        if time_slice_idx is None:
            time_slice_idx = n_samples // 2
        
        time_ms = time_slice_idx * self.sample_rate
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Original - {time_ms:.0f}ms', f'Enhanced - {time_ms:.0f}ms'),
            horizontal_spacing=0.1
        )
        
        # Original data
        fig.add_trace(
            go.Heatmap(
                z=self.original_data[:, :, time_slice_idx],
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(x=0.45, len=0.4)
            ),
            row=1, col=1
        )
        
        # Enhanced data
        fig.add_trace(
            go.Heatmap(
                z=self.enhanced_data[:, :, time_slice_idx],
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(x=1.0, len=0.4)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Time Slice Comparison at {time_ms:.0f} ms",
            width=800,
            height=400
        )
        
        fig.update_xaxes(title_text="Crossline", row=1, col=1)
        fig.update_xaxes(title_text="Crossline", row=1, col=2)
        fig.update_yaxes(title_text="Inline", row=1, col=1)
        fig.update_yaxes(title_text="Inline", row=1, col=2)
        
        return fig

    def plot_frequency_analysis(self, num_traces=10):
        """Plot frequency content analysis for multiple traces - unchanged"""
        if self.original_data is None or self.enhanced_data is None:
            st.error("No data to plot. Run enhancement first.")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Select random traces for analysis
        n_inlines, n_xlines, n_samples = self.original_data.shape
        trace_indices = []
        for _ in range(min(num_traces, n_inlines * n_xlines)):
            i = np.random.randint(0, n_inlines)
            j = np.random.randint(0, n_xlines)
            trace_indices.append((i, j))
        
        # Calculate average frequency spectra
        avg_orig_spectrum = np.zeros(n_samples // 2)
        avg_enh_spectrum = np.zeros(n_samples // 2)
        
        freqs = fftfreq(n_samples, d=self.sample_rate/1000.0)
        positive_freqs = freqs > 0
        
        for i, j in trace_indices:
            orig_trace = self.original_data[i, j, :]
            enh_trace = self.enhanced_data[i, j, :]
            
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
        """Main method to enhance seismic bandwidth - unchanged"""
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
    
    st.title("🌊 3D Seismic Bandwidth Enhancement Tool")
    st.markdown("""
    Enhance the frequency content of your 3D seismic data using spectral blueing techniques.
    Upload a 3D SEG-Y file and adjust the parameters to optimize the bandwidth enhancement.
    """)
    
    # Initialize enhancer
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = SeismicBandwidthEnhancer()
        st.session_state.original_filename = None
        st.session_state.enhanced_file_path = None
    
    enhancer = st.session_state.enhancer
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D SEG-Y File", 
        type=['sgy', 'segy'],
        help="Upload your 3D seismic data in SEG-Y format"
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
    
    if enhancer.original_data is not None:
        n_inlines, n_xlines, n_samples = enhancer.original_data.shape
        
        inline_idx = st.sidebar.slider(
            "Inline Index",
            min_value=0,
            max_value=n_inlines-1,
            value=n_inlines//2,
            help="Inline index for section view"
        )
        
        xline_idx = st.sidebar.slider(
            "Crossline Index",
            min_value=0,
            max_value=n_xlines-1,
            value=n_xlines//2,
            help="Crossline index for section view"
        )
        
        time_slice_idx = st.sidebar.slider(
            "Time Slice Index",
            min_value=0,
            max_value=n_samples-1,
            value=n_samples//2,
            help="Time slice index for horizontal view"
        )
    else:
        inline_idx = 50
        xline_idx = 50
        time_slice_idx = 100
    
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
            st.session_state.original_filename = temp_filename
        
        try:
            # Process button
            if st.button("🚀 Process Seismic Data", type="primary"):
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
                        filter_order=filter_order
                    )
                
                st.success("✅ 3D Processing completed!")
                
                # Create download section
                st.sidebar.header("💾 Download Results")
                
                # Enhanced SEG-Y download
                output_filename = "enhanced_seismic.sgy"
                
                with st.sidebar:
                    if st.button("🛠️ Generate Enhanced SEG-Y File", type="secondary"):
                        with st.spinner("Creating enhanced SEG-Y file..."):
                            enhanced_file_path = enhancer.create_downloadable_segy(temp_filename, output_filename)
                            
                            if enhanced_file_path:
                                st.session_state.enhanced_file_path = enhanced_file_path
                                
                                # Read the file for download
                                with open(enhanced_file_path, "rb") as file:
                                    file_data = file.read()
                                
                                st.sidebar.download_button(
                                    label="📥 Download Enhanced SEG-Y",
                                    data=file_data,
                                    file_name=output_filename,
                                    mime="application/octet-stream",
                                    help="Download the enhanced seismic data in SEG-Y format"
                                )
                                st.sidebar.success("Enhanced SEG-Y file ready for download!")
                            else:
                                st.sidebar.error("Failed to create enhanced SEG-Y file for download")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📈 Trace Comparison", 
                    "🖼️ 2D Sections", 
                    "🌐 3D Volume",
                    "🕒 Time Slices",
                    "📊 Frequency Analysis",
                    "📋 Data Statistics"
                ])
                
                with tab1:
                    st.header("Trace Comparison")
                    # Plot spectral comparison
                    n_inlines, n_xlines, n_samples = enhancer.original_data.shape
                    safe_inline = min(inline_idx, n_inlines - 1)
                    safe_xline = min(xline_idx, n_xlines - 1)
                    
                    original_trace = enhancer.original_data[safe_inline, safe_xline, :]
                    enhanced_trace = enhanced_data[safe_inline, safe_xline, :]
                    
                    fig_comparison = enhancer.plot_spectral_comparison(
                        original_trace, enhanced_trace, 
                        trace_idx=f"Inline {safe_inline}, Xline {safe_xline}"
                    )
                    st.pyplot(fig_comparison)
                
                with tab2:
                    st.header("2D Section Views")
                    fig_section = enhancer.plot_seismic_section(inline_idx, xline_idx)
                    if fig_section:
                        st.pyplot(fig_section)
                
                with tab3:
                    st.header("3D Volume Visualization")
                    st.info("Interactive 3D volume visualization (may take a moment to load)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Volume")
                        fig_3d_orig = enhancer.plot_3d_volume('original')
                        if fig_3d_orig:
                            st.plotly_chart(fig_3d_orig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Enhanced Volume")
                        fig_3d_enh = enhancer.plot_3d_volume('enhanced')
                        if fig_3d_enh:
                            st.plotly_chart(fig_3d_enh, use_container_width=True)
                
                with tab4:
                    st.header("Time Slice Comparison")
                    fig_time_slice = enhancer.plot_time_slice_comparison(time_slice_idx)
                    if fig_time_slice:
                        st.plotly_chart(fig_time_slice, use_container_width=True)
                
                with tab5:
                    st.header("Frequency Analysis")
                    fig_freq = enhancer.plot_frequency_analysis(num_analysis_traces)
                    if fig_freq:
                        st.pyplot(fig_freq)
                
                with tab6:
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
                        st.write("**Data Type:**", "3D" if enhancer.original_data.shape[0] > 1 else "2D")
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
            # Clean up temporary files on app restart
            pass
    
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a 3D SEG-Y file using the sidebar to begin processing.")
        
        # Example of what the tool does
        st.header("About This 3D Tool")
        st.markdown("""
        This tool enhances 3D seismic data bandwidth using **spectral blueing** techniques:
        
        - **3D Data Support**: Full support for 3D seismic volumes with proper geometry
        - **Spectral Blueing**: Frequency-dependent enhancement that boosts mid-to-high frequencies
        - **Bandpass Filtering**: Removes very low and very high frequency noise
        - **Interactive 3D Visualization**: Explore your data in 3D using Plotly
        - **Download Enhanced Data**: Export your processed data as SEG-Y file
        
        ### New Features:
        - **Download Enhanced SEG-Y**: Get your processed data in standard SEG-Y format
        - **3D Volume Rendering**: Interactive 3D visualization of seismic volumes
        - **Multiple Section Views**: Inline, crossline, and time slice views
        - **Time Slice Comparison**: Interactive horizontal slice comparisons
        - **Proper Geometry Handling**: Reads and preserves 3D survey geometry
        
        ### Supported Data:
        - 3D seismic surveys with proper geometry
        - 2D seismic lines (will be treated as 3D with one inline)
        - Both pre-stack and post-stack data
        """)

    # Clean up temporary files when session ends
    if hasattr(st.session_state, 'original_filename') and st.session_state.original_filename:
        try:
            if os.path.exists(st.session_state.original_filename):
                os.unlink(st.session_state.original_filename)
        except:
            pass
    
    if hasattr(st.session_state, 'enhanced_file_path') and st.session_state.enhanced_file_path:
        try:
            if os.path.exists(st.session_state.enhanced_file_path):
                os.unlink(st.session_state.enhanced_file_path)
        except:
            pass

if __name__ == "__main__":
    main()
