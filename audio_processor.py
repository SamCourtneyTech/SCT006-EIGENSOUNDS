import numpy as np
import librosa
import soundfile as sf
import io
from scipy import signal
from scipy.fft import fft, ifft

class AudioProcessor:
    """Handles all audio processing operations for the Eigen-Sounds application."""
    
    def __init__(self):
        self.default_sr = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.max_duration = 30.0  # Maximum duration in seconds
        self.max_size_mb = 50     # Maximum file size in MB
        
    def compute_spectrogram(self, audio_data, sample_rate):
        """Compute the Short-Time Fourier Transform (STFT) spectrogram."""
        try:
            # Limit audio length for memory management
            max_samples = int(self.max_duration * sample_rate)
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # Use smaller FFT size for very long audio
            n_fft = self.n_fft
            if len(audio_data) > sample_rate * 10:  # If longer than 10 seconds
                n_fft = 1024  # Reduce FFT size
            
            # Compute STFT
            stft = librosa.stft(
                audio_data, 
                n_fft=n_fft, 
                hop_length=self.hop_length
            )
            
            # Convert to magnitude spectrogram
            magnitude_spec = np.abs(stft)
            
            # Convert to dB scale for better visualization
            db_spec = librosa.amplitude_to_db(magnitude_spec, ref=np.max)
            
            return db_spec
            
        except Exception as e:
            raise Exception(f"Error computing spectrogram: {str(e)}")
    
    def spectrogram_to_audio(self, spectrogram, sample_rate):
        """Reconstruct audio from spectrogram using Griffin-Lim algorithm."""
        try:
            # Convert from dB back to linear scale
            magnitude_spec = librosa.db_to_amplitude(spectrogram)
            
            # Use Griffin-Lim to reconstruct phase and convert to audio
            audio_reconstructed = librosa.griffinlim(
                magnitude_spec,
                n_iter=32,
                hop_length=self.hop_length
            )
            
            # Normalize audio
            audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed))
            
            return audio_reconstructed
            
        except Exception as e:
            raise Exception(f"Error reconstructing audio: {str(e)}")
    
    def audio_to_bytes(self, audio_data, sample_rate):
        """Convert audio array to bytes for playback in Streamlit."""
        try:
            # Ensure audio is in valid range
            audio_normalized = np.clip(audio_data, -1.0, 1.0)
            
            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_normalized, sample_rate, format='WAV')
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            raise Exception(f"Error converting audio to bytes: {str(e)}")
    
    def compute_audio_stats(self, audio_data):
        """Compute basic statistics for audio signal."""
        try:
            duration = len(audio_data) / self.default_sr
            rms_energy = np.sqrt(np.mean(audio_data**2))
            peak_amplitude = np.max(np.abs(audio_data))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            return {
                'duration': duration,
                'sample_rate': self.default_sr,
                'rms_energy': rms_energy,
                'peak_amplitude': peak_amplitude,
                'zcr': zcr
            }
            
        except Exception as e:
            raise Exception(f"Error computing audio stats: {str(e)}")
    
    def segment_audio(self, audio_data, sample_rate, segment_length_sec):
        """Split audio into equal-length segments."""
        try:
            segment_length_samples = int(segment_length_sec * sample_rate)
            
            # Calculate number of complete segments
            num_segments = len(audio_data) // segment_length_samples
            
            segments = []
            for i in range(num_segments):
                start_idx = i * segment_length_samples
                end_idx = start_idx + segment_length_samples
                segment = audio_data[start_idx:end_idx]
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            raise Exception(f"Error segmenting audio: {str(e)}")
    
    def apply_window(self, audio_data, window_type='hann'):
        """Apply windowing function to audio segment."""
        try:
            if window_type == 'hann':
                window = np.hanning(len(audio_data))
            elif window_type == 'hamming':
                window = np.hamming(len(audio_data))
            elif window_type == 'blackman':
                window = np.blackman(len(audio_data))
            else:
                window = np.ones(len(audio_data))  # Rectangular window
            
            return audio_data * window
            
        except Exception as e:
            raise Exception(f"Error applying window: {str(e)}")
    
    def compute_mfcc(self, audio_data, sample_rate, n_mfcc=13):
        """Compute Mel-Frequency Cepstral Coefficients."""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                hop_length=self.hop_length
            )
            return mfcc
            
        except Exception as e:
            raise Exception(f"Error computing MFCC: {str(e)}")
    
    def compute_spectral_features(self, audio_data, sample_rate):
        """Compute various spectral features."""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            
            return {
                'centroid': spectral_centroids,
                'bandwidth': spectral_bandwidth,
                'rolloff': spectral_rolloff
            }
            
        except Exception as e:
            raise Exception(f"Error computing spectral features: {str(e)}")
    
    def normalize_audio(self, audio_data, method='peak'):
        """Normalize audio using different methods."""
        try:
            if method == 'peak':
                # Peak normalization
                peak = np.max(np.abs(audio_data))
                if peak > 0:
                    return audio_data / peak
                else:
                    return audio_data
                    
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > 0:
                    target_rms = 0.1  # Target RMS level
                    return audio_data * (target_rms / rms)
                else:
                    return audio_data
                    
            else:
                return audio_data
                
        except Exception as e:
            raise Exception(f"Error normalizing audio: {str(e)}")
    
    def apply_fade(self, audio_data, fade_length_sec=0.01, sample_rate=22050):
        """Apply fade in/out to prevent clicks."""
        try:
            fade_length_samples = int(fade_length_sec * sample_rate)
            fade_length_samples = min(fade_length_samples, len(audio_data) // 2)
            
            if fade_length_samples > 0:
                # Create fade curves
                fade_in = np.linspace(0, 1, fade_length_samples)
                fade_out = np.linspace(1, 0, fade_length_samples)
                
                # Apply fades
                audio_faded = audio_data.copy()
                audio_faded[:fade_length_samples] *= fade_in
                audio_faded[-fade_length_samples:] *= fade_out
                
                return audio_faded
            else:
                return audio_data
                
        except Exception as e:
            raise Exception(f"Error applying fade: {str(e)}")
