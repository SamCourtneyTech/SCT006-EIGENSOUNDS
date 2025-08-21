import numpy as np
from scipy.linalg import svd, qr, cholesky, lu
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import scipy.signal

class LinearAlgebraDemo:
    """Demonstrates linear algebra concepts through audio processing."""
    
    def __init__(self):
        self.eps = 1e-12  # Small value to prevent numerical issues
    
    def analyze_matrix_properties(self, matrix):
        """Analyze basic properties of a matrix."""
        try:
            # Basic properties
            shape = matrix.shape
            rank = np.linalg.matrix_rank(matrix)
            
            # Norms
            frobenius_norm = np.linalg.norm(matrix, 'fro')
            spectral_norm = np.linalg.norm(matrix, 2)
            
            # Condition number (for square matrices or using SVD)
            if shape[0] == shape[1]:
                try:
                    condition_number = np.linalg.cond(matrix)
                except:
                    condition_number = np.inf
            else:
                # Use SVD for non-square matrices
                U, s, Vt = svd(matrix, full_matrices=False)
                condition_number = s[0] / s[-1] if s[-1] > self.eps else np.inf
            
            return {
                'shape': shape,
                'rank': rank,
                'frobenius_norm': frobenius_norm,
                'spectral_norm': spectral_norm,
                'condition_number': condition_number
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing matrix properties: {str(e)}")
    
    def svd_compress_spectrogram(self, spectrogram, k_components):
        """Compress spectrogram using SVD with k components."""
        try:
            # Check matrix size to prevent memory issues
            matrix_size_mb = spectrogram.nbytes / (1024 * 1024)
            if matrix_size_mb > 100:  # If matrix is larger than 100MB
                # Downsample the spectrogram
                factor = int(np.sqrt(matrix_size_mb / 50))  # Reduce to ~50MB
                spectrogram = spectrogram[::factor, ::factor]
            
            # Perform SVD
            U, s, Vt = svd(spectrogram, full_matrices=False)
            
            # Keep only k components
            k = min(k_components, len(s))
            U_k = U[:, :k]
            s_k = s[:k]
            Vt_k = Vt[:k, :]
            
            # Reconstruct
            compressed_spectrogram = U_k @ np.diag(s_k) @ Vt_k
            
            return compressed_spectrogram, U, s, Vt
            
        except Exception as e:
            raise Exception(f"Error in SVD compression: {str(e)}")
    
    def compute_audio_similarity_matrix(self, audio_segments):
        """Compute similarity matrix between audio segments using dot products."""
        try:
            n_segments = len(audio_segments)
            similarity_matrix = np.zeros((n_segments, n_segments))
            
            # Normalize segments for cosine similarity
            normalized_segments = []
            for segment in audio_segments:
                norm = np.linalg.norm(segment)
                if norm > self.eps:
                    normalized_segments.append(segment / norm)
                else:
                    normalized_segments.append(segment)
            
            # Compute cosine similarity (normalized dot product)
            for i in range(n_segments):
                for j in range(n_segments):
                    similarity_matrix[i, j] = np.dot(
                        normalized_segments[i], 
                        normalized_segments[j]
                    )
            
            return similarity_matrix
            
        except Exception as e:
            raise Exception(f"Error computing similarity matrix: {str(e)}")
    
    def compute_frequency_covariance(self, spectrogram):
        """Compute covariance matrix of frequency bins."""
        try:
            # Transpose so that each row is a frequency bin across time
            freq_data = spectrogram
            
            # Center the data (subtract mean)
            freq_centered = freq_data - np.mean(freq_data, axis=1, keepdims=True)
            
            # Compute covariance matrix
            covariance = np.cov(freq_centered)
            
            return covariance
            
        except Exception as e:
            raise Exception(f"Error computing frequency covariance: {str(e)}")
    
    def compute_temporal_covariance(self, spectrogram):
        """Compute covariance matrix of time frames."""
        try:
            # Each column is a time frame
            time_data = spectrogram.T  # Transpose to get time frames as rows
            
            # Center the data
            time_centered = time_data - np.mean(time_data, axis=1, keepdims=True)
            
            # Compute covariance matrix
            covariance = np.cov(time_centered)
            
            return covariance
            
        except Exception as e:
            raise Exception(f"Error computing temporal covariance: {str(e)}")
    
    def compute_eigendecomposition(self, matrix):
        """Compute eigenvalues and eigenvectors of a matrix."""
        try:
            # Ensure matrix is symmetric for stability
            if not np.allclose(matrix, matrix.T, atol=1e-8):
                # Force symmetry
                matrix = (matrix + matrix.T) / 2
            
            # Compute eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            raise Exception(f"Error in eigendecomposition: {str(e)}")
    
    def project_onto_principal_components(self, spectrogram, eigenvecs, num_components, analysis_type):
        """Project spectrogram onto principal components."""
        try:
            if analysis_type == "Frequency Covariance":
                # Project frequency patterns
                data = spectrogram
                components = eigenvecs[:, :num_components]
                projected = components.T @ data
            else:
                # Project temporal patterns
                data = spectrogram.T
                components = eigenvecs[:, :num_components]
                projected = data @ components
            
            return projected
            
        except Exception as e:
            raise Exception(f"Error projecting onto principal components: {str(e)}")
    
    def reconstruct_from_principal_components(self, projected_data, eigenvecs, num_components, analysis_type, original_shape):
        """Reconstruct data from principal component projection."""
        try:
            components = eigenvecs[:, :num_components]
            
            if analysis_type == "Frequency Covariance":
                # Reconstruct frequency patterns
                reconstructed = components @ projected_data
            else:
                # Reconstruct temporal patterns
                reconstructed = (projected_data @ components.T).T
            
            return reconstructed
            
        except Exception as e:
            raise Exception(f"Error reconstructing from principal components: {str(e)}")
    
    def create_rotation_matrix(self, angle_degrees):
        """Create 2D rotation matrix."""
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        return np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
    
    def create_scaling_matrix(self, scale_x, scale_y):
        """Create 2D scaling matrix."""
        return np.array([
            [scale_x, 0],
            [0, scale_y]
        ])
    
    def create_shearing_matrix(self, shear_x, shear_y):
        """Create 2D shearing matrix."""
        return np.array([
            [1, shear_x],
            [shear_y, 1]
        ])
    
    def create_random_orthogonal_matrix(self, size):
        """Create random orthogonal matrix using QR decomposition."""
        # Generate random matrix
        random_matrix = np.random.randn(size, size)
        
        # QR decomposition to get orthogonal matrix
        Q, _ = qr(random_matrix)
        
        return Q
    
    def create_lowpass_filter_matrix(self, size, cutoff_freq):
        """Create low-pass filter matrix."""
        # Create frequency response
        freqs = np.linspace(0, 1, size)
        response = np.where(freqs <= cutoff_freq, 1.0, 0.0)
        
        # Convert to time domain (filter kernel)
        kernel = np.fft.irfft(response, n=size)
        
        # Create Toeplitz matrix
        filter_matrix = scipy.linalg.toeplitz(kernel, kernel)
        
        return filter_matrix
    
    def create_mixing_matrix(self, num_channels):
        """Create random mixing matrix for audio channels."""
        # Create random mixing matrix
        mixing_matrix = np.random.randn(num_channels, num_channels)
        
        # Normalize rows to prevent amplification
        row_norms = np.linalg.norm(mixing_matrix, axis=1, keepdims=True)
        mixing_matrix = mixing_matrix / (row_norms + self.eps)
        
        return mixing_matrix
    
    def apply_matrix_to_spectrogram_patches(self, spectrogram, transform_matrix):
        """Apply transformation matrix to spectrogram patches."""
        try:
            # Get patch size from matrix
            patch_size = min(transform_matrix.shape[0], spectrogram.shape[0])
            
            # Apply transformation to overlapping patches
            transformed_spec = spectrogram.copy()
            
            for i in range(0, spectrogram.shape[0] - patch_size + 1, patch_size // 2):
                for j in range(0, spectrogram.shape[1] - patch_size + 1, patch_size // 2):
                    # Extract patch
                    patch = spectrogram[i:i+patch_size, j:j+patch_size]
                    
                    # Apply transformation
                    if patch.shape[0] == transform_matrix.shape[1]:
                        transformed_patch = transform_matrix @ patch
                        transformed_spec[i:i+patch_size, j:j+patch_size] = transformed_patch
            
            return transformed_spec
            
        except Exception as e:
            raise Exception(f"Error applying matrix to spectrogram: {str(e)}")
    
    def apply_mixing_matrix(self, audio_segments, mixing_matrix):
        """Apply mixing matrix to create mixed audio channels."""
        try:
            # Stack segments as matrix (each row is a segment)
            audio_matrix = np.array(audio_segments)
            
            # Apply mixing
            mixed_audio = mixing_matrix @ audio_matrix
            
            return mixed_audio
            
        except Exception as e:
            raise Exception(f"Error applying mixing matrix: {str(e)}")
    
    def apply_convolution_filter(self, audio_data, filter_kernel):
        """Apply convolution filter to audio."""
        try:
            # Normalize kernel
            kernel = filter_kernel / (np.sum(np.abs(filter_kernel)) + self.eps)
            
            # Apply convolution
            filtered_audio = scipy.signal.convolve(audio_data, kernel, mode='same')
            
            return filtered_audio
            
        except Exception as e:
            raise Exception(f"Error applying convolution filter: {str(e)}")
    
    def create_rank_deficient_matrix(self, matrix, rank_reduction_factor):
        """Create rank-deficient version of matrix."""
        try:
            # Perform SVD
            U, s, Vt = svd(matrix, full_matrices=False)
            
            # Reduce rank
            target_rank = max(1, len(s) // rank_reduction_factor)
            s_reduced = s.copy()
            s_reduced[target_rank:] = 0
            
            # Reconstruct
            reduced_matrix = U @ np.diag(s_reduced) @ Vt
            
            return reduced_matrix
            
        except Exception as e:
            raise Exception(f"Error creating rank-deficient matrix: {str(e)}")
    
    def compute_null_and_range_space(self, matrix):
        """Compute null space and range space of matrix."""
        try:
            # SVD for numerical stability
            U, s, Vt = svd(matrix, full_matrices=True)
            
            # Determine rank
            tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * s[0]
            rank = np.sum(s > tolerance)
            
            # Range space (column space)
            range_space = U[:, :rank]
            
            # Null space
            if rank < Vt.shape[0]:
                null_space = Vt[rank:, :].T
            else:
                null_space = None
            
            return null_space, range_space, rank
            
        except Exception as e:
            raise Exception(f"Error computing null and range space: {str(e)}")
    
    def project_onto_subspace(self, data, subspace_basis):
        """Project data onto subspace defined by basis vectors."""
        try:
            if subspace_basis is None:
                return np.zeros_like(data)
            
            # Ensure basis is orthonormal
            Q, _ = qr(subspace_basis, mode='economic')
            
            # Project
            projected = Q @ (Q.T @ data)
            
            return projected
            
        except Exception as e:
            raise Exception(f"Error projecting onto subspace: {str(e)}")
    
    def add_noise_to_spectrogram(self, spectrogram, noise_level):
        """Add noise to spectrogram for optimization demonstrations."""
        noise = np.random.normal(0, noise_level, spectrogram.shape)
        return spectrogram + noise
    
    def gradient_descent_denoising(self, noisy_spec, target_spec, learning_rate, num_iterations):
        """Perform gradient descent for spectrogram denoising."""
        try:
            # Initialize
            current_spec = noisy_spec.copy()
            cost_history = []
            gradient_norms = []
            
            for iteration in range(num_iterations):
                # Compute cost (mean squared error)
                cost = 0.5 * np.sum((current_spec - target_spec) ** 2)
                cost_history.append(cost)
                
                # Compute gradient
                gradient = current_spec - target_spec
                gradient_norm = np.linalg.norm(gradient, 'fro')
                gradient_norms.append(gradient_norm)
                
                # Update
                current_spec = current_spec - learning_rate * gradient
            
            return current_spec, cost_history, gradient_norms
            
        except Exception as e:
            raise Exception(f"Error in gradient descent: {str(e)}")
    
    def subsample_spectrogram(self, spectrogram, factor):
        """Subsample spectrogram for overdetermined system demonstration."""
        indices = np.arange(0, spectrogram.shape[1], factor)
        subsampled = spectrogram[:, indices]
        return subsampled, indices
    
    def solve_with_pseudoinverse(self, subsampled_spec, target_shape, indices):
        """Solve interpolation problem using pseudoinverse."""
        try:
            # Create measurement matrix
            A = np.zeros((len(indices), target_shape[1]))
            for i, idx in enumerate(indices):
                if idx < target_shape[1]:
                    A[i, idx] = 1
            
            # Solve for each frequency bin
            reconstructed = np.zeros(target_shape)
            
            for freq_bin in range(subsampled_spec.shape[0]):
                b = subsampled_spec[freq_bin, :]
                # Solve Ax = b using pseudoinverse
                x = np.linalg.pinv(A) @ b
                reconstructed[freq_bin, :] = x
            
            return reconstructed
            
        except Exception as e:
            raise Exception(f"Error solving with pseudoinverse: {str(e)}")
    
    def qr_decomposition(self, matrix):
        """Perform QR decomposition."""
        return qr(matrix, mode='economic')
    
    def compute_spectral_properties(self, matrix):
        """Compute various spectral properties of matrix."""
        try:
            # SVD for spectral analysis
            U, s, Vt = svd(matrix, full_matrices=False)
            
            # Spectral norm (largest singular value)
            spectral_norm = s[0]
            
            # Nuclear norm (sum of singular values)
            nuclear_norm = np.sum(s)
            
            # Effective rank (number of significant singular values)
            total_variance = np.sum(s)
            cumulative_variance = np.cumsum(s)
            effective_rank = np.argmax(cumulative_variance >= 0.95 * total_variance) + 1
            
            # Condition number
            condition_number = s[0] / s[-1] if s[-1] > self.eps else np.inf
            
            # Stable rank
            stable_rank = nuclear_norm / spectral_norm
            
            return {
                'spectral_norm': spectral_norm,
                'nuclear_norm': nuclear_norm,
                'effective_rank': effective_rank,
                'condition_number': condition_number,
                'stable_rank': stable_rank
            }
            
        except Exception as e:
            raise Exception(f"Error computing spectral properties: {str(e)}")
