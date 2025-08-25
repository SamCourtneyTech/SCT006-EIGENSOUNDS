import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import io
import tempfile
import os
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

from audio_processor import AudioProcessor
from linear_algebra_demos import LinearAlgebraDemo
from visualization import AudioVisualizer

# Configure page
st.set_page_config(
    page_title="Eigen-Sounds: Linear Algebra Through Audio",
    layout="wide"
)

# Custom CSS with regular spinning wheel
st.markdown("""
<style>
/* Hide all default Streamlit spinners and loading indicators */
.stSpinner, .stSpinner *, .stSpinner::before, .stSpinner::after {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Hide the running indicator in top right */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Custom spinning wheel loading animation */
.stApp.stAppRunning::before {
    content: "";
    position: fixed;
    top: 15px;
    right: 15px;
    width: 20px;
    height: 20px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #FF6B6B;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    z-index: 9999;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Alternative wave bars animation */
.custom-loading {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 4px;
}

.wave-bar {
    width: 4px;
    height: 20px;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    animation: wave 1.2s ease-in-out infinite;
    border-radius: 2px;
}

.wave-bar:nth-child(2) { animation-delay: 0.1s; }
.wave-bar:nth-child(3) { animation-delay: 0.2s; }
.wave-bar:nth-child(4) { animation-delay: 0.3s; }
.wave-bar:nth-child(5) { animation-delay: 0.4s; }

@keyframes wave {
    0%, 100% { transform: scaleY(0.5); }
    50% { transform: scaleY(1.5); }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'spectrogram' not in st.session_state:
    st.session_state.spectrogram = None

def main():
    st.title("EigenSounds")
    st.subheader("Linear Algebra Through Audio Compression")
    st.markdown("""
    This is my interactive exploration of linear algebra concepts through audio processing!
    This application demonstrates how mathematical concepts like SVD, eigenvalues, and matrix operations
    can be applied to audio compression and analysis. Load times can take a bit. Be sure to check out my additional notes and explanations below each section.
    """)

    # Initialize processors
    audio_processor = AudioProcessor()
    la_demo = LinearAlgebraDemo()
    visualizer = AudioVisualizer()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a demonstration:",
        [
            "Audio Upload & Basic Analysis",
            "Spectrogram as Matrix",
            "SVD Compression",
            "Audio Similarity (Dot Products)",
            "Eigenvalue Analysis",
            "Linear Transformations",
            "Advanced Concepts"
        ]
    )

    # Audio Upload Section (always available)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Audio Upload")
    st.sidebar.info("File size limit: 50MB\nMax duration: 30 seconds")
    
    # Test audio button
    if st.sidebar.button("Load Test Audio (EDM Track)", help="Load a sample audio file to try the features"):
        try:
            # Load the test audio file preserving original quality
            audio_data, sample_rate = librosa.load("edm_track.mp3", sr=None, duration=30.0)
            
            # Only downsample if sample rate is very high (>48kHz) to preserve quality
            if sample_rate > 48000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
                sample_rate = 44100
            
            # Store in session state
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            # Compute spectrogram
            st.session_state.spectrogram = audio_processor.compute_spectrogram(audio_data, sample_rate)
            
            duration = len(audio_data) / sample_rate
            st.sidebar.success(f"Test audio loaded! Duration: {duration:.2f} seconds")
            
        except Exception as e:
            st.sidebar.error(f"Error loading test audio: {str(e)}")
    
    st.sidebar.markdown("**Or upload your own:**")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an audio file",
        type=['wav', 'mp3', 'flac', 'ogg'],
        help="Supported formats: WAV, MP3, FLAC, OGG. Keep files under 50MB for best performance."
    )

    if uploaded_file is not None:
        try:
            # Check file size (50MB limit)
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 50:
                st.sidebar.error(f"File too large: {file_size_mb:.1f}MB. Please use files under 50MB.")
                return
            
            # Load audio with duration limit
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load only first 30 seconds but preserve original sample rate when possible
            audio_data, sample_rate = librosa.load(tmp_path, sr=None, duration=30.0)
            
            # Only downsample if sample rate is very high (>48kHz) to preserve quality
            if sample_rate > 48000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
                sample_rate = 44100
            os.unlink(tmp_path)
            
            # Check if audio is too short
            if len(audio_data) < 1024:
                st.sidebar.error("Audio file is too short. Please use files with at least 1 second of audio.")
                return

            # Store in session state
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            # Compute spectrogram
            st.session_state.spectrogram = audio_processor.compute_spectrogram(audio_data, sample_rate)
            
            duration = len(audio_data) / sample_rate
            st.sidebar.success(f"Audio loaded! Duration: {duration:.2f} seconds, Sample rate: {sample_rate}Hz")
            if file_size_mb > 10:
                st.sidebar.info("Large file detected - using first 30 seconds only for performance.")
            
        except Exception as e:
            st.sidebar.error(f"Error loading audio: {str(e)}")

    # Main content based on selected page
    if page == "Audio Upload & Basic Analysis":
        show_basic_analysis(audio_processor, visualizer)
    
    elif page == "Spectrogram as Matrix":
        show_spectrogram_matrix(audio_processor, la_demo, visualizer)
    
    elif page == "SVD Compression":
        show_svd_compression(audio_processor, la_demo, visualizer)
    
    elif page == "Audio Similarity (Dot Products)":
        show_audio_similarity(audio_processor, la_demo, visualizer)
    
    elif page == "Eigenvalue Analysis":
        show_eigenvalue_analysis(audio_processor, la_demo, visualizer)
    
    elif page == "Linear Transformations":
        show_linear_transformations(audio_processor, la_demo, visualizer)
    
    elif page == "Advanced Concepts":
        show_advanced_concepts(audio_processor, la_demo, visualizer)

def show_basic_analysis(audio_processor, visualizer):
    st.header("Basic Audio Analysis")
    
    if st.session_state.audio_data is None:
        st.warning("Please upload an audio file using the sidebar.")
        return

    audio_data = st.session_state.audio_data
    sample_rate = st.session_state.sample_rate

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Waveform")
        fig_wave = visualizer.plot_waveform(audio_data, sample_rate)
        st.plotly_chart(fig_wave, use_container_width=True)
        
        # Play original audio
        st.audio(audio_processor.audio_to_bytes(audio_data, sample_rate), format='audio/wav')

    with col2:
        st.subheader("Audio Statistics")
        stats = audio_processor.compute_audio_stats(audio_data)
        
        st.metric("Duration (seconds)", f"{stats['duration']:.2f}")
        st.metric("Sample Rate (Hz)", stats['sample_rate'])
        st.metric("RMS Energy", f"{stats['rms_energy']:.4f}")
        st.metric("Peak Amplitude", f"{stats['peak_amplitude']:.4f}")
        st.metric("Zero Crossing Rate", f"{stats['zcr']:.4f}")

    st.markdown("""
    ### Linear Algebra Connection: Vectors and Norms
    
    - **Audio as Vector**: The audio signal is represented as a vector in ℝⁿ where n is the number of samples
    - **L2 Norm (Energy)**: The RMS energy is related to the L2 norm: ||x||₂ = √(Σxᵢ²)
    - **L∞ Norm (Peak)**: The peak amplitude represents the L∞ norm: ||x||∞ = max|xᵢ|
    """)
    
    st.markdown("""
    ### Sam's Note:
    
    We're using mono audio to show linear algebra operations with 2D matrices. Stereo audio would create 3D data structures that make the math demonstrations a little more complicated. By using mono, we can focus on understanding the core concepts like SVD and eigenvalues without getting distracted by left-right channel differences.
    """)

def show_spectrogram_matrix(audio_processor, la_demo, visualizer):
    st.header("Spectrogram as Matrix Representation")
    
    if st.session_state.spectrogram is None:
        st.warning("Please upload an audio file first.")
        return

    spectrogram = st.session_state.spectrogram
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spectrogram Visualization")
        fig_spec = visualizer.plot_spectrogram(spectrogram, st.session_state.sample_rate)
        st.plotly_chart(fig_spec, use_container_width=True)

    with col2:
        st.subheader("Matrix Properties")
        matrix_props = la_demo.analyze_matrix_properties(spectrogram)
        
        st.metric("Matrix Shape", f"{matrix_props['shape'][0]} × {matrix_props['shape'][1]}")
        st.metric("Rank", matrix_props['rank'])
        st.metric("Condition Number", f"{matrix_props['condition_number']:.2e}")
        st.metric("Frobenius Norm", f"{matrix_props['frobenius_norm']:.2e}")

    # Interactive matrix exploration
    st.subheader("Interactive Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        freq_range = st.slider(
            "Frequency Range to Display",
            0, spectrogram.shape[0]-1,
            (0, min(100, spectrogram.shape[0]-1)),
            help="Select frequency bins to visualize"
        )
        
    with col2:
        time_range = st.slider(
            "Time Range to Display",
            0, spectrogram.shape[1]-1,
            (0, min(200, spectrogram.shape[1]-1)),
            help="Select time frames to visualize"
        )

    # Show matrix subset
    matrix_subset = spectrogram[freq_range[0]:freq_range[1]+1, time_range[0]:time_range[1]+1]
    
    if matrix_subset.size > 0:
        fig_matrix = visualizer.plot_matrix_heatmap(matrix_subset, "Spectrogram Matrix Subset")
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("""
    ### Linear Algebra Concepts Demonstrated:
    
    - **Matrix Representation**: Each column represents a time frame, each row a frequency bin
    - **Rank**: Indicates the dimensionality of the frequency content
    - **Column Space**: Represents all possible linear combinations of frequency patterns
    - **Row Space**: Represents all possible linear combinations of temporal patterns
    - **Condition Number**: Measures how well-conditioned the matrix is for numerical operations
    """)
    


def show_svd_compression(audio_processor, la_demo, visualizer):
    st.header("🗜️ SVD-Based Audio Compression")
    
    if st.session_state.spectrogram is None:
        st.warning("Please upload an audio file first.")
        return

    spectrogram = st.session_state.spectrogram
    
    # SVD compression controls
    st.subheader("Compression Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_components = min(spectrogram.shape)
        k_components = st.slider(
            "Number of SVD Components (k)",
            1, max_components,
            max_components // 4,
            help=f"Original has {max_components} components"
        )
    
    with col2:
        compression_ratio = (1 - k_components / max_components) * 100
        st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
    
    with col3:
        # Calculate storage reduction
        original_size = np.prod(spectrogram.shape)
        compressed_size = k_components * (spectrogram.shape[0] + spectrogram.shape[1] + 1)
        storage_reduction = (1 - compressed_size / original_size) * 100
        st.metric("Storage Reduction", f"{storage_reduction:.1f}%")

    # Check spectrogram size and warn user
    spec_size_mb = spectrogram.nbytes / (1024 * 1024)
    if spec_size_mb > 50:
        st.warning(f"Large spectrogram detected ({spec_size_mb:.1f}MB). Processing may be slow and use automatic optimization.")
    
    # Perform SVD compression
    try:
        # Custom loading animation for SVD
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class="custom-loading">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        <p style="text-align: center; margin-top: 10px;">🎵 Computing SVD compression...</p>
        """, unsafe_allow_html=True)
        
        compressed_spec, U, s, Vt = la_demo.svd_compress_spectrogram(spectrogram, k_components)
        
        # Clear loading animation
        loading_placeholder.empty()
        
        # Reconstruct audio
        compressed_audio = audio_processor.spectrogram_to_audio(
            compressed_spec, st.session_state.sample_rate
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Spectrogram")
            fig_orig = visualizer.plot_spectrogram(spectrogram, st.session_state.sample_rate)
            st.plotly_chart(fig_orig, use_container_width=True)
            
            st.subheader("Original Audio (Unprocessed)")
            # Use the original audio data directly - no reconstruction
            st.audio(audio_processor.audio_to_bytes(
                st.session_state.audio_data, st.session_state.sample_rate
            ), format='audio/wav')

        with col2:
            st.subheader("Compressed Spectrogram")
            fig_comp = visualizer.plot_spectrogram(compressed_spec, st.session_state.sample_rate)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.subheader("Compressed Audio")
            st.audio(audio_processor.audio_to_bytes(
                compressed_audio, st.session_state.sample_rate
            ), format='audio/wav')

        # Singular value analysis
        st.subheader("Singular Value Analysis")
        fig_sv = visualizer.plot_singular_values(s, k_components)
        st.plotly_chart(fig_sv, use_container_width=True)

        # Error analysis
        reconstruction_error = np.linalg.norm(spectrogram - compressed_spec, 'fro')
        st.metric("Reconstruction Error (Frobenius Norm)", f"{reconstruction_error:.2e}")

    except Exception as e:
        st.error(f"Error in SVD compression: {str(e)}")

    st.markdown("""
    ### SVD Compression Theory:
    
    - **Singular Value Decomposition**: A = UΣVᵀ where U, V are orthogonal and Σ is diagonal
    - **Low-rank Approximation**: Keep only the k largest singular values
    - **Optimal Approximation**: SVD gives the best rank-k approximation in terms of Frobenius norm
    - **Energy Compaction**: Most signal energy is captured by the largest singular values
    """)
    
    st.markdown("""
    ### Sam's Note: How SVD Compression Works
    
    SVD compression works by turning audio into a matrix, applying Singular Value Decomposition to break it into important components, keeping only the top singular values that capture most of the signal's information, and then reconstructing the audio from those. This reduces storage while keeping the sound close to the original. Singular values can be thought of as values that hold information about the sound, with larger ones carrying more important information and smaller ones carrying less.
    """)

def show_audio_similarity(audio_processor, la_demo, visualizer):
    st.header("🔍 Audio Similarity Using Dot Products")
    
    if st.session_state.audio_data is None:
        st.warning("Please upload an audio file first.")
        return

    audio_data = st.session_state.audio_data
    sample_rate = st.session_state.sample_rate

    st.subheader("Audio Segmentation and Comparison")
    
    # Segment the audio
    segment_length = st.slider(
        "Segment Length (seconds)",
        0.1, 2.0, 0.5,
        step=0.1,
        help="Length of audio segments to compare"
    )
    
    segments = audio_processor.segment_audio(audio_data, sample_rate, segment_length)
    
    if len(segments) < 2:
        st.warning("Audio is too short for segmentation. Try a shorter segment length.")
        return

    st.write(f"Created {len(segments)} segments of {segment_length} seconds each")

    # Compute similarity matrix
    similarity_matrix = la_demo.compute_audio_similarity_matrix(segments)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Similarity Matrix")
        fig_sim = visualizer.plot_similarity_matrix(similarity_matrix)
        st.plotly_chart(fig_sim, use_container_width=True)

    with col2:
        st.subheader("Segment Analysis")
        
        # Select segments to compare
        seg1_idx = st.selectbox("Select first segment", range(len(segments)), 0)
        seg2_idx = st.selectbox("Select second segment", range(len(segments)), min(1, len(segments)-1))
        
        if seg1_idx != seg2_idx:
            similarity = similarity_matrix[seg1_idx, seg2_idx]
            st.metric("Cosine Similarity", f"{similarity:.4f}")
            
            # Vector analysis
            vec1 = segments[seg1_idx]
            vec2 = segments[seg2_idx]
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            st.metric("Dot Product", f"{dot_product:.2e}")
            st.metric("Norm of Segment 1", f"{norm1:.4f}")
            st.metric("Norm of Segment 2", f"{norm2:.4f}")

    # Play selected segments
    st.subheader("Listen to Selected Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Segment {seg1_idx + 1}")
        st.audio(audio_processor.audio_to_bytes(
            segments[seg1_idx], sample_rate
        ), format='audio/wav')

    with col2:
        st.write(f"Segment {seg2_idx + 1}")
        st.audio(audio_processor.audio_to_bytes(
            segments[seg2_idx], sample_rate
        ), format='audio/wav')

    # Most similar and dissimilar pairs
    st.subheader("Similarity Analysis")
    
    # Find most similar and dissimilar pairs
    similarity_pairs = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            similarity_pairs.append((i, j, similarity_matrix[i, j]))
    
    similarity_pairs.sort(key=lambda x: x[2])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Most Similar Segments:**")
        for i, j, sim in similarity_pairs[-3:]:
            st.write(f"Segments {i+1} & {j+1}: {sim:.4f}")

    with col2:
        st.write("**Most Dissimilar Segments:**")
        for i, j, sim in similarity_pairs[:3]:
            st.write(f"Segments {i+1} & {j+1}: {sim:.4f}")

    st.markdown("""
    ### Dot Products and Similarity:
    
    - **Dot Product**: a·b = Σaᵢbᵢ measures both magnitude and alignment
    - **Cosine Similarity**: cos(θ) = (a·b)/(||a||·||b||) measures only alignment
    - **Correlation**: Similar to cosine similarity but with mean-centered vectors
    - **Applications**: Audio matching, pattern recognition, content-based retrieval
    """)

def show_eigenvalue_analysis(audio_processor, la_demo, visualizer):
    st.header("🔍 Eigenvalue Analysis of Audio Patterns")
    
    if st.session_state.spectrogram is None:
        st.warning("Please upload an audio file first.")
        return

    spectrogram = st.session_state.spectrogram
    
    st.subheader("Covariance Matrix Analysis")
    
    # Choose analysis type
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Frequency Covariance", "Temporal Covariance"],
        help="Frequency: patterns across frequency bins, Temporal: patterns across time"
    )
    
    if analysis_type == "Frequency Covariance":
        # Analyze frequency patterns (covariance of frequency bins)
        covariance_matrix = la_demo.compute_frequency_covariance(spectrogram)
        eigenvals, eigenvecs = la_demo.compute_eigendecomposition(covariance_matrix)
        
        st.write("Analyzing patterns across frequency bins...")
        
    else:
        # Analyze temporal patterns (covariance of time frames)
        covariance_matrix = la_demo.compute_temporal_covariance(spectrogram)
        eigenvals, eigenvecs = la_demo.compute_eigendecomposition(covariance_matrix)
        
        st.write("Analyzing patterns across time frames...")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Eigenvalue Spectrum")
        fig_eigen = visualizer.plot_eigenvalue_spectrum(eigenvals)
        st.plotly_chart(fig_eigen, use_container_width=True)
        
        # Eigenvalue statistics
        total_variance = np.sum(eigenvals)
        explained_variance = np.cumsum(eigenvals) / total_variance * 100
        
        num_components = st.slider(
            "Number of Principal Components",
            1, min(20, len(eigenvals)),
            min(5, len(eigenvals)),
            help="Number of top eigenvalues to analyze"
        )
        
        st.metric("Variance Explained by Top Components", f"{explained_variance[num_components-1]:.1f}%")

    with col2:
        st.subheader("Principal Components")
        
        # Select which eigenvector to visualize
        component_idx = st.selectbox(
            "Select Component to Visualize",
            range(min(10, len(eigenvecs))),
            help="Choose which principal component to display"
        )
        
        if component_idx < len(eigenvecs):
            selected_eigenvec = eigenvecs[:, component_idx]
            
            fig_eigenvec = visualizer.plot_eigenvector(
                selected_eigenvec, 
                f"Principal Component {component_idx + 1}",
                analysis_type
            )
            st.plotly_chart(fig_eigenvec, use_container_width=True)
            
            st.metric(f"Eigenvalue {component_idx + 1}", f"{eigenvals[component_idx]:.2e}")

    # Dimensionality reduction visualization
    st.subheader("Dimensionality Reduction")
    
    # Project onto principal components
    projected_data = la_demo.project_onto_principal_components(
        spectrogram, eigenvecs, num_components, analysis_type
    )
    
    # Reconstruct from reduced dimensions
    reconstructed = la_demo.reconstruct_from_principal_components(
        projected_data, eigenvecs, num_components, analysis_type, spectrogram.shape
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Spectrogram")
        fig_orig = visualizer.plot_spectrogram(spectrogram, st.session_state.sample_rate)
        st.plotly_chart(fig_orig, use_container_width=True)

    with col2:
        st.subheader("Reconstructed (PCA)")
        fig_recon = visualizer.plot_spectrogram(reconstructed, st.session_state.sample_rate)
        st.plotly_chart(fig_recon, use_container_width=True)

    # Reconstruction quality
    reconstruction_error = np.linalg.norm(spectrogram - reconstructed, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(spectrogram, 'fro') * 100
    
    st.metric("Reconstruction Error (%)", f"{relative_error:.2f}%")

    st.markdown("""
    ### Eigenvalue Analysis Theory:
    
    - **Covariance Matrix**: C = (1/n)XXᵀ captures relationships between variables
    - **Eigendecomposition**: C = QΛQᵀ where Q contains eigenvectors, Λ contains eigenvalues
    - **Principal Components**: Eigenvectors represent directions of maximum variance
    - **Dimensionality Reduction**: Project data onto top eigenvectors to reduce dimensions
    - **Applications**: Feature extraction, noise reduction, pattern recognition
    """)

def show_linear_transformations(audio_processor, la_demo, visualizer):
    st.header("🔄 Linear Transformations on Audio")
    
    if st.session_state.audio_data is None:
        st.warning("Please upload an audio file first.")
        return

    audio_data = st.session_state.audio_data
    sample_rate = st.session_state.sample_rate

    st.subheader("Apply Linear Transformations")
    
    # Transformation type selection
    transform_type = st.selectbox(
        "Select Transformation",
        [
            "Rotation Matrix",
            "Scaling Matrix", 
            "Shearing Matrix",
            "Random Orthogonal Matrix",
            "Low-pass Filter Matrix",
            "Mixing Matrix"
        ],
        help="Choose a linear transformation to apply"
    )

    # Parameters for different transformations
    if transform_type == "Rotation Matrix":
        angle = st.slider("Rotation Angle (degrees)", -180, 180, 45)
        transform_matrix = la_demo.create_rotation_matrix(angle)
        
    elif transform_type == "Scaling Matrix":
        scale_x = st.slider("X-axis Scale", 0.1, 3.0, 1.0, 0.1)
        scale_y = st.slider("Y-axis Scale", 0.1, 3.0, 1.0, 0.1)
        transform_matrix = la_demo.create_scaling_matrix(scale_x, scale_y)
        
    elif transform_type == "Shearing Matrix":
        shear_x = st.slider("X Shear", -2.0, 2.0, 0.5, 0.1)
        shear_y = st.slider("Y Shear", -2.0, 2.0, 0.0, 0.1)
        transform_matrix = la_demo.create_shearing_matrix(shear_x, shear_y)
        
    elif transform_type == "Random Orthogonal Matrix":
        matrix_size = st.slider("Matrix Size", 2, 10, 4)
        if st.button("Generate New Random Matrix"):
            st.session_state.random_matrix = la_demo.create_random_orthogonal_matrix(matrix_size)
        if 'random_matrix' not in st.session_state:
            st.session_state.random_matrix = la_demo.create_random_orthogonal_matrix(matrix_size)
        transform_matrix = st.session_state.random_matrix
        
    elif transform_type == "Low-pass Filter Matrix":
        cutoff_freq = st.slider("Cutoff Frequency", 0.1, 0.9, 0.3, 0.05)
        matrix_size = st.slider("Filter Size", 3, 15, 7, 2)
        transform_matrix = la_demo.create_lowpass_filter_matrix(matrix_size, cutoff_freq)
        
    else:  # Mixing Matrix
        num_channels = st.slider("Number of Channels", 2, 6, 3)
        transform_matrix = la_demo.create_mixing_matrix(num_channels)

    # Display transformation matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transformation Matrix")
        if transform_matrix.shape[0] <= 10 and transform_matrix.shape[1] <= 10:
            fig_matrix = visualizer.plot_matrix_heatmap(
                transform_matrix, f"{transform_type} Matrix"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
        else:
            st.write(f"Matrix shape: {transform_matrix.shape}")
            st.write("Matrix too large to display")

    with col2:
        st.subheader("Matrix Properties")
        det = np.linalg.det(transform_matrix) if transform_matrix.shape[0] == transform_matrix.shape[1] else "N/A"
        
        st.metric("Shape", f"{transform_matrix.shape[0]}×{transform_matrix.shape[1]}")
        if isinstance(det, (int, float)):
            st.metric("Determinant", f"{det:.4f}")
        else:
            st.write("Determinant: N/A (non-square)")
        
        if transform_matrix.shape[0] == transform_matrix.shape[1]:
            eigenvals = np.linalg.eigvals(transform_matrix)
            st.metric("Largest Eigenvalue", f"{np.max(np.abs(eigenvals)):.4f}")

    # Apply transformation to audio segments or spectrogram
    st.subheader("Apply Transformation to Audio")
    
    application_method = st.selectbox(
        "Application Method",
        ["Spectrogram Transform", "Segment Mixing", "Filter Application"],
        help="How to apply the transformation"
    )

    try:
        if application_method == "Spectrogram Transform":
            # Apply to spectrogram patches
            spectrogram = st.session_state.spectrogram
            transformed_spec = la_demo.apply_matrix_to_spectrogram_patches(
                spectrogram, transform_matrix
            )
            
            # Reconstruct audio
            transformed_audio = audio_processor.spectrogram_to_audio(
                transformed_spec, sample_rate
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                fig_orig = visualizer.plot_spectrogram(spectrogram, sample_rate)
                st.plotly_chart(fig_orig, use_container_width=True)
                st.audio(audio_processor.audio_to_bytes(audio_data, sample_rate), format='audio/wav')

            with col2:
                st.subheader("Transformed")
                fig_trans = visualizer.plot_spectrogram(transformed_spec, sample_rate)
                st.plotly_chart(fig_trans, use_container_width=True)
                st.audio(audio_processor.audio_to_bytes(transformed_audio, sample_rate), format='audio/wav')

        elif application_method == "Segment Mixing":
            # Apply to audio segments
            segments = audio_processor.segment_audio(audio_data, sample_rate, 0.5)
            if len(segments) >= transform_matrix.shape[1]:
                mixed_audio = la_demo.apply_mixing_matrix(segments[:transform_matrix.shape[1]], transform_matrix)
                
                st.subheader("Mixed Audio Channels")
                for i, channel in enumerate(mixed_audio):
                    st.write(f"Channel {i+1}")
                    st.audio(audio_processor.audio_to_bytes(channel, sample_rate), format='audio/wav')
            else:
                st.warning(f"Need at least {transform_matrix.shape[1]} segments. Current: {len(segments)}")

        elif application_method == "Filter Application":
            # Apply as convolution filter
            if transform_matrix.shape[0] == transform_matrix.shape[1] and transform_matrix.shape[0] <= 15:
                # Use matrix as convolution kernel
                kernel = transform_matrix[0, :]  # Use first row as filter
                filtered_audio = la_demo.apply_convolution_filter(audio_data, kernel)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Audio")
                    fig_orig = visualizer.plot_waveform(audio_data, sample_rate)
                    st.plotly_chart(fig_orig, use_container_width=True)
                    st.audio(audio_processor.audio_to_bytes(audio_data, sample_rate), format='audio/wav')

                with col2:
                    st.subheader("Filtered Audio")
                    fig_filt = visualizer.plot_waveform(filtered_audio, sample_rate)
                    st.plotly_chart(fig_filt, use_container_width=True)
                    st.audio(audio_processor.audio_to_bytes(filtered_audio, sample_rate), format='audio/wav')
            else:
                st.warning("Filter application requires a small square matrix.")

    except Exception as e:
        st.error(f"Error applying transformation: {str(e)}")

    st.markdown("""
    ### Linear Transformations Theory:
    
    - **Matrix Multiplication**: y = Ax transforms vector x to vector y
    - **Determinant**: Measures area/volume scaling factor
    - **Orthogonal Matrices**: Preserve distances and angles (det = ±1)
    - **Eigenvalues**: Indicate scaling along eigenvector directions
    - **Applications**: Rotation, scaling, filtering, mixing, feature extraction
    """)

def show_advanced_concepts(audio_processor, la_demo, visualizer):
    st.header("🧮 Advanced Linear Algebra Concepts")
    
    if st.session_state.spectrogram is None:
        st.warning("Please upload an audio file first.")
        return

    # Concept selection
    concept = st.selectbox(
        "Select Advanced Concept",
        [
            "Null Space & Range Analysis",
            "Matrix Calculus & Gradients", 
            "Pseudoinverse & Least Squares",
            "Matrix Factorizations",
            "Spectral Analysis"
        ]
    )

    if concept == "Null Space & Range Analysis":
        show_null_space_analysis(audio_processor, la_demo, visualizer)
    elif concept == "Matrix Calculus & Gradients":
        show_matrix_calculus(audio_processor, la_demo, visualizer)
    elif concept == "Pseudoinverse & Least Squares":
        show_pseudoinverse_analysis(audio_processor, la_demo, visualizer)
    elif concept == "Matrix Factorizations":
        show_matrix_factorizations(audio_processor, la_demo, visualizer)
    elif concept == "Spectral Analysis":
        show_spectral_analysis(audio_processor, la_demo, visualizer)

def show_null_space_analysis(audio_processor, la_demo, visualizer):
    st.subheader("Null Space & Range Analysis")
    
    spectrogram = st.session_state.spectrogram
    
    # Create a rank-deficient matrix from spectrogram
    rank_reduction = st.slider("Rank Reduction Factor", 1, 10, 3)
    reduced_matrix = la_demo.create_rank_deficient_matrix(spectrogram, rank_reduction)
    
    # Compute null space and range
    null_space, range_space, rank = la_demo.compute_null_and_range_space(reduced_matrix)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Matrix Properties:**")
        st.metric("Original Rank", np.linalg.matrix_rank(spectrogram))
        st.metric("Reduced Rank", rank)
        st.metric("Null Space Dimension", null_space.shape[1] if null_space is not None else 0)
        st.metric("Range Space Dimension", range_space.shape[1])

    with col2:
        if null_space is not None and null_space.shape[1] > 0:
            # Project audio onto null space
            null_projection = la_demo.project_onto_subspace(spectrogram, null_space)
            null_audio = audio_processor.spectrogram_to_audio(null_projection, st.session_state.sample_rate)
            
            st.write("**Null Space Projection (What's Lost):**")
            st.audio(audio_processor.audio_to_bytes(null_audio, st.session_state.sample_rate), format='audio/wav')
        
        # Project onto range space
        range_projection = la_demo.project_onto_subspace(spectrogram, range_space)
        range_audio = audio_processor.spectrogram_to_audio(range_projection, st.session_state.sample_rate)
        
        st.write("**Range Space Projection (What's Kept):**")
        st.audio(audio_processor.audio_to_bytes(range_audio, st.session_state.sample_rate), format='audio/wav')

    st.markdown("""
    **Theory:**
    - **Null Space**: N(A) = {x : Ax = 0} - vectors that map to zero
    - **Range Space**: R(A) = {Ax : x ∈ ℝⁿ} - all possible outputs
    - **Fundamental Theorem**: dim(N(A)) + dim(R(A)) = n (number of columns)
    """)

def show_matrix_calculus(audio_processor, la_demo, visualizer):
    st.subheader("Matrix Calculus & Gradient Descent")
    
    spectrogram = st.session_state.spectrogram
    
    # Set up optimization problem
    st.write("**Optimization Problem: Audio Denoising via Gradient Descent**")
    
    # Add noise to create optimization target
    noise_level = st.slider("Noise Level", 0.01, 0.5, 0.1, 0.01)
    noisy_spec = la_demo.add_noise_to_spectrogram(spectrogram, noise_level)
    
    # Optimization parameters
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    num_iterations = st.slider("Number of Iterations", 10, 500, 100, 10)
    
    if st.button("Run Gradient Descent"):
        # Run optimization
        optimized_spec, cost_history, gradient_norms = la_demo.gradient_descent_denoising(
            noisy_spec, spectrogram, learning_rate, num_iterations
        )
        
        # Store results in session state
        st.session_state.optimized_spec = optimized_spec
        st.session_state.cost_history = cost_history
        st.session_state.gradient_norms = gradient_norms

    if 'cost_history' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot cost function
            fig_cost = visualizer.plot_cost_function(st.session_state.cost_history)
            st.plotly_chart(fig_cost, use_container_width=True)

        with col2:
            # Plot gradient norms
            fig_grad = visualizer.plot_gradient_norms(st.session_state.gradient_norms)
            st.plotly_chart(fig_grad, use_container_width=True)

        # Audio comparison
        optimized_audio = audio_processor.spectrogram_to_audio(
            st.session_state.optimized_spec, st.session_state.sample_rate
        )
        noisy_audio = audio_processor.spectrogram_to_audio(
            noisy_spec, st.session_state.sample_rate
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Original**")
            st.audio(audio_processor.audio_to_bytes(
                st.session_state.audio_data, st.session_state.sample_rate
            ), format='audio/wav')

        with col2:
            st.write("**Noisy**")
            st.audio(audio_processor.audio_to_bytes(noisy_audio, st.session_state.sample_rate), format='audio/wav')

        with col3:
            st.write("**Denoised**")
            st.audio(audio_processor.audio_to_bytes(optimized_audio, st.session_state.sample_rate), format='audio/wav')

def show_pseudoinverse_analysis(audio_processor, la_demo, visualizer):
    st.subheader("Pseudoinverse & Least Squares")
    
    spectrogram = st.session_state.spectrogram
    
    # Create overdetermined system
    st.write("**Overdetermined System: Audio Interpolation**")
    
    # Subsample the spectrogram
    subsample_factor = st.slider("Subsample Factor", 2, 8, 4)
    subsampled_spec, indices = la_demo.subsample_spectrogram(spectrogram, subsample_factor)
    
    # Solve using pseudoinverse
    reconstructed_spec = la_demo.solve_with_pseudoinverse(subsampled_spec, spectrogram.shape, indices)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original**")
        fig_orig = visualizer.plot_spectrogram(spectrogram, st.session_state.sample_rate)
        st.plotly_chart(fig_orig, use_container_width=True)

    with col2:
        st.write("**Subsampled**")
        fig_sub = visualizer.plot_spectrogram(subsampled_spec, st.session_state.sample_rate)
        st.plotly_chart(fig_sub, use_container_width=True)

    with col3:
        st.write("**Reconstructed**")
        fig_recon = visualizer.plot_spectrogram(reconstructed_spec, st.session_state.sample_rate)
        st.plotly_chart(fig_recon, use_container_width=True)

    # Audio reconstruction
    reconstructed_audio = audio_processor.spectrogram_to_audio(
        reconstructed_spec, st.session_state.sample_rate
    )
    
    st.write("**Reconstructed Audio:**")
    st.audio(audio_processor.audio_to_bytes(reconstructed_audio, st.session_state.sample_rate), format='audio/wav')
    
    # Error analysis
    reconstruction_error = np.linalg.norm(spectrogram - reconstructed_spec, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(spectrogram, 'fro') * 100
    st.metric("Reconstruction Error (%)", f"{relative_error:.2f}%")

def show_matrix_factorizations(audio_processor, la_demo, visualizer):
    st.subheader("Matrix Factorizations")
    
    spectrogram = st.session_state.spectrogram
    
    factorization_type = st.selectbox(
        "Factorization Type",
        ["QR Decomposition", "Cholesky Decomposition", "LU Decomposition"]
    )
    
    try:
        if factorization_type == "QR Decomposition":
            Q, R = la_demo.qr_decomposition(spectrogram)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Q Matrix (Orthogonal)**")
                if Q.shape[1] <= 50:
                    fig_q = visualizer.plot_matrix_heatmap(Q[:, :50], "Q Matrix")
                    st.plotly_chart(fig_q, use_container_width=True)
                st.metric("Q Shape", f"{Q.shape[0]}×{Q.shape[1]}")

            with col2:
                st.write("**R Matrix (Upper Triangular)**")
                if R.shape[0] <= 50:
                    fig_r = visualizer.plot_matrix_heatmap(R[:50, :50], "R Matrix")
                    st.plotly_chart(fig_r, use_container_width=True)
                st.metric("R Shape", f"{R.shape[0]}×{R.shape[1]}")

        # Add other factorizations as needed
        
    except Exception as e:
        st.error(f"Error in factorization: {str(e)}")

def show_spectral_analysis(audio_processor, la_demo, visualizer):
    st.subheader("Spectral Analysis")
    
    spectrogram = st.session_state.spectrogram
    
    # Compute spectral properties
    spectral_props = la_demo.compute_spectral_properties(spectrogram)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Spectral Properties:**")
        st.metric("Spectral Norm", f"{spectral_props['spectral_norm']:.2e}")
        st.metric("Nuclear Norm", f"{spectral_props['nuclear_norm']:.2e}")
        st.metric("Effective Rank", spectral_props['effective_rank'])

    with col2:
        st.write("**Condition Numbers:**")
        st.metric("L2 Condition Number", f"{spectral_props['condition_number']:.2e}")
        st.metric("Stable Rank", f"{spectral_props['stable_rank']:.2f}")

if __name__ == "__main__":
    main()
