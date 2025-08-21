import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

class AudioVisualizer:
    """Handles all visualization tasks for the Eigen-Sounds application."""
    
    def __init__(self):
        self.color_scale = 'Viridis'
        self.default_height = 400
    
    def plot_waveform(self, audio_data, sample_rate):
        """Plot audio waveform."""
        try:
            time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=audio_data,
                mode='lines',
                name='Amplitude',
                line=dict(width=1)
            ))
            
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=self.default_height,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting waveform: {str(e)}")
    
    def plot_spectrogram(self, spectrogram, sample_rate):
        """Plot spectrogram as heatmap."""
        try:
            # Create time and frequency axes
            time_frames = spectrogram.shape[1]
            freq_bins = spectrogram.shape[0]
            
            # Time axis (in seconds)
            hop_length = 512  # Default hop length
            time_axis = np.linspace(0, time_frames * hop_length / sample_rate, time_frames)
            
            # Frequency axis (in Hz)
            freq_axis = np.linspace(0, sample_rate / 2, freq_bins)
            
            fig = go.Figure(data=go.Heatmap(
                z=spectrogram,
                x=time_axis,
                y=freq_axis,
                colorscale=self.color_scale,
                colorbar=dict(title="Magnitude (dB)")
            ))
            
            fig.update_layout(
                title="Spectrogram",
                xaxis_title="Time (seconds)",
                yaxis_title="Frequency (Hz)",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting spectrogram: {str(e)}")
    
    def plot_matrix_heatmap(self, matrix, title="Matrix"):
        """Plot matrix as heatmap."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                colorscale=self.color_scale,
                colorbar=dict(title="Value")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Column Index",
                yaxis_title="Row Index",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting matrix heatmap: {str(e)}")
    
    def plot_singular_values(self, singular_values, k_components=None):
        """Plot singular value spectrum."""
        try:
            fig = go.Figure()
            
            # Plot all singular values
            fig.add_trace(go.Scatter(
                x=np.arange(len(singular_values)),
                y=singular_values,
                mode='lines+markers',
                name='Singular Values',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Highlight selected components
            if k_components is not None and k_components <= len(singular_values):
                fig.add_trace(go.Scatter(
                    x=np.arange(k_components),
                    y=singular_values[:k_components],
                    mode='markers',
                    name=f'Selected (k={k_components})',
                    marker=dict(color='red', size=8)
                ))
                
                # Add vertical line
                fig.add_vline(
                    x=k_components - 0.5,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"k = {k_components}"
                )
            
            fig.update_layout(
                title="Singular Value Spectrum",
                xaxis_title="Component Index",
                yaxis_title="Singular Value",
                yaxis_type="log",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting singular values: {str(e)}")
    
    def plot_similarity_matrix(self, similarity_matrix):
        """Plot audio similarity matrix."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Cosine Similarity")
            ))
            
            fig.update_layout(
                title="Audio Segment Similarity Matrix",
                xaxis_title="Segment Index",
                yaxis_title="Segment Index",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting similarity matrix: {str(e)}")
    
    def plot_eigenvalue_spectrum(self, eigenvalues):
        """Plot eigenvalue spectrum."""
        try:
            # Plot positive eigenvalues only for better visualization
            positive_eigenvals = eigenvalues[eigenvalues > 0]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(positive_eigenvals)),
                y=positive_eigenvals,
                mode='lines+markers',
                name='Eigenvalues',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Eigenvalue Spectrum",
                xaxis_title="Eigenvalue Index",
                yaxis_title="Eigenvalue",
                yaxis_type="log",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting eigenvalue spectrum: {str(e)}")
    
    def plot_eigenvector(self, eigenvector, title="Eigenvector", analysis_type="Frequency"):
        """Plot eigenvector components."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(eigenvector)),
                y=eigenvector,
                mode='lines+markers',
                name='Component Value',
                line=dict(width=2),
                marker=dict(size=3)
            ))
            
            xlabel = "Frequency Bin" if "Frequency" in analysis_type else "Time Frame"
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title="Component Value",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting eigenvector: {str(e)}")
    
    def plot_cost_function(self, cost_history):
        """Plot cost function over iterations."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(cost_history)),
                y=cost_history,
                mode='lines',
                name='Cost',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Cost Function During Optimization",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                yaxis_type="log",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting cost function: {str(e)}")
    
    def plot_gradient_norms(self, gradient_norms):
        """Plot gradient norms over iterations."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(gradient_norms)),
                y=gradient_norms,
                mode='lines',
                name='Gradient Norm',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Gradient Norm During Optimization",
                xaxis_title="Iteration",
                yaxis_title="Gradient Norm",
                yaxis_type="log",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting gradient norms: {str(e)}")
    
    def plot_vector_comparison(self, vector1, vector2, labels=None):
        """Plot comparison between two vectors."""
        try:
            if labels is None:
                labels = ['Vector 1', 'Vector 2']
            
            fig = go.Figure()
            
            x_axis = np.arange(len(vector1))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=vector1,
                mode='lines+markers',
                name=labels[0],
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=vector2,
                mode='lines+markers',
                name=labels[1],
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title="Vector Comparison",
                xaxis_title="Index",
                yaxis_title="Value",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting vector comparison: {str(e)}")
    
    def plot_3d_scatter(self, data_3d, color_values=None, title="3D Visualization"):
        """Plot 3D scatter plot."""
        try:
            if color_values is not None:
                fig = go.Figure(data=go.Scatter3d(
                    x=data_3d[:, 0],
                    y=data_3d[:, 1],
                    z=data_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_values,
                        colorscale=self.color_scale,
                        showscale=True
                    )
                ))
            else:
                fig = go.Figure(data=go.Scatter3d(
                    x=data_3d[:, 0],
                    y=data_3d[:, 1],
                    z=data_3d[:, 2],
                    mode='markers',
                    marker=dict(size=5)
                ))
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    zaxis_title='Component 3'
                ),
                height=500
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting 3D scatter: {str(e)}")
    
    def plot_reconstruction_comparison(self, original, reconstructed):
        """Plot side-by-side comparison of original and reconstructed spectrograms."""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original', 'Reconstructed'),
                horizontal_spacing=0.1
            )
            
            # Original
            fig.add_trace(
                go.Heatmap(z=original, colorscale=self.color_scale, showscale=False),
                row=1, col=1
            )
            
            # Reconstructed
            fig.add_trace(
                go.Heatmap(z=reconstructed, colorscale=self.color_scale, showscale=True),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Reconstruction Comparison",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting reconstruction comparison: {str(e)}")
    
    def plot_frequency_response(self, frequencies, response):
        """Plot frequency response."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=response,
                mode='lines',
                name='Frequency Response',
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title="Frequency Response",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=self.default_height
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting frequency response: {str(e)}")
