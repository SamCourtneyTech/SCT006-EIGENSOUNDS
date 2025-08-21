# Eigen-Sounds: Linear Algebra Through Audio

## Overview

Eigen-Sounds is an interactive educational application built with Streamlit that demonstrates linear algebra concepts through audio processing and compression. The application uses advanced mathematical techniques like Singular Value Decomposition (SVD), eigenvalue analysis, and matrix operations to process and compress audio files while providing visual feedback on the mathematical operations being performed.

The core concept revolves around converting audio signals into spectrograms (2D matrix representations) and then applying linear algebra transformations to demonstrate compression, noise reduction, and signal analysis. This approach makes abstract mathematical concepts tangible by showing their real-world applications in audio processing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit as the primary web framework, providing an interactive dashboard with multiple pages for different demonstrations. The interface is organized with a sidebar navigation system and utilizes session state management to maintain data persistence across user interactions. The layout employs a wide configuration to accommodate complex visualizations and multiple data displays simultaneously.

### Core Processing Pipeline
The system follows a modular architecture with three main processing components:

**AudioProcessor**: Handles all audio-related operations including STFT computation, spectrogram generation, and audio reconstruction using the Griffin-Lim algorithm. This component serves as the bridge between raw audio data and mathematical matrix representations.

**LinearAlgebraDemo**: Implements mathematical operations and demonstrations, focusing on SVD compression, matrix property analysis, and various decomposition techniques. This component translates abstract linear algebra concepts into practical audio processing applications.

**AudioVisualizer**: Manages all plotting and visualization tasks using Plotly for interactive graphs and Matplotlib for static plots. The visualization system provides real-time feedback on mathematical transformations and their effects on audio data.

### Data Flow Design
The application follows a transformation pipeline where audio files are converted to spectrograms (frequency-time matrices), mathematical operations are applied to these matrices, and results are converted back to audio or displayed as visualizations. Session state management ensures data persistence throughout the user's exploration of different mathematical concepts.

### Mathematical Framework
The core mathematical operations are built around matrix decomposition techniques, particularly SVD for compression demonstrations. The system includes analysis of matrix properties such as rank, condition numbers, and various norms to provide educational insights into the mathematical foundations of audio processing.

## External Dependencies

### Audio Processing Libraries
- **librosa**: Primary audio analysis library for STFT computation, spectrogram generation, and audio reconstruction
- **soundfile**: Audio file I/O operations for reading and writing various audio formats
- **scipy**: Signal processing functions, FFT operations, and advanced mathematical algorithms including SVD and matrix decompositions

### Mathematical and Scientific Computing
- **numpy**: Core numerical computing library for matrix operations and array manipulations
- **scipy.linalg**: Advanced linear algebra operations including SVD, QR decomposition, and matrix analysis
- **sklearn**: Machine learning utilities for cosine similarity calculations and PCA demonstrations

### Visualization and Interface
- **streamlit**: Web application framework providing the interactive user interface and session management
- **plotly**: Interactive plotting library for spectrograms, waveforms, and mathematical visualizations
- **matplotlib**: Static plotting capabilities for additional visualization options

### Supporting Libraries
- **io**: In-memory file operations for audio data handling
- **tempfile**: Temporary file management for audio processing workflows
- **os**: Operating system interface for file path operations

The application is designed to be self-contained with no external API dependencies, relying entirely on local computation for all audio processing and mathematical operations.

## Recent Changes

### Memory Management and Performance Optimizations (August 21, 2025)
Added comprehensive safeguards to prevent application crashes with large audio files:

- **File Size Limits**: 50MB maximum file size with user warnings
- **Duration Limits**: Automatic truncation to 30 seconds for performance  
- **Memory Management**: Automatic downsampling of large spectrograms during SVD operations
- **Progressive Processing**: Adaptive FFT sizes based on audio length
- **User Feedback**: Clear warnings and progress indicators for large file processing
- **Error Handling**: Graceful degradation instead of crashes for oversized inputs

These changes ensure stable operation while maintaining full functionality for typical audio analysis tasks.