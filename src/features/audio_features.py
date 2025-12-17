"""
Audio Feature Extraction Module

This module implements feature extraction as specified in Section 3.3:
- 40 MFCCs (Mel-Frequency Cepstral Coefficients)
- Pitch features (F0 using pYIN algorithm)
- Spectral features (Centroid, Rolloff, ZCR)

Total: 47 features per time frame (for sequential input to CNN)
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0,
    fmax: Optional[float] = 8000
) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract (default: 40)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length for STFT (default: 512)
        n_mels: Number of Mel bands (default: 128)
        fmin: Minimum frequency (default: 0)
        fmax: Maximum frequency (default: 8000)
    
    Returns:
        MFCC features of shape (n_mfcc, T) where T is number of time frames
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    return mfccs


def extract_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 80.0,
    fmax: float = 400.0,
    frame_length: int = 2048
) -> Dict[str, float]:
    """
    Extract pitch features using pYIN algorithm.
    
    Returns mean, std, min, max of F0 (4 features).
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        fmin: Minimum frequency in Hz (default: 80.0)
        fmax: Maximum frequency in Hz (default: 400.0)
        frame_length: Frame length for analysis (default: 2048)
    
    Returns:
        Dictionary with pitch statistics: {mean, std, min, max}
    """
    # Extract F0 using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length
    )
    
    # Filter out unvoiced frames (NaN values)
    f0_voiced = f0[~np.isnan(f0)]
    
    if len(f0_voiced) > 0:
        pitch_features = {
            'mean': np.mean(f0_voiced),
            'std': np.std(f0_voiced),
            'min': np.min(f0_voiced),
            'max': np.max(f0_voiced)
        }
    else:
        # If no voiced frames detected, return zeros
        pitch_features = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return pitch_features


def extract_pitch_contour(
    audio: np.ndarray,
    sr: int,
    fmin: float = 80.0,
    fmax: float = 400.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract pitch contour (F0 over time) for sequential features.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        fmin: Minimum frequency in Hz (default: 80.0)
        fmax: Maximum frequency in Hz (default: 400.0)
        frame_length: Frame length for analysis (default: 2048)
        hop_length: Hop length for analysis (default: 512)
    
    Returns:
        F0 contour of shape (T,) where T is number of time frames
        NaN values are replaced with 0
    """
    # Extract F0 using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0


def extract_spectral_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract spectral features: Spectral Centroid, Spectral Rolloff, Zero Crossing Rate.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length for STFT (default: 512)
    
    Returns:
        Dictionary with spectral features, each of shape (T,)
    """
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zcr': zcr
    }


def extract_all_features(
    audio: np.ndarray,
    sr: int,
    config: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all features for CNN (sequential features).
    
    Features extracted:
    - 40 MFCCs (40, T)
    - 1 Pitch contour (T,)
    - 3 Spectral features (3, T)
    - 3 Pitch statistics replicated across time (3, T)
    Total: 47 features per time frame
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        config: Configuration dictionary (optional)
    
    Returns:
        Dictionary containing all features:
        - 'sequential': Combined sequential features of shape (T, 47)
        - 'mfcc': MFCC features (40, T)
        - 'pitch_contour': Pitch contour (T,)
        - 'pitch_stats': Pitch statistics (4,)
        - 'spectral': Spectral features (3, T)
    """
    if config is None:
        config = {}
    
    # Get feature extraction parameters from config
    mfcc_config = config.get('mfcc', {})
    pitch_config = config.get('pitch', {})
    
    # Extract MFCCs
    mfccs = extract_mfcc(
        audio, sr,
        n_mfcc=mfcc_config.get('n_mfcc', 40),
        n_fft=mfcc_config.get('n_fft', 2048),
        hop_length=mfcc_config.get('hop_length', 512),
        n_mels=mfcc_config.get('n_mels', 128),
        fmin=mfcc_config.get('fmin', 0),
        fmax=mfcc_config.get('fmax', 8000)
    )
    
    # Extract pitch statistics
    pitch_stats = extract_pitch(
        audio, sr,
        fmin=pitch_config.get('fmin', 80.0),
        fmax=pitch_config.get('fmax', 400.0)
    )
    
    # Extract pitch contour
    pitch_contour = extract_pitch_contour(
        audio, sr,
        fmin=pitch_config.get('fmin', 80.0),
        fmax=pitch_config.get('fmax', 400.0),
        hop_length=mfcc_config.get('hop_length', 512)
    )
    
    # Extract spectral features
    spectral_features = extract_spectral_features(
        audio, sr,
        n_fft=mfcc_config.get('n_fft', 2048),
        hop_length=mfcc_config.get('hop_length', 512)
    )
    
    # Get number of time frames (from MFCCs)
    n_frames = mfccs.shape[1]
    
    # Adjust pitch contour length to match MFCCs
    if len(pitch_contour) > n_frames:
        pitch_contour = pitch_contour[:n_frames]
    elif len(pitch_contour) < n_frames:
        pitch_contour = np.pad(pitch_contour, (0, n_frames - len(pitch_contour)), mode='edge')
    
    # Adjust spectral features length
    for key in spectral_features:
        if len(spectral_features[key]) > n_frames:
            spectral_features[key] = spectral_features[key][:n_frames]
        elif len(spectral_features[key]) < n_frames:
            pad_width = n_frames - len(spectral_features[key])
            spectral_features[key] = np.pad(spectral_features[key], (0, pad_width), mode='edge')
    
    # Combine all sequential features
    # Shape: (T, 47) = (T, 40 MFCCs + 1 pitch contour + 3 spectral + 3 pitch stats replicated)
    sequential_features = []
    
    # Transpose MFCCs from (40, T) to (T, 40)
    sequential_features.append(mfccs.T)
    
    # Add pitch contour (T, 1)
    sequential_features.append(pitch_contour.reshape(-1, 1))
    
    # Add spectral features (T, 3)
    spectral_array = np.column_stack([
        spectral_features['spectral_centroid'],
        spectral_features['spectral_rolloff'],
        spectral_features['zcr']
    ])
    sequential_features.append(spectral_array)
    
    # Add pitch statistics replicated across time (T, 3) - mean, std, range
    pitch_range = pitch_stats['max'] - pitch_stats['min']
    pitch_stats_array = np.tile([
        pitch_stats['mean'],
        pitch_stats['std'],
        pitch_range
    ], (n_frames, 1))
    sequential_features.append(pitch_stats_array)
    
    # Concatenate all features
    combined = np.concatenate(sequential_features, axis=1)
    
    return {
        'sequential': combined,  # (T, 47)
        'mfcc': mfccs,  # (40, T)
        'pitch_contour': pitch_contour,  # (T,)
        'pitch_stats': pitch_stats,  # dict with 4 values
        'spectral': spectral_features  # dict with 3 arrays of shape (T,)
    }


def pad_features_to_max_length(
    features: np.ndarray,
    max_frames: int = 100
) -> np.ndarray:
    """
    Pad or truncate feature sequence to fixed length.
    
    Args:
        features: Feature array of shape (T, F) where T is time, F is features
        max_frames: Target number of frames (default: 100)
    
    Returns:
        Padded/truncated features of shape (max_frames, F)
    """
    current_frames = features.shape[0]
    
    if current_frames > max_frames:
        # Truncate from center
        start = (current_frames - max_frames) // 2
        features_fixed = features[start:start + max_frames, :]
    elif current_frames < max_frames:
        # Pad with zeros
        pad_amount = max_frames - current_frames
        pad_top = pad_amount // 2
        pad_bottom = pad_amount - pad_top
        features_fixed = np.pad(
            features,
            ((pad_top, pad_bottom), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        features_fixed = features
    
    return features_fixed


if __name__ == "__main__":
    # Example usage
    print("Audio Feature Extraction Module")
    print("=" * 50)
    print("\nFeatures extracted:")
    print("- 40 MFCCs (Mel-Frequency Cepstral Coefficients)")
    print("- 4 Pitch statistics (mean, std, min, max)")
    print("- 3 Spectral features (centroid, rolloff, ZCR)")
    print("\nTotal: 47 features per time frame")
