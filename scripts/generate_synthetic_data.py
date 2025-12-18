"""
Script para Gerar Dataset Sintético

Gera áudio sintético para simular o VoxCeleb1 durante desenvolvimento/testes.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse


def generate_voice_audio(duration=3.0, sr=16000, base_freq=120, variation=0.2):
    """
    Gera áudio sintético simulando voz humana.
    
    Args:
        duration: Duração em segundos
        sr: Taxa de amostragem
        base_freq: Frequência fundamental base (Hz)
        variation: Variação na frequência
    
    Returns:
        Array numpy com áudio gerado
    """
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)
    
    # Frequência fundamental variável (simula entonação)
    f0 = base_freq * (1 + variation * np.sin(2 * np.pi * 0.5 * t))
    
    # Harmônicos (simulam timbre vocal)
    audio = np.zeros(n_samples)
    harmonics = [1, 2, 3, 4, 5, 6, 8]
    amplitudes = [1.0, 0.5, 0.3, 0.15, 0.1, 0.05, 0.03]
    
    for h, amp in zip(harmonics, amplitudes):
        phase = 2 * np.pi * np.random.random()
        audio += amp * np.sin(2 * np.pi * h * f0 * t + phase)
    
    # Adicionar formantes (ressonâncias vocais)
    formants = [800, 1200, 2500]
    for formant in formants:
        modulation = 0.3 * np.sin(2 * np.pi * formant * t)
        audio += modulation * np.random.randn(n_samples) * 0.05
    
    # Envelope de amplitude (simula padrão de fala)
    envelope = np.ones(n_samples)
    n_segments = int(duration * 3)  # 3 segmentos por segundo
    segment_length = n_samples // n_segments
    
    for i in range(n_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, n_samples)
        amplitude = 0.5 + 0.5 * np.random.random()
        envelope[start:end] *= amplitude
    
    audio *= envelope
    
    # Adicionar ruído de fundo leve
    noise = np.random.randn(n_samples) * 0.02
    audio += noise
    
    # Normalizar
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def generate_dataset(output_dir, num_speakers=10, samples_per_speaker=20, 
                    min_duration=2.0, max_duration=5.0, sr=16000):
    """
    Gera dataset sintético completo.
    
    Args:
        output_dir: Diretório de saída
        num_speakers: Número de falantes
        samples_per_speaker: Amostras por falante
        min_duration: Duração mínima do áudio (segundos)
        max_duration: Duração máxima do áudio (segundos)
        sr: Taxa de amostragem
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Gerando dataset sintético...")
    print(f"  Falantes: {num_speakers}")
    print(f"  Amostras por falante: {samples_per_speaker}")
    print(f"  Total de arquivos: {num_speakers * samples_per_speaker}")
    print(f"  Diretório de saída: {output_dir}")
    print()
    
    # Características vocais diferentes para cada falante
    speaker_profiles = []
    for i in range(num_speakers):
        profile = {
            'id': f'id{i:05d}',
            'base_freq': 80 + i * 15 + np.random.randn() * 10,  # 80-230 Hz
            'variation': 0.1 + np.random.random() * 0.3,
        }
        speaker_profiles.append(profile)
    
    total_files = num_speakers * samples_per_speaker
    with tqdm(total=total_files, desc="Gerando arquivos") as pbar:
        for speaker in speaker_profiles:
            speaker_dir = output_dir / speaker['id']
            speaker_dir.mkdir(exist_ok=True)
            
            for sample_idx in range(samples_per_speaker):
                # Duração aleatória
                duration = min_duration + np.random.random() * (max_duration - min_duration)
                
                # Pequena variação nas características para cada amostra
                base_freq = speaker['base_freq'] + np.random.randn() * 5
                variation = speaker['variation'] + np.random.randn() * 0.05
                
                # Gerar áudio
                audio = generate_voice_audio(
                    duration=duration,
                    sr=sr,
                    base_freq=base_freq,
                    variation=variation
                )
                
                # Salvar arquivo
                filename = f"{speaker['id']}_sample_{sample_idx:03d}.wav"
                filepath = speaker_dir / filename
                sf.write(filepath, audio, sr)
                
                pbar.update(1)
    
    print(f"\n✓ Dataset gerado com sucesso em: {output_dir}")
    print(f"  Total de arquivos: {total_files}")
    print(f"  Tamanho aproximado: {total_files * 3 * sr * 4 / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Gerar dataset sintético para testes')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Diretório de saída (default: data/raw)')
    parser.add_argument('--num-speakers', type=int, default=10,
                       help='Número de falantes (default: 10)')
    parser.add_argument('--samples-per-speaker', type=int, default=20,
                       help='Amostras por falante (default: 20)')
    parser.add_argument('--min-duration', type=float, default=2.0,
                       help='Duração mínima em segundos (default: 2.0)')
    parser.add_argument('--max-duration', type=float, default=5.0,
                       help='Duração máxima em segundos (default: 5.0)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Taxa de amostragem (default: 16000)')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        samples_per_speaker=args.samples_per_speaker,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sr=args.sample_rate
    )


if __name__ == '__main__':
    main()
