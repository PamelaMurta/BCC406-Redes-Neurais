"""
Módulo de Utilitários Auxiliares

Este módulo fornece funções utilitárias para carregamento de configuração, logging e operações comuns.
"""

import yaml
import json
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    Carregar configuração de arquivo YAML.
    
    Args:
        config_path: Caminho para arquivo de configuração
    
    Returns:
        Dicionário de configuração
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, output_path: str) -> None:
    """
    Salvar configuração em arquivo YAML.
    
    Args:
        config: Dicionário de configuração
        output_path: Caminho para salvar configuração
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuração salva em: {output_path}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Definir sementes aleatórias para reprodutibilidade.
    
    Args:
        seed: Valor da semente aleatória
    """
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (se disponível)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"Sementes aleatórias definidas para: {seed}")


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configurar logging.
    
    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log (opcional)
        log_format: Formato de log personalizado (opcional)
    
    Returns:
        Logger configurado
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Obter nível de log
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configurar logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado no nível {log_level}")
    
    if log_file:
        logger.info(f"Logging para arquivo: {log_file}")
    
    return logger


def save_results(
    results: Dict,
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Salvar resultados em arquivo.
    
    Args:
        results: Dicionário de resultados
        output_path: Caminho para salvar resultados
        format: Formato ('json', 'yaml')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Converter arrays numpy para listas para serialização JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    results_converted = convert_numpy(results)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
    elif format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results_converted, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Formato não suportado: {format}")
    
    print(f"Resultados salvos em: {output_path}")


def load_results(input_path: str) -> Dict:
    """
    Carregar resultados de arquivo.
    
    Args:
        input_path: Caminho para arquivo de resultados
    
    Returns:
        Dicionário de resultados
    """
    ext = Path(input_path).suffix.lower()
    
    if ext == '.json':
        with open(input_path, 'r') as f:
            results = json.load(f)
    elif ext in ['.yaml', '.yml']:
        with open(input_path, 'r') as f:
            results = yaml.safe_load(f)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")
    
    return results


def ensure_dir(directory: str) -> None:
    """
    Garantir que o diretório existe.
    
    Args:
        directory: Caminho do diretório
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Obter diretório raiz do projeto.
    
    Returns:
        Caminho raiz do projeto
    """
    return Path(__file__).parent.parent.parent


def print_system_info() -> None:
    """Imprimir informações do sistema."""
    import platform
    import sys
    
    print("=" * 60)
    print("Informações do Sistema")
    print("=" * 60)
    print(f"Versão Python: {sys.version}")
    print(f"Plataforma: {platform.platform()}")
    print(f"Processador: {platform.processor()}")
    
    # Verificar TensorFlow e GPU
    try:
        import tensorflow as tf
        print(f"Versão TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs disponíveis: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("Nenhuma GPU disponível (usando CPU)")
    except ImportError:
        print("TensorFlow não instalado")
    
    # Verificar outras bibliotecas
    try:
        import sklearn
        print(f"Versão scikit-learn: {sklearn.__version__}")
    except ImportError:
        pass
    
    try:
        import librosa
        print(f"Versão librosa: {librosa.__version__}")
    except ImportError:
        pass
    
    print("=" * 60)


def format_time(seconds: float) -> str:
    """
    Formatar tempo em formato legível.
    
    Args:
        seconds: Tempo em segundos
    
    Returns:
        String de tempo formatada
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """
    Contar parâmetros treináveis em um modelo.
    
    Args:
        model: Instância do modelo (Keras ou sklearn)
    
    Returns:
        Número de parâmetros
    """
    try:
        # Para modelos Keras
        return model.count_params()
    except AttributeError:
        # Para modelos sklearn
        try:
            return sum(tree.tree_.node_count for tree in model.estimators_)
        except:
            return 0


if __name__ == "__main__":
    print("Módulo de Utilitários Auxiliares")
    print("=" * 50)
    print("\nFunções utilitárias disponíveis:")
    print("- load_config(): Carregar configuração de YAML")
    print("- save_config(): Salvar configuração em YAML")
    print("- set_random_seeds(): Definir sementes para reprodutibilidade")
    print("- setup_logging(): Configurar logging")
    print("- save_results(): Salvar resultados em JSON/YAML")
    print("- load_results(): Carregar resultados de arquivo")
    print("- print_system_info(): Exibir informações do sistema")
    
    print("\nExemplo:")
    print_system_info()
