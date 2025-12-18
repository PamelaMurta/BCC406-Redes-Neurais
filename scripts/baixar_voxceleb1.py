"""
Script para Download do VoxCeleb1 no Google Colab
Baixa e extrai o dataset VoxCeleb1 em partes
"""

import os
import wget
import hashlib
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

# URLs e checksums do VoxCeleb1
VOXCELEB1_URLS = {
    'dev_a': {
        'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa',
        'md5': 'e395d020928bc15670b570a21695ed96'
    },
    'dev_b': {
        'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab',
        'md5': 'bbfaaccefab65d82b21903e81a8a8020'
    },
    'dev_c': {
        'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac',
        'md5': '017d579a2a96a077f40042ec33e51512'
    },
    'dev_d': {
        'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad',
        'md5': '7bb1e9f70fddc7a678fa998ea8b3ba19'
    },
    'test': {
        'url': 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip',
        'md5': '185fdc63c3c739954633d50379a3d102'
    }
}

def verificar_md5(arquivo, md5_esperado):
    """Verificar checksum MD5 do arquivo"""
    print(f"Verificando MD5 de {arquivo}...")
    md5 = hashlib.md5()
    with open(arquivo, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    
    md5_calculado = md5.hexdigest()
    if md5_calculado == md5_esperado:
        print(f"✓ MD5 verificado: {md5_calculado}")
        return True
    else:
        print(f"✗ MD5 não corresponde!")
        print(f"  Esperado: {md5_esperado}")
        print(f"  Obtido:   {md5_calculado}")
        return False

def baixar_arquivo(url, destino, md5_esperado=None):
    """Baixar arquivo com verificação de MD5"""
    if os.path.exists(destino):
        print(f"Arquivo já existe: {destino}")
        if md5_esperado and not verificar_md5(destino, md5_esperado):
            print("MD5 não corresponde. Baixando novamente...")
            os.remove(destino)
        else:
            return destino
    
    print(f"\nBaixando: {url}")
    print(f"Destino: {destino}")
    
    try:
        wget.download(url, destino)
        print("\n✓ Download concluído")
        
        if md5_esperado:
            if not verificar_md5(destino, md5_esperado):
                raise ValueError("MD5 não corresponde após download!")
        
        return destino
    
    except Exception as e:
        print(f"\n✗ Erro no download: {e}")
        if os.path.exists(destino):
            os.remove(destino)
        raise

def concatenar_partes(partes, saida):
    """Concatenar partes do arquivo"""
    print(f"\nConcatenando {len(partes)} partes...")
    with open(saida, 'wb') as arquivo_saida:
        for parte in partes:
            print(f"  Processando: {parte}")
            with open(parte, 'rb') as arquivo_parte:
                arquivo_saida.write(arquivo_parte.read())
    
    print(f"✓ Arquivo concatenado: {saida}")
    return saida

def extrair_arquivo(arquivo, destino):
    """Extrair arquivo comprimido"""
    print(f"\nExtraindo: {arquivo}")
    print(f"Destino: {destino}")
    
    Path(destino).mkdir(parents=True, exist_ok=True)
    
    if arquivo.endswith('.zip'):
        with zipfile.ZipFile(arquivo, 'r') as zip_ref:
            zip_ref.extractall(destino)
    elif arquivo.endswith('.tar') or arquivo.endswith('.tar.gz'):
        with tarfile.open(arquivo, 'r:*') as tar_ref:
            tar_ref.extractall(destino)
    else:
        raise ValueError(f"Formato não suportado: {arquivo}")
    
    print(f"✓ Extração concluída")

def baixar_voxceleb1_completo(diretorio_saida='data/raw', apenas_teste=False):
    """
    Baixar e preparar dataset VoxCeleb1 completo
    
    Args:
        diretorio_saida: Diretório onde salvar os dados
        apenas_teste: Se True, baixa apenas o conjunto de teste (menor)
    """
    print("=" * 80)
    print("DOWNLOAD DO VOXCELEB1 DATASET")
    print("=" * 80)
    print()
    
    # Criar diretórios
    dir_download = Path('downloads')
    dir_saida = Path(diretorio_saida)
    dir_download.mkdir(exist_ok=True)
    dir_saida.mkdir(parents=True, exist_ok=True)
    
    # Baixar conjunto de teste
    print("\n1. BAIXANDO CONJUNTO DE TESTE")
    print("-" * 80)
    arquivo_teste = dir_download / 'vox1_test_wav.zip'
    baixar_arquivo(
        VOXCELEB1_URLS['test']['url'],
        str(arquivo_teste),
        VOXCELEB1_URLS['test']['md5']
    )
    
    print("\n2. EXTRAINDO CONJUNTO DE TESTE")
    print("-" * 80)
    extrair_arquivo(str(arquivo_teste), str(dir_saida))
    
    if apenas_teste:
        print("\n✓ Download do conjunto de teste concluído!")
        return
    
    # Baixar conjunto de desenvolvimento (partes)
    print("\n3. BAIXANDO CONJUNTO DE DESENVOLVIMENTO (4 PARTES)")
    print("-" * 80)
    
    partes_dev = []
    for nome, info in [('dev_a', VOXCELEB1_URLS['dev_a']),
                       ('dev_b', VOXCELEB1_URLS['dev_b']),
                       ('dev_c', VOXCELEB1_URLS['dev_c']),
                       ('dev_d', VOXCELEB1_URLS['dev_d'])]:
        
        arquivo_parte = dir_download / f'vox1_dev_wav_part{nome[-1]}'
        baixar_arquivo(info['url'], str(arquivo_parte), info['md5'])
        partes_dev.append(str(arquivo_parte))
    
    # Concatenar partes
    print("\n4. CONCATENANDO PARTES DO DESENVOLVIMENTO")
    print("-" * 80)
    arquivo_dev = dir_download / 'vox1_dev_wav.zip'
    concatenar_partes(partes_dev, str(arquivo_dev))
    
    # Verificar MD5 do arquivo concatenado
    md5_concatenado = 'ae63e55b951748cc486645f532ba230b'
    if not verificar_md5(str(arquivo_dev), md5_concatenado):
        raise ValueError("MD5 do arquivo concatenado não corresponde!")
    
    # Extrair desenvolvimento
    print("\n5. EXTRAINDO CONJUNTO DE DESENVOLVIMENTO")
    print("-" * 80)
    extrair_arquivo(str(arquivo_dev), str(dir_saida))
    
    # Limpar arquivos temporários (opcional)
    print("\n6. LIMPEZA (OPCIONAL)")
    print("-" * 80)
    resposta = input("Deseja remover arquivos de download? (s/n): ")
    if resposta.lower() == 's':
        for parte in partes_dev:
            os.remove(parte)
            print(f"  Removido: {parte}")
        os.remove(arquivo_teste)
        os.remove(arquivo_dev)
        print("✓ Arquivos de download removidos")
    
    print("\n" + "=" * 80)
    print("✓ DOWNLOAD E EXTRAÇÃO CONCLUÍDOS!")
    print("=" * 80)
    print(f"\nDataset disponível em: {dir_saida}")
    print("\nEstrutura esperada:")
    print(f"  {dir_saida}/")
    print("    ├── wav/")
    print("    │   ├── id00001/")
    print("    │   ├── id00002/")
    print("    │   └── ...")
    print("    └── ...")

def baixar_voxceleb1_amostra(diretorio_saida='data/raw', num_falantes=10):
    """
    Baixar apenas uma amostra do VoxCeleb1 para testes rápidos
    
    Args:
        diretorio_saida: Diretório onde salvar os dados
        num_falantes: Número de falantes para baixar
    """
    print("=" * 80)
    print(f"DOWNLOAD DE AMOSTRA DO VOXCELEB1 ({num_falantes} falantes)")
    print("=" * 80)
    print()
    
    # Por enquanto, baixa apenas o teste que é menor
    baixar_voxceleb1_completo(diretorio_saida, apenas_teste=True)
    
    print(f"\n✓ Use os primeiros {num_falantes} falantes para treinamento rápido")

if __name__ == "__main__":
    import sys
    
    print("DOWNLOAD DO VOXCELEB1")
    print("=" * 80)
    print()
    print("Opções:")
    print("  1. Baixar apenas conjunto de teste (~5GB)")
    print("  2. Baixar dataset completo (~38GB)")
    print("  3. Sair")
    print()
    
    opcao = input("Escolha uma opção (1-3): ")
    
    if opcao == '1':
        baixar_voxceleb1_completo(apenas_teste=True)
    elif opcao == '2':
        print("\n⚠️  ATENÇÃO: O download completo requer ~38GB de espaço!")
        confirmar = input("Deseja continuar? (s/n): ")
        if confirmar.lower() == 's':
            baixar_voxceleb1_completo()
        else:
            print("Download cancelado.")
    else:
        print("Saindo...")
