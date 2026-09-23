"""
08_gravar_identificar_cadastrar.py

Funcionalidades:

1 - Gravar um novo áudio e identificar uma pessoa conhecida.
2 - Gravar um áudio de uma pessoa desconhecida.
3 - Cadastrar uma nova pessoa gravando vários áudios.
4 - Treinar novamente o modelo de pessoas após o cadastro.

O modo de identificação NÃO altera o dataset.

O modo de cadastro:
- Cria uma pasta para a pessoa.
- Grava vários arquivos WAV.
- Executa novamente o script 03_treinar_pessoas.py.
"""

from pathlib import Path
from datetime import datetime
import subprocess
import sys

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import joblib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PESSOAS_DIR = BASE_DIR / "dataset_pessoas"
MODELOS_DIR = BASE_DIR / "modelos"

MODELO_PESSOAS_PATH = MODELOS_DIR / "modelo_pessoas.pkl"
MODELO_EMOCOES_PATH = MODELOS_DIR / "modelo_emocoes.pkl"

TESTES_DIR = BASE_DIR / "reunioes" / "reuniao_01" / "testes"
GRAVACOES_DEMONSTRACAO_DIR = (
    BASE_DIR / "reunioes" / "reuniao_01" / "gravacoes_demonstracao"
)

TREINAR_PESSOAS_SCRIPT = BASE_DIR / "src" / "03_treinar_pessoas.py"

SAMPLE_RATE = 16000
CANAIS = 1

DURACAO_PADRAO = 4

CONFIANCA_MINIMA_DESCONHECIDO = 0.70


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def limpar_nome_pessoa(nome: str) -> str:
    """
    Normaliza o nome utilizado para criar a pasta da pessoa.
    """

    nome = nome.strip().lower()
    nome = nome.replace(" ", "_")

    caracteres_permitidos = (
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "_-"
    )

    nome_limpo = "".join(
        caractere
        for caractere in nome
        if caractere in caracteres_permitidos
    )

    return nome_limpo


def carregar_modelo(caminho: Path):
    """
    Carrega modelos salvos diretamente ou dentro de dicionários.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}"
        )

    objeto = joblib.load(caminho)

    if isinstance(objeto, dict):
        modelo = (
            objeto.get("modelo")
            or objeto.get("model")
            or objeto.get("classificador")
        )

        classes = objeto.get("classes")

        if modelo is None:
            raise ValueError(
                f"O arquivo {caminho.name} é um dicionário, "
                "mas não contém a chave 'modelo', 'model' "
                "ou 'classificador'."
            )

        if classes is None and hasattr(modelo, "classes_"):
            classes = modelo.classes_

        return modelo, np.array(classes)

    modelo = objeto

    if not hasattr(modelo, "predict"):
        raise ValueError(
            f"O arquivo {caminho.name} não contém um modelo válido."
        )

    classes = getattr(modelo, "classes_", None)

    if classes is None:
        raise ValueError(
            f"Não foi possível identificar as classes de {caminho.name}."
        )

    return modelo, np.array(classes)


def extrair_features(caminho_audio: Path) -> np.ndarray:
    """
    Extrai exatamente as mesmas 38 características utilizadas
    no treinamento atual do projeto.

    Total:
    - MFCC: 13 médias + 13 desvios = 26
    - RMS: média + desvio = 2
    - ZCR: média + desvio = 2
    - Centroid: média + desvio = 2
    - Bandwidth: média + desvio = 2
    - Rolloff: média + desvio = 2
    - Pitch: média + desvio = 2

    Total final: 38 características.
    """

    audio, sr = librosa.load(
        caminho_audio,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("O áudio está vazio.")

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # RMS
    rms = librosa.feature.rms(y=audio)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]

    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )[0]

    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Pitch
    try:
        pitch = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        pitch = pitch[np.isfinite(pitch)]

        if len(pitch) == 0:
            pitch_mean = 0.0
            pitch_std = 0.0
        else:
            pitch_mean = np.mean(pitch)
            pitch_std = np.std(pitch)

    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0

    features.append(pitch_mean)
    features.append(pitch_std)

    vetor = np.array(features, dtype=np.float32)

    if len(vetor) != 38:
        raise ValueError(
            f"Quantidade incorreta de features: {len(vetor)}. "
            "O modelo atual espera 38."
        )

    return vetor


def gravar_audio(
    caminho_saida: Path,
    duracao: int = DURACAO_PADRAO
):
    """
    Grava um áudio pelo microfone e salva em WAV.
    """

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(" GRAVAÇÃO DE ÁUDIO")
    print("=" * 70)
    print(f"Duração: {duracao} segundos")
    print("Prepare-se...")
    input("Pressione ENTER para iniciar a gravação.")

    print()
    print("Gravando... Fale agora!")

    audio = sd.rec(
        int(duracao * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CANAIS,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        caminho_saida,
        audio,
        SAMPLE_RATE
    )

    print("Gravação concluída.")
    print(f"Arquivo salvo em: {caminho_saida}")

    return caminho_saida


def gerar_nome_gravacao(prefixo: str = "audio") -> str:
    """
    Gera um nome único baseado na data e horário.
    """

    agora = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    return f"{prefixo}_{agora}.wav"


def prever_pessoa(caminho_audio: Path):
    """
    Realiza a previsão da pessoa e retorna:
    - nome previsto
    - confiança
    - probabilidades
    """

    modelo, classes = carregar_modelo(
        MODELO_PESSOAS_PATH
    )

    features = extrair_features(caminho_audio)
    entrada = features.reshape(1, -1)

    probabilidades = modelo.predict_proba(entrada)[0]

    indice_maior = int(np.argmax(probabilidades))

    pessoa_prevista = str(classes[indice_maior])
    confianca = float(probabilidades[indice_maior])

    probabilidades_ordenadas = sorted(
        zip(classes, probabilidades),
        key=lambda item: item[1],
        reverse=True
    )

    return (
        pessoa_prevista,
        confianca,
        probabilidades_ordenadas
    )


def prever_emocao(caminho_audio: Path):
    """
    Realiza a previsão da emoção.
    """

    modelo, classes = carregar_modelo(
        MODELO_EMOCOES_PATH
    )

    features = extrair_features(caminho_audio)
    entrada = features.reshape(1, -1)

    probabilidades = modelo.predict_proba(entrada)[0]

    indice_maior = int(np.argmax(probabilidades))

    emocao_prevista = str(classes[indice_maior])
    confianca = float(probabilidades[indice_maior])

    return emocao_prevista, confianca


def exibir_probabilidades(probabilidades):
    """
    Exibe as probabilidades de todas as classes.
    """

    print()
    print("Probabilidades por pessoa:")

    for classe, probabilidade in probabilidades:
        print(
            f"   {str(classe):<15} "
            f"{probabilidade * 100:6.2f}%"
        )


# ============================================================
# MODO 1 - IDENTIFICAR PESSOA
# ============================================================

def modo_identificar():
    """
    Grava um novo áudio e tenta identificar a pessoa.

    Este modo NÃO adiciona o áudio ao dataset.
    """

    print()
    print("=" * 70)
    print(" MODO 1 - IDENTIFICAR PESSOA")
    print("=" * 70)

    print()
    print("Pessoas conhecidas pelo modelo:")

    _, classes = carregar_modelo(
        MODELO_PESSOAS_PATH
    )

    for classe in classes:
        print(f"   - {classe}")

    print()
    print(
        "O áudio será utilizado apenas para teste."
    )
    print(
        "Ele NÃO será adicionado ao dataset."
    )

    caminho_audio = (
        GRAVACOES_DEMONSTRACAO_DIR
        / gerar_nome_gravacao("identificacao")
    )

    gravar_audio(caminho_audio)

    print()
    print("Analisando áudio...")

    pessoa, confianca_pessoa, probabilidades = prever_pessoa(
        caminho_audio
    )

    emocao, confianca_emocao = prever_emocao(
        caminho_audio
    )

    print()
    print("=" * 70)
    print(" RESULTADO DA IDENTIFICAÇÃO")
    print("=" * 70)

    if confianca_pessoa < CONFIANCA_MINIMA_DESCONHECIDO:
        print("Pessoa identificada: DESCONHECIDO")
        print(
            f"Previsão mais próxima: {pessoa}"
        )
        print(
            f"Confiança da previsão: "
            f"{confianca_pessoa * 100:.2f}%"
        )
        print(
            "Motivo: nenhuma classe atingiu "
            f"{CONFIANCA_MINIMA_DESCONHECIDO * 100:.0f}% "
            "de confiança."
        )
    else:
        print(f"Pessoa identificada: {pessoa}")
        print(
            f"Confiança da pessoa: "
            f"{confianca_pessoa * 100:.2f}%"
        )

    print()
    print(f"Emoção detectada: {emocao}")
    print(
        f"Confiança da emoção: "
        f"{confianca_emocao * 100:.2f}%"
    )

    exibir_probabilidades(probabilidades)

    print()
    print(f"Áudio de teste preservado em:")
    print(f"{caminho_audio}")


# ============================================================
# MODO 2 - CADASTRAR NOVA PESSOA
# ============================================================

def modo_cadastrar():
    """
    Cadastra uma nova pessoa gravando vários áudios.

    Depois executa novamente o treinamento do modelo de pessoas.
    """

    print()
    print("=" * 70)
    print(" MODO 2 - CADASTRAR NOVA PESSOA")
    print("=" * 70)

    nome_digitado = input(
        "Digite o nome da nova pessoa: "
    )

    nome_pessoa = limpar_nome_pessoa(
        nome_digitado
    )

    if not nome_pessoa:
        print()
        print("Nome inválido.")
        return

    pasta_pessoa = (
        DATASET_PESSOAS_DIR / nome_pessoa
    )

    if pasta_pessoa.exists():
        arquivos_existentes = list(
            pasta_pessoa.glob("*.wav")
        )

        print()
        print(
            f"A pessoa '{nome_pessoa}' "
            "já possui uma pasta."
        )
        print(
            f"Áudios existentes: "
            f"{len(arquivos_existentes)}"
        )

        resposta = input(
            "Deseja adicionar mais áudios? (s/n): "
        ).strip().lower()

        if resposta != "s":
            print("Cadastro cancelado.")
            return

    else:
        pasta_pessoa.mkdir(
            parents=True,
            exist_ok=True
        )

    quantidade_texto = input(
        "Quantos áudios deseja gravar? "
        "(recomendado: 20 a 30): "
    )

    try:
        quantidade = int(quantidade_texto)
    except ValueError:
        print("Quantidade inválida.")
        return

    if quantidade <= 0:
        print("A quantidade deve ser maior que zero.")
        return

    duracao_texto = input(
        "Duração de cada áudio em segundos "
        "(pressione ENTER para usar 4): "
    ).strip()

    if duracao_texto:
        try:
            duracao = int(duracao_texto)
        except ValueError:
            print("Duração inválida.")
            return
    else:
        duracao = DURACAO_PADRAO

    if duracao <= 0:
        print("A duração deve ser maior que zero.")
        return

    print()
    print("=" * 70)
    print(" INÍCIO DO CADASTRO")
    print("=" * 70)
    print(f"Pessoa: {nome_pessoa}")
    print(f"Quantidade de áudios: {quantidade}")
    print(f"Duração de cada áudio: {duracao} segundos")
    print(f"Pasta: {pasta_pessoa}")

    resposta = input(
        "Deseja iniciar? (s/n): "
    ).strip().lower()

    if resposta != "s":
        print("Cadastro cancelado.")
        return

    arquivos_gravados = []

    for numero in range(1, quantidade + 1):
        nome_arquivo = (
            f"{nome_pessoa}_{numero:03d}.wav"
        )

        caminho_audio = (
            pasta_pessoa / nome_arquivo
        )

        print()
        print(
            f"[{numero}/{quantidade}] "
            f"Prepare-se para falar."
        )

        gravar_audio(
            caminho_audio,
            duracao
        )

        arquivos_gravados.append(
            caminho_audio
        )

    print()
    print("=" * 70)
    print(" CADASTRO DE ÁUDIOS CONCLUÍDO")
    print("=" * 70)
    print(
        f"Áudios gravados nesta operação: "
        f"{len(arquivos_gravados)}"
    )
    print(f"Pasta: {pasta_pessoa}")

    resposta = input(
        "Deseja treinar novamente o modelo de pessoas? (s/n): "
    ).strip().lower()

    if resposta != "s":
        print()
        print(
            "Áudios salvos, mas o modelo ainda "
            "não foi atualizado."
        )
        print(
            "Execute posteriormente:"
        )
        print(
            "python .\\src\\03_treinar_pessoas.py"
        )
        return

    treinar_modelo_pessoas()


def treinar_modelo_pessoas():
    """
    Executa o script atual de treinamento de pessoas.
    """

    print()
    print("=" * 70)
    print(" TREINANDO NOVAMENTE O MODELO DE PESSOAS")
    print("=" * 70)

    if not TREINAR_PESSOAS_SCRIPT.exists():
        print()
        print(
            "Script de treinamento não encontrado:"
        )
        print(TREINAR_PESSOAS_SCRIPT)
        return

    resultado = subprocess.run(
        [
            sys.executable,
            str(TREINAR_PESSOAS_SCRIPT)
        ],
        cwd=BASE_DIR
    )

    print()

    if resultado.returncode == 0:
        print("=" * 70)
        print(" MODELO ATUALIZADO COM SUCESSO")
        print("=" * 70)
        print(
            f"A pessoa foi adicionada ao dataset:"
        )
        print(
            "O arquivo modelo_pessoas.pkl "
            "foi recriado."
        )
    else:
        print("=" * 70)
        print(" ERRO AO TREINAR O MODELO")
        print("=" * 70)
        print(
            "Execute manualmente para visualizar "
            "mais detalhes:"
        )
        print(
            "python .\\src\\03_treinar_pessoas.py"
        )


# ============================================================
# MENU PRINCIPAL
# ============================================================

def exibir_menu():
    print()
    print("=" * 70)
    print(" SALA DE REUNIÃO INTELIGENTE")
    print(" GRAVAÇÃO, IDENTIFICAÇÃO E CADASTRO")
    print("=" * 70)
    print()
    print("1 - Gravar áudio e identificar pessoa")
    print("2 - Cadastrar nova pessoa")
    print("3 - Sair")
    print()


def main():
    while True:
        exibir_menu()

        opcao = input(
            "Escolha uma opção: "
        ).strip()

        if opcao == "1":
            try:
                modo_identificar()
            except Exception as erro:
                print()
                print("=" * 70)
                print(" ERRO DURANTE A IDENTIFICAÇÃO")
                print("=" * 70)
                print(type(erro).__name__)
                print(erro)

        elif opcao == "2":
            try:
                modo_cadastrar()
            except Exception as erro:
                print()
                print("=" * 70)
                print(" ERRO DURANTE O CADASTRO")
                print("=" * 70)
                print(type(erro).__name__)
                print(erro)

        elif opcao == "3":
            print()
            print("Encerrando sistema...")
            break

        else:
            print()
            print("Opção inválida.")


if __name__ == "__main__":
    main()