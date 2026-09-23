"""
comum.py

Código compartilhado pelos scripts do projeto.

Responsabilidades:
- Configuração geral
- Extração de características
- Carregamento de áudios
- Contagem dos datasets
- Treinamento dos Random Forests
- Avaliação
- Matriz de confusão
- Salvamento/carregamento dos modelos
- Classificação de novos áudios
- Confiança e ranking Top-3
- Detecção de pessoa desconhecida
- Detecção de emoção incerta
"""

from pathlib import Path
import json
import warnings

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PASTA_DATASET_PESSOAS = BASE_DIR / "dataset_pessoas"
PASTA_DATASET_EMOCOES = BASE_DIR / "dataset_emocoes"

PASTA_MODELOS = BASE_DIR / "modelos"

PASTA_REUNIAO = (
    BASE_DIR
    / "reunioes"
    / "reuniao_01"
)

PASTA_GRAFICOS = PASTA_REUNIAO / "graficos"

SR = 16000

DURACAO_CLIPE = 3.0

EMOCOES = [
    "alegre",
    "neutro",
    "triste",
    "irritado",
]


# ============================================================
# REGRAS DE CLASSIFICAÇÃO
# ============================================================

# Pessoa:
# precisa ter confiança mínima e diferença suficiente
# entre a primeira e a segunda opção.
LIMIAR_PESSOA = 0.45
MARGEM_PESSOA = 0.20

# Emoção:
# apenas a maior probabilidade é considerada.
LIMIAR_EMOCAO = 0.40

# Silêncio:
# abaixo desse RMS o áudio não deve ser classificado.
LIMIAR_SILENCIO = 0.004

ROTULO_DESCONHECIDO = "desconhecido"
ROTULO_INCERTO = "incerta"


# Quantidade mínima recomendada.
MINIMO_POR_PESSOA = 20
MINIMO_POR_EMOCAO = 25

# Para treinar pessoas:
# pelo menos duas pessoas.
MINIMO_PESSOAS = 2


# ============================================================
# PASTAS
# ============================================================

PASTA_MODELOS.mkdir(
    parents=True,
    exist_ok=True
)

PASTA_GRAFICOS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CAMINHOS DOS MODELOS
# ============================================================

def caminho_modelo(nome_modelo):
    return PASTA_MODELOS / nome_modelo


def caminho_metadados():
    return PASTA_MODELOS / "metadados.json"


# ============================================================
# EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================

def _estatisticas(vetor):
    """
    Retorna quatro estatísticas:
    média, desvio padrão, mínimo e máximo.
    """

    vetor = np.asarray(
        vetor,
        dtype=np.float32
    )

    vetor = vetor[
        np.isfinite(vetor)
    ]

    if len(vetor) == 0:
        return [
            0.0,
            0.0,
            0.0,
            0.0
        ]

    return [
        float(np.mean(vetor)),
        float(np.std(vetor)),
        float(np.min(vetor)),
        float(np.max(vetor)),
    ]


def extrair_features_base(caminho_audio):
    """
    Extrai 76 características.

    MFCC:
        13 coeficientes x 4 estatísticas = 52

    RMS:
        4

    ZCR:
        4

    Spectral Centroid:
        4

    Spectral Bandwidth:
        4

    Spectral Rolloff:
        4

    Pitch:
        4

    Total:
        52 + 24 = 76
    """

    audio, sr = librosa.load(
        str(caminho_audio),
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "Áudio vazio."
        )

    features = []

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    for coeficiente in mfcc:
        features.extend(
            _estatisticas(coeficiente)
        )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )[0]

    features.extend(
        _estatisticas(rms)
    )

    # --------------------------------------------------------
    # ZCR
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    features.extend(
        _estatisticas(zcr)
    )

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    features.extend(
        _estatisticas(centroid)
    )

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]

    features.extend(
        _estatisticas(bandwidth)
    )

    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )[0]

    features.extend(
        _estatisticas(rolloff)
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    try:

        pitch = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        features.extend(
            _estatisticas(pitch)
        )

    except Exception:

        features.extend([
            0.0,
            0.0,
            0.0,
            0.0
        ])

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.size != 76:
        raise ValueError(
            f"Quantidade inesperada de features: "
            f"{features.size}. Esperado: 76."
        )

    return features


def calcular_features_temporais(audio):
    """
    Retorna quatro características adicionais
    utilizadas pelo modelo de emoções:

    1. taxa de silêncio
    2. energia no início
    3. energia no meio
    4. energia no fim
    """

    if len(audio) == 0:
        return [
            1.0,
            0.0,
            0.0,
            0.0
        ]

    rms = librosa.feature.rms(
        y=audio
    )[0]

    if len(rms) == 0:
        return [
            1.0,
            0.0,
            0.0,
            0.0
        ]

    taxa_silencio = float(
        np.mean(
            rms < LIMIAR_SILENCIO
        )
    )

    tamanho = len(rms)

    terco = max(
        1,
        tamanho // 3
    )

    energia_inicio = float(
        np.mean(
            rms[:terco]
        )
    )

    energia_meio = float(
        np.mean(
            rms[terco:terco * 2]
        )
    )

    energia_fim = float(
        np.mean(
            rms[terco * 2:]
        )
    )

    return [
        taxa_silencio,
        energia_inicio,
        energia_meio,
        energia_fim
    ]


def extrair_features_pessoa(caminho_audio):
    """
    Features utilizadas pelo modelo de pessoas.

    Total: 76
    """

    return extrair_features_base(
        caminho_audio
    )


def extrair_features_emocao(caminho_audio):
    """
    Features utilizadas pelo modelo de emoções.

    Base:
        76

    Características temporais:
        4

    Total:
        80
    """

    audio, sr = librosa.load(
        str(caminho_audio),
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "Áudio vazio."
        )

    base = extrair_features_base(
        caminho_audio
    )

    temporais = calcular_features_temporais(
        audio
    )

    features = np.concatenate([
        base,
        np.asarray(
            temporais,
            dtype=np.float32
        )
    ])

    if features.size != 80:
        raise ValueError(
            f"Quantidade inesperada de features "
            f"para emoção: {features.size}. "
            f"Esperado: 80."
        )

    return features.astype(
        np.float32
    )


# ============================================================
# CARREGAMENTO DE WAV
# ============================================================

def carregar_wav(caminho):
    """
    Carrega áudio para classificação.
    """

    audio, sr = librosa.load(
        str(caminho),
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "Áudio vazio."
        )

    return audio


# ============================================================
# CONTAGEM DO DATASET
# ============================================================

def contar_wavs(pasta_dataset):
    """
    Retorna:
        {
            "caio": 30,
            "lorenzo": 30,
            ...
        }
    """

    pasta_dataset = Path(
        pasta_dataset
    )

    contagem = {}

    if not pasta_dataset.exists():
        return contagem

    for pasta in sorted(
        p
        for p in pasta_dataset.iterdir()
        if p.is_dir()
    ):

        contagem[pasta.name] = len(
            list(
                pasta.glob("*.wav")
            )
        )

    return contagem


def imprimir_contagem(
    titulo,
    contagem,
    minimo=None
):

    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)
    print()

    if not contagem:
        print("Nenhuma classe encontrada.")
        return

    for classe, quantidade in sorted(
        contagem.items()
    ):

        status = ""

        if (
            minimo is not None
            and quantidade < minimo
        ):
            status = (
                f"  [ABAIXO DO MINIMO: {minimo}]"
            )

        print(
            f"{classe:<15} "
            f"{quantidade:>4} áudios"
            f"{status}"
        )

    print()


# ============================================================
# CARREGAR DATASET
# ============================================================

def carregar_dataset(
    pasta_dataset,
    extrator,
    classes_permitidas=None
):

    pasta_dataset = Path(
        pasta_dataset
    )

    X = []
    y = []

    erros = 0

    if not pasta_dataset.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: "
            f"{pasta_dataset}"
        )

    pastas = sorted(
        p
        for p in pasta_dataset.iterdir()
        if p.is_dir()
    )

    for pasta in pastas:

        classe = pasta.name

        if (
            classes_permitidas is not None
            and classe not in classes_permitidas
        ):
            continue

        arquivos = sorted(
            pasta.glob("*.wav")
        )

        print(
            f"Classe: {classe:<12} "
            f"áudios: {len(arquivos)}"
        )

        for arquivo in arquivos:

            try:

                features = extrator(
                    arquivo
                )

                features = np.asarray(
                    features,
                    dtype=np.float32
                ).flatten()

                X.append(features)
                y.append(classe)

            except Exception as erro:

                erros += 1

                print(
                    f"   ERRO: {arquivo.name}"
                )

                print(
                    f"      {erro}"
                )

    if not X:
        raise RuntimeError(
            "Nenhum áudio válido foi processado."
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(y)

    return X, y, erros


# ============================================================
# TREINAMENTO
# ============================================================

def treinar(
    pasta_dataset,
    nome_modelo,
    chave_meta,
    titulo,
    minimo_por_classe,
    minimo_classes,
    forcar=False,
    n_arvores=300,
    extrator=None
):

    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)

    print()
    print(
        f"Dataset: {pasta_dataset}"
    )

    contagem = contar_wavs(
        pasta_dataset
    )

    imprimir_contagem(
        "CONTAGEM DO DATASET",
        contagem,
        minimo=minimo_por_classe
    )

    classes_validas = [
        classe
        for classe, quantidade
        in contagem.items()
        if quantidade >= minimo_por_classe
    ]

    if len(classes_validas) < minimo_classes:

        mensagem = (
            f"São necessárias pelo menos "
            f"{minimo_classes} classes com "
            f"{minimo_por_classe} áudios cada."
        )

        if not forcar:
            raise RuntimeError(
                mensagem
                + "\nUse --forcar para treinar mesmo assim."
            )

        print(
            "AVISO: "
            + mensagem
        )

        classes_validas = [
            classe
            for classe, quantidade
            in contagem.items()
            if quantidade > 0
        ]

    if extrator is None:
        raise ValueError(
            "Extrator de características não informado."
        )

    print()
    print("=" * 70)
    print("PROCESSANDO ÁUDIOS")
    print("=" * 70)
    print()

    X, y, erros = carregar_dataset(
        pasta_dataset,
        extrator,
        classes_permitidas=classes_validas
    )

    classes = sorted(
        np.unique(y)
    )

    if len(classes) < 2:
        raise RuntimeError(
            "O modelo precisa de pelo menos "
            "duas classes."
        )

    print()
    print(
        f"Áudios processados: {len(X)}"
    )

    print(
        f"Áudios com erro: {erros}"
    )

    print(
        f"Quantidade de features: {X.shape[1]}"
    )

    print(
        f"Classes: {list(classes)}"
    )

    print()
    print("=" * 70)
    print("DIVIDINDO DATASET")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(
        f"Treinamento: {len(X_train)} áudios"
    )

    print(
        f"Teste: {len(X_test)} áudios"
    )

    print()
    print("=" * 70)
    print("TREINANDO RANDOM FOREST")
    print("=" * 70)
    print()

    modelo = RandomForestClassifier(
        n_estimators=n_arvores,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    print(
        "Modelo treinado com sucesso."
    )

    # --------------------------------------------------------
    # AVALIAÇÃO
    # --------------------------------------------------------

    y_pred = modelo.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print()

    print(
        f"Acurácia: {accuracy:.2%}"
    )

    print()
    print("Relatório de classificação:")
    print()

    print(
        classification_report(
            y_test,
            y_pred,
            labels=classes,
            zero_division=0
        )
    )

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    print("Matriz de confusão:")
    print()
    print(matriz)
    print()

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

    if chave_meta == "pessoas":

        nome_grafico = (
            "matriz_confusao_pessoas.png"
        )

        titulo_grafico = (
            "Matriz de Confusão - "
            "Identificação de Pessoas"
        )

        xlabel = "Pessoa prevista"
        ylabel = "Pessoa real"

    else:

        nome_grafico = (
            "matriz_confusao_emocoes.png"
        )

        titulo_grafico = (
            "Matriz de Confusão - "
            "Reconhecimento de Emoções"
        )

        xlabel = "Emoção prevista"
        ylabel = "Emoção real"

    caminho_grafico = (
        PASTA_GRAFICOS
        / nome_grafico
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        titulo_grafico
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    plt.tight_layout()

    plt.savefig(
        caminho_grafico,
        dpi=150
    )

    plt.close()

    print(
        f"Matriz salva em: {caminho_grafico}"
    )

    # --------------------------------------------------------
    # SALVAR MODELO
    # --------------------------------------------------------

    pacote = {
        "modelo": modelo,
        "classes": list(modelo.classes_),
        "sample_rate": SR,
        "quantidade_features": int(
            X.shape[1]
        ),
        "tipo": chave_meta
    }

    caminho_modelo_arquivo = (
        PASTA_MODELOS
        / nome_modelo
    )

    joblib.dump(
        pacote,
        caminho_modelo_arquivo
    )

    print(
        f"Modelo salvo em: "
        f"{caminho_modelo_arquivo}"
    )

    # --------------------------------------------------------
    # METADADOS
    # --------------------------------------------------------

    caminho_meta = caminho_metadados()

    if caminho_meta.exists():

        try:

            with open(
                caminho_meta,
                "r",
                encoding="utf-8"
            ) as arquivo:

                metadados = json.load(
                    arquivo
                )

        except Exception:

            metadados = {}

    else:

        metadados = {}

    metadados[chave_meta] = {
        "modelo": "RandomForestClassifier",
        "classes": list(modelo.classes_),
        "sample_rate": SR,
        "quantidade_amostras": int(len(X)),
        "quantidade_treinamento": int(
            len(X_train)
        ),
        "quantidade_teste": int(
            len(X_test)
        ),
        "quantidade_features": int(
            X.shape[1]
        ),
        "acuracia": float(
            accuracy
        ),
        "arvores": int(
            n_arvores
        )
    }

    with open(
        caminho_meta,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Metadados salvos em: "
        f"{caminho_meta}"
    )

    return modelo


# ============================================================
# CARREGAMENTO DOS MODELOS
# ============================================================

def carregar_modelos():

    caminho_pessoas = (
        PASTA_MODELOS
        / "modelo_pessoas.pkl"
    )

    caminho_emocoes = (
        PASTA_MODELOS
        / "modelo_emocoes.pkl"
    )

    if not caminho_pessoas.exists():

        raise FileNotFoundError(
            f"Modelo de pessoas não encontrado:\n"
            f"{caminho_pessoas}"
        )

    if not caminho_emocoes.exists():

        raise FileNotFoundError(
            f"Modelo de emoções não encontrado:\n"
            f"{caminho_emocoes}"
        )

    pacote_pessoas = joblib.load(
        caminho_pessoas
    )

    pacote_emocoes = joblib.load(
        caminho_emocoes
    )

    with open(
        caminho_metadados(),
        "r",
        encoding="utf-8"
    ) as arquivo:

        metadados = json.load(
            arquivo
        )

    modelo_pessoas = pacote_pessoas["modelo"]
    modelo_emocoes = pacote_emocoes["modelo"]

    return (
        modelo_pessoas,
        modelo_emocoes,
        metadados
    )


# ============================================================
# RANKING
# ============================================================

def ranking_probabilidades(
    modelo,
    features
):

    probabilidades = (
        modelo.predict_proba(
            features.reshape(1, -1)
        )[0]
    )

    classes = modelo.classes_

    ranking = sorted(
        zip(
            classes,
            probabilidades
        ),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (
            classe,
            float(probabilidade)
        )
        for classe, probabilidade
        in ranking
    ]


def formatar_ranking(
    ranking,
    quantidade=3
):

    itens = []

    for classe, probabilidade in ranking[
        :quantidade
    ]:

        itens.append(
            f"{classe} {probabilidade:.0%}"
        )

    return " | ".join(
        itens
    )


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def classificar_pessoa(
    features
):

    modelo, _, _ = carregar_modelos()

    ranking = ranking_probabilidades(
        modelo,
        features
    )

    primeira_classe, primeira_prob = (
        ranking[0]
    )

    if len(ranking) > 1:

        segunda_prob = ranking[1][1]

    else:

        segunda_prob = 0.0

    margem = (
        primeira_prob
        - segunda_prob
    )

    if (
        primeira_prob < LIMIAR_PESSOA
        or margem < MARGEM_PESSOA
    ):

        pessoa = ROTULO_DESCONHECIDO

    else:

        pessoa = primeira_classe

    return {
        "pessoa": pessoa,
        "conf_pessoa": primeira_prob,
        "margem_pessoa": margem,
        "ranking_pessoa": ranking
    }


def classificar_emocao(
    features
):

    _, modelo, _ = carregar_modelos()

    ranking = ranking_probabilidades(
        modelo,
        features
    )

    emocao, confianca = ranking[0]

    if confianca < LIMIAR_EMOCAO:

        emocao_final = ROTULO_INCERTO

    else:

        emocao_final = emocao

    return {
        "emocao": emocao_final,
        "conf_emocao": confianca,
        "ranking_emocao": ranking
    }


def classificar_audio(
    audio,
    modelo_pessoas,
    modelo_emocoes
):

    if len(audio) == 0:

        raise ValueError(
            "Áudio vazio."
        )

    # --------------------------------------------------------
    # Silêncio
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )[0]

    rms_medio = float(
        np.mean(rms)
    )

    if rms_medio < LIMIAR_SILENCIO:

        return {
            "silencio": True,
            "pessoa": ROTULO_DESCONHECIDO,
            "emocao": ROTULO_INCERTO,
            "conf_pessoa": 0.0,
            "conf_emocao": 0.0,
            "ranking_pessoa": [],
            "ranking_emocao": []
        }

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    # Pessoas -> 76 features
    features_pessoa = extrair_features_base_do_audio(
        audio
    )

    # Emoções -> 80 features
    features_emocao = extrair_features_emocao_do_audio(
        audio
    )

    # --------------------------------------------------------
    # Pessoa
    # --------------------------------------------------------

    ranking_pessoa = ranking_probabilidades(
        modelo_pessoas,
        features_pessoa
    )

    pessoa_top, conf_pessoa = (
        ranking_pessoa[0]
    )

    segunda_conf = (
        ranking_pessoa[1][1]
        if len(ranking_pessoa) > 1
        else 0.0
    )

    margem = (
        conf_pessoa
        - segunda_conf
    )

    if (
        conf_pessoa < LIMIAR_PESSOA
        or margem < MARGEM_PESSOA
    ):

        pessoa = ROTULO_DESCONHECIDO

    else:

        pessoa = pessoa_top

    # --------------------------------------------------------
    # Emoção
    # --------------------------------------------------------

    ranking_emocao = ranking_probabilidades(
        modelo_emocoes,
        features_emocao
    )

    emocao_top, conf_emocao = (
        ranking_emocao[0]
    )

    if conf_emocao < LIMIAR_EMOCAO:

        emocao = ROTULO_INCERTO

    else:

        emocao = emocao_top

    return {
        "silencio": False,

        "pessoa": pessoa,
        "conf_pessoa": conf_pessoa,
        "margem_pessoa": margem,
        "ranking_pessoa": ranking_pessoa,

        "emocao": emocao,
        "conf_emocao": conf_emocao,
        "ranking_emocao": ranking_emocao
    }


# ============================================================
# EXTRAÇÃO DIRETA DE FEATURES A PARTIR DO AUDIO
# ============================================================

def extrair_features_base_do_audio(audio):

    """
    Versão da extração de 76 features recebendo
    diretamente o vetor de áudio.
    """

    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=13
    )

    for coeficiente in mfcc:

        features.extend(
            _estatisticas(coeficiente)
        )

    rms = librosa.feature.rms(
        y=audio
    )[0]

    features.extend(
        _estatisticas(rms)
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    features.extend(
        _estatisticas(zcr)
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SR
    )[0]

    features.extend(
        _estatisticas(centroid)
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SR
    )[0]

    features.extend(
        _estatisticas(bandwidth)
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SR
    )[0]

    features.extend(
        _estatisticas(rolloff)
    )

    try:

        pitch = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SR
        )

        features.extend(
            _estatisticas(pitch)
        )

    except Exception:

        features.extend([
            0.0,
            0.0,
            0.0,
            0.0
        ])

    return np.asarray(
        features,
        dtype=np.float32
    )


def extrair_features_emocao_do_audio(audio):

    base = extrair_features_base_do_audio(
        audio
    )

    temporais = calcular_features_temporais(
        audio
    )

    return np.concatenate([
        base,
        np.asarray(
            temporais,
            dtype=np.float32
        )
    ]).astype(
        np.float32
    )


# ============================================================
# GRAVAÇÃO / DISPOSITIVOS
# ============================================================

def listar_dispositivos():
    try:

        import sounddevice as sd

        print(
            sd.query_devices()
        )

    except ImportError:

        print(
            "sounddevice não instalado."
        )


def contagem_regressiva(segundos=3):

    import time

    for numero in range(
        segundos,
        0,
        -1
    ):

        print(
            f"Gravando em {numero}..."
        )

        time.sleep(1)


def gravar(
    duracao,
    dispositivo=None
):

    import sounddevice as sd

    audio = sd.rec(
        int(
            duracao * SR
        ),
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=dispositivo
    )

    sd.wait()

    return audio.flatten()