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
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset_pessoas"
MODELOS_DIR = BASE_DIR / "modelos"
GRAFICOS_DIR = BASE_DIR / "reunioes" / "reuniao_01" / "graficos"

MODELO_PATH = MODELOS_DIR / "modelo_pessoas.pkl"
METADADOS_PATH = MODELOS_DIR / "metadados_pessoas.json"
MATRIZ_PATH = GRAFICOS_DIR / "matriz_confusao_pessoas.png"

SR = 16000


# ============================================================
# PREPARAR PASTAS
# ============================================================

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================

def extrair_features(caminho_audio):

    audio, sr = librosa.load(
        str(caminho_audio),
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Áudio vazio.")

    features = []

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )

    # --------------------------------------------------------
    # Energia RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    features.append(
        float(np.mean(rms))
    )

    features.append(
        float(np.std(rms))
    )

    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    features.append(
        float(np.mean(zcr))
    )

    features.append(
        float(np.std(zcr))
    )

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.append(
        float(np.mean(centroid))
    )

    features.append(
        float(np.std(centroid))
    )

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    features.append(
        float(np.mean(bandwidth))
    )

    features.append(
        float(np.std(bandwidth))
    )

    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    features.append(
        float(np.mean(rolloff))
    )

    features.append(
        float(np.std(rolloff))
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

        pitch = pitch[np.isfinite(pitch)]

        if len(pitch) > 0:

            features.append(
                float(np.mean(pitch))
            )

            features.append(
                float(np.std(pitch))
            )

        else:

            features.append(0.0)
            features.append(0.0)

    except Exception:

        features.append(0.0)
        features.append(0.0)

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# CARREGAR DATASET
# ============================================================

def carregar_dataset():

    X = []
    y = []

    total_processados = 0
    total_erros = 0

    print("=" * 70)
    print("TREINAMENTO DO MODELO DE IDENTIFICAÇÃO DE PESSOAS")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_DIR}")
    print()

    pastas_pessoas = sorted(
        [
            pasta
            for pasta in DATASET_DIR.iterdir()
            if pasta.is_dir()
        ]
    )

    if len(pastas_pessoas) < 2:

        raise RuntimeError(
            "É necessário ter pelo menos 2 pessoas no dataset."
        )

    print(
        f"Pessoas encontradas: {len(pastas_pessoas)}"
    )

    print()

    for pasta_pessoa in pastas_pessoas:

        pessoa = pasta_pessoa.name

        arquivos = sorted(
            pasta_pessoa.glob("*.wav")
        )

        print(
            f"Pessoa: {pessoa}"
        )

        print(
            f"Arquivos encontrados: {len(arquivos)}"
        )

        if len(arquivos) == 0:

            print(
                "   AVISO: nenhuma gravação encontrada."
            )

            print()

            continue

        for arquivo in arquivos:

            try:

                features = extrair_features(
                    arquivo
                )

                X.append(features)

                y.append(pessoa)

                total_processados += 1

                print(
                    f"   OK: {arquivo.name}"
                )

            except Exception as erro:

                total_erros += 1

                print(
                    f"   ERRO: {arquivo.name}"
                )

                print(
                    f"      {erro}"
                )

        print()

    if len(X) == 0:

        raise RuntimeError(
            "Nenhum áudio foi processado."
        )

    X = np.array(X)
    y = np.array(y)

    print("=" * 70)
    print("DATASET CARREGADO")
    print("=" * 70)

    print(
        f"Áudios processados: {total_processados}"
    )

    print(
        f"Áudios com erro: {total_erros}"
    )

    print(
        f"Quantidade de características: {X.shape[1]}"
    )

    print()

    print("Distribuição:")

    for pessoa in sorted(
        np.unique(y)
    ):

        quantidade = np.sum(
            y == pessoa
        )

        print(
            f"   {pessoa}: {quantidade}"
        )

    print()

    return X, y


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_modelo(X, y):

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

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("=" * 70)
    print("TREINANDO RANDOM FOREST")
    print("=" * 70)

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    modelo.fit(
        X_train,
        y_train
    )

    print(
        "Modelo treinado com sucesso."
    )

    print()

    # --------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------

    y_pred = modelo.predict(
        X_test
    )

    # --------------------------------------------------------
    # ACURÁCIA
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print()

    print(
        f"Acurácia: {accuracy * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # RELATÓRIO
    # --------------------------------------------------------

    classes = sorted(
        np.unique(y)
    )

    print(
        "Relatório de classificação:"
    )

    print()

    print(
        classification_report(
            y_test,
            y_pred,
            labels=classes,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # MATRIZ DE CONFUSÃO
    # --------------------------------------------------------

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    print(
        "Matriz de confusão:"
    )

    print()

    print(
        matriz
    )

    print()

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

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

    plt.title(
        "Matriz de Confusão - Identificação de Pessoas"
    )

    plt.xlabel(
        "Pessoa prevista"
    )

    plt.ylabel(
        "Pessoa real"
    )

    plt.tight_layout()

    plt.savefig(
        MATRIZ_PATH,
        dpi=150
    )

    plt.close()

    print(
        "Matriz salva em:"
    )

    print(
        MATRIZ_PATH
    )

    print()

    # --------------------------------------------------------
    # SALVAR MODELO
    # --------------------------------------------------------

    pacote_modelo = {

        "modelo": modelo,

        "classes": classes,

        "sample_rate": SR,

        "quantidade_features": X.shape[1]

    }

    joblib.dump(
        pacote_modelo,
        MODELO_PATH
    )

    print(
        "Modelo salvo em:"
    )

    print(
        MODELO_PATH
    )

    print()

    # --------------------------------------------------------
    # SALVAR METADADOS
    # --------------------------------------------------------

    metadados = {

        "modelo": "RandomForestClassifier",

        "classes": classes,

        "sample_rate": SR,

        "quantidade_amostras": int(len(X)),

        "quantidade_treinamento": int(len(X_train)),

        "quantidade_teste": int(len(X_test)),

        "quantidade_features": int(X.shape[1]),

        "acuracia": float(accuracy),

        "features": [

            "MFCC_mean_13",
            "MFCC_std_13",

            "RMS_mean",
            "RMS_std",

            "ZCR_mean",
            "ZCR_std",

            "SpectralCentroid_mean",
            "SpectralCentroid_std",

            "SpectralBandwidth_mean",
            "SpectralBandwidth_std",

            "SpectralRolloff_mean",
            "SpectralRolloff_std",

            "Pitch_mean",
            "Pitch_std"

        ]

    }

    with open(
        METADADOS_PATH,
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
        "Metadados salvos em:"
    )

    print(
        METADADOS_PATH
    )

    print()

    return modelo


# ============================================================
# MAIN
# ============================================================

def main():

    X, y = carregar_dataset()

    treinar_modelo(
        X,
        y
    )

    print("=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)

    print()

    print("Arquivos gerados:")

    print(
        f"1. {MODELO_PATH}"
    )

    print(
        f"2. {METADADOS_PATH}"
    )

    print(
        f"3. {MATRIZ_PATH}"
    )

    print()


if __name__ == "__main__":
    main()