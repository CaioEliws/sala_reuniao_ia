from pathlib import Path

import joblib
import librosa
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELO_PESSOAS = BASE_DIR / "modelos" / "modelo_pessoas.pkl"
PASTA_TESTES = BASE_DIR / "reunioes" / "reuniao_01" / "testes"


# ============================================================
# EXTRAÇÃO DAS 38 FEATURES
# ============================================================

def extrair_features(caminho_audio):

    y, sr = librosa.load(
        caminho_audio,
        sr=16000,
        mono=True
    )

    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ========================================================
    # RMS
    # ========================================================

    rms = librosa.feature.rms(y=y)[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # ========================================================
    # ZCR
    # ========================================================

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # ========================================================
    # SPECTRAL CENTROID
    # ========================================================

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # ========================================================
    # SPECTRAL BANDWIDTH
    # ========================================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # ========================================================
    # SPECTRAL ROLLOFF
    # ========================================================

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # ========================================================
    # PITCH
    # ========================================================

    try:

        pitch = librosa.yin(
            y,
            fmin=50,
            fmax=500,
            sr=sr
        )

        pitch = pitch[np.isfinite(pitch)]

        if len(pitch) > 0:

            pitch_mean = np.mean(pitch)
            pitch_std = np.std(pitch)

        else:

            pitch_mean = 0.0
            pitch_std = 0.0

    except Exception:

        pitch_mean = 0.0
        pitch_std = 0.0

    # ========================================================
    # MONTAR AS 38 FEATURES
    # ========================================================

    features = np.concatenate([
        mfcc_mean,          # 13
        mfcc_std,           # 13

        [
            rms_mean,       # 1
            rms_std,        # 1

            zcr_mean,       # 1
            zcr_std,        # 1

            centroid_mean,  # 1
            centroid_std,   # 1

            bandwidth_mean, # 1
            bandwidth_std,  # 1

            rolloff_mean,   # 1
            rolloff_std,    # 1

            pitch_mean,     # 1
            pitch_std       # 1
        ]
    ])

    # ========================================================
    # VERIFICAÇÃO
    # ========================================================

    if len(features) != 38:

        raise ValueError(
            f"Quantidade incorreta de features: "
            f"{len(features)}. Esperado: 38."
        )

    return features.reshape(1, -1)


# ============================================================
# CARREGAR MODELO
# ============================================================

print("=" * 70)
print(" DIAGNÓSTICO DE PROBABILIDADES DO MODELO")
print("=" * 70)

print()

if not MODELO_PESSOAS.exists():

    print("ERRO: modelo não encontrado:")
    print(MODELO_PESSOAS)

    raise SystemExit(1)


modelo_carregado = joblib.load(MODELO_PESSOAS)


# ============================================================
# COMPATIBILIDADE COM MODELO SALVO COMO DICT
# ============================================================

if isinstance(modelo_carregado, dict):

    modelo = (
        modelo_carregado.get("modelo")
        or modelo_carregado.get("model")
        or modelo_carregado.get("classifier")
    )

    if modelo is None:

        print(
            "ERRO: não foi possível encontrar "
            "o classificador dentro do arquivo."
        )

        print()

        print("Chaves encontradas:")

        for chave in modelo_carregado.keys():
            print(f"    - {chave}")

        raise SystemExit(1)

else:

    modelo = modelo_carregado


# ============================================================
# CLASSES
# ============================================================

classes = modelo.classes_


print("Pessoas conhecidas pelo modelo:")

for classe in classes:
    print(f"    - {classe}")

print()

print(
    f"Modelo espera: "
    f"{modelo.n_features_in_} features"
)

print()


# ============================================================
# LOCALIZAR ÁUDIOS
# ============================================================

audios = sorted(
    PASTA_TESTES.rglob("*.wav")
)


print(f"Áudios encontrados: {len(audios)}")

print()


# ============================================================
# TESTAR CADA ÁUDIO
# ============================================================

for indice, audio in enumerate(audios, start=1):

    esperado = audio.parent.name

    print("=" * 70)

    print(
        f"[{indice}/{len(audios)}] "
        f"{audio.name}"
    )

    print(
        f"Esperado: {esperado}"
    )

    print("=" * 70)

    try:

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = extrair_features(audio)

        print(
            f"Features geradas: "
            f"{features.shape[1]}"
        )

        # ----------------------------------------------------
        # PROBABILIDADES
        # ----------------------------------------------------

        probabilidades = modelo.predict_proba(
            features
        )[0]

        # ----------------------------------------------------
        # ORDENAR
        # ----------------------------------------------------

        resultados = sorted(
            zip(classes, probabilidades),
            key=lambda x: x[1],
            reverse=True
        )

        print()

        for pessoa, probabilidade in resultados:

            marcador = ""

            if pessoa == esperado:
                marcador = " <- ESPERADO"

            print(
                f"{str(pessoa):12} "
                f"{probabilidade * 100:6.2f}%"
                f"{marcador}"
            )

        # ----------------------------------------------------
        # PREDIÇÃO
        # ----------------------------------------------------

        pessoa_predita = resultados[0][0]

        confianca = resultados[0][1]

        print()

        if pessoa_predita == esperado:

            print(
                f"RESULTADO: OK -> "
                f"{pessoa_predita} "
                f"({confianca * 100:.2f}%)"
            )

        else:

            print(
                f"RESULTADO: ERRO -> "
                f"{pessoa_predita} "
                f"({confianca * 100:.2f}%)"
            )

        print()

    except Exception as e:

        print(
            "ERRO ao processar áudio:"
        )

        print(e)

        print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print(" DIAGNÓSTICO FINALIZADO")
print("=" * 70)