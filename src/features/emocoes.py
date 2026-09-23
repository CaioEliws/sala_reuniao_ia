from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16000


def calcular_estatisticas(valores):
    valores = np.asarray(
        valores,
        dtype=np.float32
    )

    valores = valores[
        np.isfinite(valores)
    ]

    if valores.size == 0:
        return [
            0.0,
            0.0,
            0.0,
            0.0
        ]

    return [
        float(np.mean(valores)),
        float(np.std(valores)),
        float(np.min(valores)),
        float(np.max(valores))
    ]


def extrair_features_emocao(
    caminho_audio: Path
) -> np.ndarray:

    audio, sr = librosa.load(
        str(caminho_audio),
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError(
            "O áudio está vazio."
        )

    maior_amplitude = np.max(
        np.abs(audio)
    )

    if maior_amplitude > 0:
        audio = audio / maior_amplitude

    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    for linha in mfcc:
        features.extend(
            calcular_estatisticas(linha)
        )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=512,
        hop_length=160
    )[0]

    features.extend(
        calcular_estatisticas(rms)
    )

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=512,
        hop_length=160
    )[0]

    features.extend(
        calcular_estatisticas(zcr)
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=160
    )[0]

    features.extend(
        calcular_estatisticas(centroid)
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=160
    )[0]

    features.extend(
        calcular_estatisticas(bandwidth)
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=160
    )[0]

    features.extend(
        calcular_estatisticas(rolloff)
    )

    try:
        pitch = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=1024,
            hop_length=160
        )

        pitch = pitch[
            np.isfinite(pitch)
            & (pitch > 60)
            & (pitch < 600)
        ]

        features.extend(
            calcular_estatisticas(pitch)
        )

    except Exception:
        features.extend(
            [
                0.0,
                0.0,
                0.0,
                0.0
            ]
        )

    intervalos_voz = librosa.effects.split(
        audio,
        top_db=30
    )

    total_amostras_faladas = sum(
        fim - inicio
        for inicio, fim in intervalos_voz
    )

    taxa_silencio = 1.0 - (
        total_amostras_faladas / len(audio)
    )

    features.append(
        float(taxa_silencio)
    )

    tamanho = len(rms)

    if tamanho >= 3:
        partes = np.array_split(
            rms,
            3
        )

        for parte in partes:
            features.append(
                float(np.mean(parte))
            )
    else:
        features.extend(
            [
                0.0,
                0.0,
                0.0
            ]
        )

    vetor = np.asarray(
        features,
        dtype=np.float32
    ).flatten()

    if not np.all(
        np.isfinite(vetor)
    ):
        raise ValueError(
            "Foram encontradas características inválidas."
        )

    if vetor.size != 80:
        raise ValueError(
            "O extrator de emoções gerou "
            f"{vetor.size} características. "
            "O esperado é 80."
        )

    return vetor