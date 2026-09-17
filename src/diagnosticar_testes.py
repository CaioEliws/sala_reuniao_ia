from pathlib import Path

import librosa
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

TESTES_DIR = BASE_DIR / "reunioes" / "reuniao_01" / "testes"


def analisar_audio(arquivo):
    audio, sr = librosa.load(
        arquivo,
        sr=None,
        mono=True
    )

    duracao = len(audio) / sr

    rms = float(
        np.mean(
            librosa.feature.rms(y=audio)
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(audio)
        )
    )

    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sr
            )
        )
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = float(np.mean(mfcc))

    return {
        "duracao": duracao,
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "mfcc": mfcc_mean,
        "sample_rate": sr,
    }


def main():

    print("=" * 70)
    print(" DIAGNÓSTICO DOS ÁUDIOS DE TESTE")
    print("=" * 70)

    pessoas = [
        "caio",
        "kaique",
        "lorenzo",
        "lucascas",
    ]

    for pessoa in pessoas:

        pasta = TESTES_DIR / pessoa

        arquivos = sorted(
            pasta.glob("*.wav")
        )

        if not arquivos:
            continue

        print()
        print("=" * 70)
        print(f" TESTES: {pessoa.upper()}")
        print("=" * 70)

        for arquivo in arquivos:

            try:

                dados = analisar_audio(arquivo)

                print()
                print(f"Arquivo: {arquivo.name}")
                print(f"  Duração       : {dados['duracao']:.2f}s")
                print(f"  Sample rate   : {dados['sample_rate']} Hz")
                print(f"  RMS           : {dados['rms']:.6f}")
                print(f"  ZCR           : {dados['zcr']:.6f}")
                print(f"  Centroid      : {dados['centroid']:.2f}")
                print(f"  MFCC médio    : {dados['mfcc']:.4f}")

            except Exception as e:

                print()
                print(f"ERRO: {arquivo.name}")
                print(f"      {e}")

    print()
    print("=" * 70)
    print(" DIAGNÓSTICO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()