from pathlib import Path

import librosa
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_pessoas"

PESSOAS = [
    "caio",
    "kaique",
    "lorenzo",
    "lucascas",
]


def analisar_pessoa(pessoa):
    pasta = DATASET_DIR / pessoa
    arquivos = sorted(pasta.glob("*.wav"))

    print()
    print("=" * 70)
    print(f" PESSOA: {pessoa.upper()}")
    print("=" * 70)

    if not arquivos:
        print("Nenhum arquivo encontrado.")
        return

    duracoes = []
    rms_values = []
    zcr_values = []
    centroid_values = []
    mfcc_values = []

    erros = 0

    for arquivo in arquivos:
        try:
            audio, sr = librosa.load(
                arquivo,
                sr=None,
                mono=True
            )

            duracao = len(audio) / sr

            rms = float(np.mean(librosa.feature.rms(y=audio)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
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

            duracoes.append(duracao)
            rms_values.append(rms)
            zcr_values.append(zcr)
            centroid_values.append(centroid)
            mfcc_values.append(mfcc_mean)

        except Exception as e:
            erros += 1
            print(f"ERRO: {arquivo.name}")
            print(f"      {e}")

    if not duracoes:
        print("Nenhum áudio pôde ser analisado.")
        return

    print(f"Arquivos encontrados : {len(arquivos)}")
    print(f"Arquivos analisados  : {len(duracoes)}")
    print(f"Erros                : {erros}")

    print()
    print("DURAÇÃO")
    print("-" * 70)
    print(f"Mínima               : {min(duracoes):.2f}s")
    print(f"Máxima               : {max(duracoes):.2f}s")
    print(f"Média                : {np.mean(duracoes):.2f}s")
    print(f"Desvio padrão        : {np.std(duracoes):.2f}s")

    print()
    print("RMS / ENERGIA")
    print("-" * 70)
    print(f"Mínimo               : {min(rms_values):.6f}")
    print(f"Máximo               : {max(rms_values):.6f}")
    print(f"Média                : {np.mean(rms_values):.6f}")
    print(f"Desvio padrão        : {np.std(rms_values):.6f}")

    print()
    print("ZERO CROSSING RATE")
    print("-" * 70)
    print(f"Mínimo               : {min(zcr_values):.6f}")
    print(f"Máximo               : {max(zcr_values):.6f}")
    print(f"Média                : {np.mean(zcr_values):.6f}")
    print(f"Desvio padrão        : {np.std(zcr_values):.6f}")

    print()
    print("SPECTRAL CENTROID")
    print("-" * 70)
    print(f"Mínimo               : {min(centroid_values):.2f}")
    print(f"Máximo               : {max(centroid_values):.2f}")
    print(f"Média                : {np.mean(centroid_values):.2f}")
    print(f"Desvio padrão        : {np.std(centroid_values):.2f}")

    print()
    print("MFCC")
    print("-" * 70)
    print(f"Mínimo               : {min(mfcc_values):.4f}")
    print(f"Máximo               : {max(mfcc_values):.4f}")
    print(f"Média                : {np.mean(mfcc_values):.4f}")
    print(f"Desvio padrão        : {np.std(mfcc_values):.4f}")


def main():
    print("=" * 70)
    print(" DIAGNÓSTICO DO DATASET DE PESSOAS")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_DIR}")

    for pessoa in PESSOAS:
        analisar_pessoa(pessoa)

    print()
    print("=" * 70)
    print(" DIAGNÓSTICO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()