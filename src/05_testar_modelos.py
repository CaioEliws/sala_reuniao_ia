from pathlib import Path

import joblib
import librosa
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELOS_DIR = BASE_DIR / "modelos"

TESTES_DIR = (
    BASE_DIR
    / "reunioes"
    / "reuniao_01"
    / "testes"
)

MODELO_PESSOAS = MODELOS_DIR / "modelo_pessoas.pkl"
MODELO_EMOCOES = MODELOS_DIR / "modelo_emocoes.pkl"


# ============================================================
# EXTRAÇÃO DE FEATURES
# ============================================================

def extrair_features(caminho_audio):

    y, sr = librosa.load(
        caminho_audio,
        sr=16000,
        mono=True
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=y)[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # --------------------------------------------------------
    # ZCR
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)

    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    f0_valid = f0[np.isfinite(f0)]

    if len(f0_valid) > 0:

        pitch_mean = np.mean(f0_valid)
        pitch_std = np.std(f0_valid)

    else:

        pitch_mean = 0.0
        pitch_std = 0.0

    # --------------------------------------------------------
    # Vetor final
    # --------------------------------------------------------

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,

        [rms_mean],
        [rms_std],

        [zcr_mean],
        [zcr_std],

        [spectral_centroid_mean],
        [spectral_centroid_std],

        [spectral_bandwidth_mean],
        [spectral_bandwidth_std],

        [spectral_rolloff_mean],
        [spectral_rolloff_std],

        [pitch_mean],
        [pitch_std],
    ])

    return features


# ============================================================
# CARREGAR MODELOS
# ============================================================

def carregar_modelos():

    if not MODELO_PESSOAS.exists():

        print()
        print("ERRO: modelo de pessoas não encontrado.")
        print(f"Esperado: {MODELO_PESSOAS}")
        return None, None

    if not MODELO_EMOCOES.exists():

        print()
        print("ERRO: modelo de emoções não encontrado.")
        print(f"Esperado: {MODELO_EMOCOES}")
        return None, None

    pacote_pessoas = joblib.load(MODELO_PESSOAS)
    pacote_emocoes = joblib.load(MODELO_EMOCOES)

    modelo_pessoas = pacote_pessoas["modelo"]
    modelo_emocoes = pacote_emocoes["modelo"]

    return modelo_pessoas, modelo_emocoes


# ============================================================
# TESTAR UM ÁUDIO
# ============================================================

def testar_audio(
    caminho_audio,
    modelo_pessoas,
    modelo_emocoes
):

    try:

        features = extrair_features(caminho_audio)

        features = features.reshape(1, -1)

        # ----------------------------------------------------
        # Pessoa
        # ----------------------------------------------------

        pessoa = modelo_pessoas.predict(features)[0]

        probabilidades_pessoa = (
            modelo_pessoas.predict_proba(features)[0]
        )

        confianca_pessoa = float(
            np.max(probabilidades_pessoa)
        )

        # ----------------------------------------------------
        # Emoção
        # ----------------------------------------------------

        emocao = modelo_emocoes.predict(features)[0]

        probabilidades_emocao = (
            modelo_emocoes.predict_proba(features)[0]
        )

        confianca_emocao = float(
            np.max(probabilidades_emocao)
        )

        return {
            "pessoa": pessoa,
            "confianca_pessoa": confianca_pessoa,
            "emocao": emocao,
            "confianca_emocao": confianca_emocao
        }

    except Exception as erro:

        return {
            "erro": str(erro)
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print(" TESTE DOS MODELOS")
    print("=" * 65)

    # --------------------------------------------------------
    # Verificar pasta
    # --------------------------------------------------------

    if not TESTES_DIR.exists():

        print()
        print("ERRO: pasta de testes não encontrada.")
        print(f"Esperado: {TESTES_DIR}")
        return

    # --------------------------------------------------------
    # Carregar modelos
    # --------------------------------------------------------

    modelo_pessoas, modelo_emocoes = carregar_modelos()

    if modelo_pessoas is None or modelo_emocoes is None:
        return

    # --------------------------------------------------------
    # Mostrar pessoas conhecidas pelo modelo
    # --------------------------------------------------------

    pessoas_modelo = list(modelo_pessoas.classes_)

    print()
    print("Pessoas conhecidas pelo modelo:")

    for pessoa in pessoas_modelo:

        print(f"    - {pessoa}")

    # --------------------------------------------------------
    # Encontrar áudios
    # --------------------------------------------------------

    audios = sorted(
        TESTES_DIR.rglob("*.wav")
    )

    if not audios:

        print()
        print("Nenhum arquivo WAV encontrado.")
        return

    print()
    print(f"Áudios encontrados: {len(audios)}")
    print()

    # --------------------------------------------------------
    # Contadores gerais
    # --------------------------------------------------------

    acertos = 0
    erros = 0
    processados = 0

    # --------------------------------------------------------
    # Estatísticas por pessoa
    # --------------------------------------------------------

    estatisticas = {}

    for pessoa in pessoas_modelo:

        estatisticas[pessoa] = {
            "total": 0,
            "acertos": 0
        }

    # --------------------------------------------------------
    # Testar cada áudio
    # --------------------------------------------------------

    for indice, audio in enumerate(audios, start=1):

        esperado = audio.parent.name

        resultado = testar_audio(
            audio,
            modelo_pessoas,
            modelo_emocoes
        )

        # ----------------------------------------------------
        # Erro
        # ----------------------------------------------------

        if "erro" in resultado:

            print(
                f"[{indice}/{len(audios)}] "
                f"{audio.name}"
            )

            print(
                f"    ERRO: {resultado['erro']}"
            )

            print()

            continue

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        pessoa = resultado["pessoa"]
        conf_pessoa = resultado["confianca_pessoa"]

        emocao = resultado["emocao"]
        conf_emocao = resultado["confianca_emocao"]

        correto = (
            esperado.lower()
            == pessoa.lower()
        )

        # ----------------------------------------------------
        # Estatísticas
        # ----------------------------------------------------

        processados += 1

        if esperado not in estatisticas:

            estatisticas[esperado] = {
                "total": 0,
                "acertos": 0
            }

        estatisticas[esperado]["total"] += 1

        if correto:

            status = "OK"

            acertos += 1

            estatisticas[esperado]["acertos"] += 1

        else:

            status = "ERRO"

            erros += 1

        # ----------------------------------------------------
        # Mostrar resultado
        # ----------------------------------------------------

        print(
            f"[{indice}/{len(audios)}] "
            f"{audio.name}"
        )

        print(
            f"    Pessoa : {pessoa:<10} "
            f"{conf_pessoa * 100:6.2f}%   "
            f"Esperado: {esperado:<10} "
            f"[{status}]"
        )

        print(
            f"    Emoção : {emocao:<10} "
            f"{conf_emocao * 100:6.2f}%"
        )

        print()

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    if processados == 0:

        print("Nenhum áudio foi processado.")
        return

    taxa = (
        acertos
        / processados
        * 100
    )

    print("=" * 65)
    print(" RESULTADO FINAL")
    print("=" * 65)

    print()
    print(f"Áudios testados : {processados}")
    print(f"Acertos         : {acertos}")
    print(f"Erros           : {erros}")
    print(f"Taxa de acerto  : {taxa:.2f}%")

    # ========================================================
    # RESULTADO POR PESSOA
    # ========================================================

    print()
    print("=" * 65)
    print(" RESULTADO POR PESSOA")
    print("=" * 65)

    print()

    for pessoa, dados in estatisticas.items():

        total = dados["total"]
        acertos_pessoa = dados["acertos"]

        if total == 0:
            continue

        taxa_pessoa = (
            acertos_pessoa
            / total
            * 100
        )

        print(
            f"{pessoa:<12} "
            f"{acertos_pessoa}/{total} "
            f"({taxa_pessoa:.2f}%)"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 65)
    print(" TESTE FINALIZADO")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()