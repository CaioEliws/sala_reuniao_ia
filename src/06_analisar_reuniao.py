from pathlib import Path
import csv
import json
import librosa
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
REUNIAO_DIR = BASE_DIR / "reunioes" / "reuniao_01"
TESTES_DIR = REUNIAO_DIR / "testes"
MODELOS_DIR = BASE_DIR / "modelos"
MODELO_PESSOAS = MODELOS_DIR / "modelo_pessoas.pkl"
MODELO_EMOCOES = MODELOS_DIR / "modelo_emocoes.pkl"
CSV_PATH = REUNIAO_DIR / "registros.csv"
METADADOS_PATH = REUNIAO_DIR / "metadados_analise.json"

SAMPLE_RATE = 16000
N_MFCC = 13

def extrair_features(caminho_audio):
    y, sr = librosa.load(
        caminho_audio,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Áudio vazio.")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = librosa.feature.rms(y=y)

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_centroid_mean = np.mean(
        spectral_centroid
    )

    spectral_centroid_std = np.std(
        spectral_centroid
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    spectral_bandwidth_mean = np.mean(
        spectral_bandwidth
    )

    spectral_bandwidth_std = np.std(
        spectral_bandwidth
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    spectral_rolloff_mean = np.mean(
        spectral_rolloff
    )

    spectral_rolloff_std = np.std(
        spectral_rolloff
    )

    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr
    )

    pitch_values = pitches[pitches > 0]

    if len(pitch_values) > 0:

        pitch_mean = np.mean(
            pitch_values
        )

        pitch_std = np.std(
            pitch_values
        )

    else:

        pitch_mean = 0.0
        pitch_std = 0.0

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,

        [
            rms_mean,
            rms_std
        ],

        [
            zcr_mean,
            zcr_std
        ],

        [
            spectral_centroid_mean,
            spectral_centroid_std
        ],

        [
            spectral_bandwidth_mean,
            spectral_bandwidth_std
        ],

        [
            spectral_rolloff_mean,
            spectral_rolloff_std
        ],

        [
            pitch_mean,
            pitch_std
        ]
    ])

    return features

print("=" * 70)
print("ANÁLISE GERAL DA REUNIÃO")
print("=" * 70)

if not MODELO_PESSOAS.exists():
    print("\n❌ Modelo de pessoas não encontrado:")
    print(f"   {MODELO_PESSOAS}")
    print("\nExecute primeiro:")
    print("   python .\\src\\03_treinar_pessoas.py")
    raise SystemExit(1)

if not MODELO_EMOCOES.exists():
    print("\n❌ Modelo de emoções não encontrado:")
    print(f"   {MODELO_EMOCOES}")
    print("\nExecute primeiro:")
    print("   python .\\src\\04_treinar_emocoes.py")
    raise SystemExit(1)

if not TESTES_DIR.exists():
    print("\n❌ Diretório de testes não encontrado:")
    print(f"   {TESTES_DIR}")
    print("\nCrie a estrutura:")
    print("   reunioes\\reuniao_01\\testes\\caio")
    print("   reunioes\\reuniao_01\\testes\\lorenzo")
    print("   reunioes\\reuniao_01\\testes\\lucascas")
    raise SystemExit(1)

print("\n🤖 Carregando modelos...")
modelo_pessoas = joblib.load(
    MODELO_PESSOAS
)
modelo_emocoes = joblib.load(
    MODELO_EMOCOES
)

if isinstance(modelo_pessoas, dict):
    if "modelo" in modelo_pessoas:
        modelo_pessoas = modelo_pessoas["modelo"]
    elif "model" in modelo_pessoas:
        modelo_pessoas = modelo_pessoas["model"]
    else:
        print(
            "\n❌ Não foi possível encontrar o classificador "
            "dentro do modelo de pessoas."
        )
        print(
            f"   Chaves encontradas: "
            f"{list(modelo_pessoas.keys())}"
        )
        raise SystemExit(1)

if isinstance(modelo_emocoes, dict):
    if "modelo" in modelo_emocoes:
        modelo_emocoes = modelo_emocoes["modelo"]
    elif "model" in modelo_emocoes:
        modelo_emocoes = modelo_emocoes["model"]
    else:
        print(
            "\n❌ Não foi possível encontrar o classificador "
            "dentro do modelo de emoções."
        )
        print(
            f"   Chaves encontradas: "
            f"{list(modelo_emocoes.keys())}"
        )
        raise SystemExit(1)

print(" ✅ Modelo de pessoas carregado")
print(
    f" Classes: {list(modelo_pessoas.classes_)}"
)
print(" ✅ Modelo de emoções carregado")
print(
    f" Classes: {list(modelo_emocoes.classes_)}"
)

arquivos_audio = sorted(
    TESTES_DIR.rglob("*.wav")
)

if not arquivos_audio:
    print("\n❌ Nenhum arquivo WAV encontrado.")
    print("\nColoque os áudios em:")
    print(f"   {TESTES_DIR}")
    raise SystemExit(1)

print("\n🎵 Arquivos encontrados:")
print(
    f" {len(arquivos_audio)} segmento(s) de áudio"
)

REUNIAO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

registros = []
tempo_atual = 0.0

print("\n" + "=" * 70)
print("PROCESSANDO REUNIÃO")
print("=" * 70)
print(
    "\nA análise individual dos áudios está sendo realizada..."
)

for caminho_audio in arquivos_audio:
    try:
        duracao = librosa.get_duration(
            path=caminho_audio
        )

        inicio = tempo_atual
        fim = inicio + duracao

        features = extrair_features(
            caminho_audio
        )

        X = features.reshape(
            1,
            -1
        )

        pessoa_predita = modelo_pessoas.predict(
            X
        )[0]

        probabilidades_pessoa = (
            modelo_pessoas.predict_proba(X)[0]
        )

        indice_pessoa = np.argmax(
            probabilidades_pessoa
        )

        conf_pessoa = float(
            probabilidades_pessoa[
                indice_pessoa
            ]
        )

        emocao_predita = modelo_emocoes.predict(
            X
        )[0]

        probabilidades_emocao = (
            modelo_emocoes.predict_proba(X)[0]
        )

        indice_emocao = np.argmax(
            probabilidades_emocao
        )

        conf_emocao = float(
            probabilidades_emocao[
                indice_emocao
            ]
        )

        pessoa_esperada = caminho_audio.parent.name

        if pessoa_predita == pessoa_esperada:
            status = "OK"
        else:
            status = "DIFERENTE"

        arquivo_relativo = str(
            caminho_audio.relative_to(
                REUNIAO_DIR
            )
        )

        registro = {
            "inicio": round(
                inicio,
                2
            ),
            "fim": round(
                fim,
                2
            ),
            "pessoa": str(
                pessoa_predita
            ),
            "conf_pessoa": round(
                conf_pessoa,
                4
            ),
            "emocao": str(
                emocao_predita
            ),
            "conf_emocao": round(
                conf_emocao,
                4
            ),
            "arquivo": arquivo_relativo
        }

        registros.append(
            registro
        )

        tempo_atual = fim

    except Exception as erro:
        print(
            f"\n⚠️ Erro ao processar:"
        )
        print(
            f"   {caminho_audio.name}"
        )
        print(
            f"   {erro}"
        )

total_processados = len(
    registros
)

if total_processados == 0:
    print("\n❌ Nenhum áudio foi processado com sucesso.")
    raise SystemExit(1)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:
    escritor = csv.DictWriter(
        arquivo,
        fieldnames=[
            "inicio",
            "fim",
            "pessoa",
            "conf_pessoa",
            "emocao",
            "conf_emocao",
            "arquivo"
        ]
    )

    escritor.writeheader()

    escritor.writerows(
        registros
    )

confianca_pessoa_media = float(
    np.mean([
        registro["conf_pessoa"]
        for registro in registros
    ])
)

confianca_emocao_media = float(
    np.mean([
        registro["conf_emocao"]
        for registro in registros
    ])
)

estatisticas_pessoas = {}

for registro in registros:
    pessoa = registro["pessoa"]

    if pessoa not in estatisticas_pessoas:
        estatisticas_pessoas[pessoa] = {
            "segmentos": 0,
            "duracao": 0.0,
            "confianca": [],
            "acertos": 0
        }

    estatisticas_pessoas[pessoa]["segmentos"] += 1

    duracao_segmento = (
        registro["fim"]
        - registro["inicio"]
    )

    estatisticas_pessoas[pessoa]["duracao"] += (
        duracao_segmento
    )

    estatisticas_pessoas[pessoa]["confianca"].append(
        registro["conf_pessoa"]
    )

for registro in registros:
    pessoa_predita = registro["pessoa"]
    caminho = registro["arquivo"]
    caminho_completo = REUNIAO_DIR / caminho
    pessoa_esperada = caminho_completo.parent.name

    if pessoa_predita == pessoa_esperada:
        if pessoa_predita in estatisticas_pessoas:
            estatisticas_pessoas[
                pessoa_predita
            ]["acertos"] += 1

estatisticas_emocoes = {}

for registro in registros:
    emocao = registro["emocao"]

    if emocao not in estatisticas_emocoes:
        estatisticas_emocoes[emocao] = {
            "segmentos": 0,
            "duracao": 0.0,
            "confianca": []
        }

    estatisticas_emocoes[emocao]["segmentos"] += 1

    duracao_segmento = (
        registro["fim"]
        - registro["inicio"]
    )

    estatisticas_emocoes[emocao]["duracao"] += (
        duracao_segmento
    )

    estatisticas_emocoes[emocao]["confianca"].append(
        registro["conf_emocao"]
    )

LIMITE_BAIXA_CONFIANCA = 0.70

baixa_confianca_pessoa = [
    registro
    for registro in registros
    if registro["conf_pessoa"] <
    LIMITE_BAIXA_CONFIANCA
]

baixa_confianca_emocao = [
    registro
    for registro in registros
    if registro["conf_emocao"] <
    LIMITE_BAIXA_CONFIANCA
]

metadados = {
    "tipo_analise":
        "segmentos_de_audio_gravados",
    "total_arquivos_encontrados":
        len(arquivos_audio),
    "total_arquivos_processados":
        total_processados,
    "duracao_total_segundos":
        round(
            tempo_atual,
            2
        ),
    "confianca_media_pessoa":
        round(
            confianca_pessoa_media,
            4
        ),
    "confianca_media_emocao":
        round(
            confianca_emocao_media,
            4
        ),
    "total_baixa_confianca_pessoa":
        len(
            baixa_confianca_pessoa
        ),
    "total_baixa_confianca_emocao":
        len(
            baixa_confianca_emocao
        ),
    "limite_baixa_confianca":
        LIMITE_BAIXA_CONFIANCA,
    "modelo_pessoas":
        str(
            MODELO_PESSOAS
        ),
    "modelo_emocoes":
        str(
            MODELO_EMOCOES
        )
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

print("\n" + "=" * 70)
print("RESUMO GERAL DA REUNIÃO")
print("=" * 70)

print("\n📋 INFORMAÇÕES GERAIS")
print("-" * 70)
print(
    f"Segmentos analisados : {total_processados}"
)
print(
    f"Duração total : {tempo_atual:.2f} segundos"
)
print(
    f"Confiança pessoa : "
    f"{confianca_pessoa_media * 100:.2f}%"
)
print(
    f"Confiança emoção : "
    f"{confianca_emocao_media * 100:.2f}%"
)

print("\n👥 PARTICIPANTES")
print("-" * 70)

for pessoa, dados in sorted(
    estatisticas_pessoas.items()
):
    percentual_tempo = (
        dados["duracao"]
        / tempo_atual
        * 100
    )

    confianca_media = np.mean(
        dados["confianca"]
    )

    print(
        f"{pessoa:<15}"
        f" {dados['duracao']:>6.2f}s"
        f"  ({percentual_tempo:>5.1f}%)"
        f"  {dados['segmentos']} segmento(s)"
        f"  confiança {confianca_media * 100:>5.1f}%"
    )

print("\n🎭 EMOÇÕES IDENTIFICADAS")
print("-" * 70)

for emocao, dados in sorted(
    estatisticas_emocoes.items()
):
    percentual_tempo = (
        dados["duracao"]
        / tempo_atual
        * 100
    )

    confianca_media = np.mean(
        dados["confianca"]
    )

    print(
        f"{emocao:<15}"
        f" {dados['duracao']:>6.2f}s"
        f"  ({percentual_tempo:>5.1f}%)"
        f"  {dados['segmentos']} segmento(s)"
        f"  confiança {confianca_media * 100:>5.1f}%"
    )

print("\n📊 QUALIDADE DAS PREVISÕES")
print("-" * 70)

total_baixa_pessoa = len(
    baixa_confianca_pessoa
)

total_baixa_emocao = len(
    baixa_confianca_emocao
)

print(
    f"Baixa confiança em pessoa : "
    f"{total_baixa_pessoa}/{total_processados}"
)

print(
    f"Baixa confiança em emoção : "
    f"{total_baixa_emocao}/{total_processados}"
)

total_diferentes = 0

for registro in registros:
    caminho_completo = (
        REUNIAO_DIR /
        registro["arquivo"]
    )

    pessoa_esperada = (
        caminho_completo.parent.name
    )

    if registro["pessoa"] != pessoa_esperada:
        total_diferentes += 1

print(
    f"Identificações diferentes : "
    f"{total_diferentes}/{total_processados}"
)

print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)
print("\n📄 Registros detalhados:")
print(
    f" {CSV_PATH}"
)
print("\n📄 Metadados:")
print(
    f" {METADADOS_PATH}"
)
print("\n💡 O CSV contém os dados individuais de cada áudio.")
print(
    "💡 O relatório geral pode ser gerado pelo script 07."
)

print("\n" + "=" * 70)
print("✅ ANÁLISE FINALIZADA COM SUCESSO!")
print("=" * 70)
print("\nPróximo passo:")
print(
    " python .\src\07_gerar_relatorio.py"
)
print("\n" + "=" * 70)