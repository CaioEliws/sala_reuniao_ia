from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REUNIAO_DIR = BASE_DIR / "reunioes" / "reuniao_01"

CSV_PATH = REUNIAO_DIR / "registros.csv"

GRAFICOS_DIR = REUNIAO_DIR / "graficos"

RELATORIO_PATH = REUNIAO_DIR / "relatorio.txt"


# Limite utilizado para identificar segmentos com baixa confiança
LIMITE_CONFIANCA = 0.70


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def formatar_tempo(segundos):
    """
    Converte segundos para o formato HH:MM:SS.
    """

    segundos = int(round(segundos))

    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos_restantes = segundos % 60

    return f"{horas:02d}:{minutos:02d}:{segundos_restantes:02d}"


def formatar_percentual(valor):
    """
    Formata um número como percentual.
    """

    return f"{valor:.2f}%"


# ============================================================
# VERIFICAÇÕES
# ============================================================

print("=" * 70)
print("GERADOR DE RELATÓRIO")
print("=" * 70)

print(f"\n📁 Arquivo CSV:")
print(f"   {CSV_PATH}")

if not CSV_PATH.exists():
    print("\n❌ ERRO: o arquivo registros.csv não foi encontrado.")
    print("\nExecute primeiro:")
    print("   python .\\src\\06_analisar_reuniao.py")
    raise SystemExit(1)


GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LEITURA DO CSV
# ============================================================

print("\n📊 Carregando registros...")

try:
    df = pd.read_csv(CSV_PATH)
except Exception as erro:
    print(f"\n❌ Erro ao ler o CSV:")
    print(f"   {erro}")
    raise SystemExit(1)


if df.empty:
    print("\n❌ O registros.csv está vazio.")
    raise SystemExit(1)


print(f"   Registros encontrados: {len(df)}")


# ============================================================
# CONVERSÃO DOS DADOS
# ============================================================

colunas_numericas = [
    "inicio",
    "fim",
    "conf_pessoa",
    "conf_emocao",
]

for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")


# Duração de cada segmento

df["duracao"] = df["fim"] - df["inicio"]

df["duracao"] = df["duracao"].clip(lower=0)


# ============================================================
# INFORMAÇÕES GERAIS
# ============================================================

total_segmentos = len(df)

duracao_total = df["duracao"].sum()

pessoas = sorted(
    df["pessoa"]
    .dropna()
    .astype(str)
    .unique()
)

emocoes = sorted(
    df["emocao"]
    .dropna()
    .astype(str)
    .unique()
)


# ============================================================
# TEMPO DE FALA POR PARTICIPANTE
# ============================================================

tempo_por_pessoa = (
    df.groupby("pessoa")["duracao"]
    .sum()
    .sort_values(ascending=False)
)


# ============================================================
# SEGMENTOS POR PARTICIPANTE
# ============================================================

segmentos_por_pessoa = (
    df["pessoa"]
    .value_counts()
)


# ============================================================
# PERCENTUAL DE TEMPO DE FALA
# ============================================================

percentual_tempo_pessoa = (
    tempo_por_pessoa / duracao_total * 100
    if duracao_total > 0
    else tempo_por_pessoa * 0
)


# ============================================================
# DISTRIBUIÇÃO GERAL DAS EMOÇÕES
# ============================================================

quantidade_emocoes = (
    df["emocao"]
    .value_counts()
)


percentual_emocoes = (
    quantidade_emocoes / total_segmentos * 100
)


# ============================================================
# TEMPO POR EMOÇÃO
# ============================================================

tempo_por_emocao = (
    df.groupby("emocao")["duracao"]
    .sum()
    .sort_values(ascending=False)
)


# ============================================================
# EMOÇÕES POR PARTICIPANTE
# ============================================================

emocao_por_pessoa = pd.crosstab(
    df["pessoa"],
    df["emocao"]
)


# ============================================================
# CONFIANÇAS
# ============================================================

conf_pessoa_media = df["conf_pessoa"].mean()

conf_emocao_media = df["conf_emocao"].mean()


# ============================================================
# BAIXA CONFIANÇA
# ============================================================

df["baixa_confianca"] = (
    (df["conf_pessoa"] < LIMITE_CONFIANCA)
    |
    (df["conf_emocao"] < LIMITE_CONFIANCA)
)


segmentos_baixa_confianca = df[
    df["baixa_confianca"]
].copy()


# ============================================================
# GRÁFICO 1
# TEMPO DE FALA POR PARTICIPANTE
# ============================================================

print("\n📈 Gerando gráfico de tempo de fala...")

plt.figure(figsize=(10, 6))

tempo_por_pessoa.plot(
    kind="bar"
)

plt.title("Tempo de fala por participante")

plt.xlabel("Participante")

plt.ylabel("Tempo de fala (segundos)")

plt.xticks(rotation=0)

plt.tight_layout()

grafico_tempo = (
    GRAFICOS_DIR /
    "tempo_fala_por_participante.png"
)

plt.savefig(grafico_tempo, dpi=150)

plt.close()


# ============================================================
# GRÁFICO 2
# DISTRIBUIÇÃO GERAL DAS EMOÇÕES
# ============================================================

print("📈 Gerando gráfico de distribuição das emoções...")

plt.figure(figsize=(10, 6))

quantidade_emocoes.plot(
    kind="bar"
)

plt.title("Distribuição geral das emoções")

plt.xlabel("Emoção")

plt.ylabel("Quantidade de segmentos")

plt.xticks(rotation=0)

plt.tight_layout()

grafico_emocoes = (
    GRAFICOS_DIR /
    "distribuicao_emocoes.png"
)

plt.savefig(grafico_emocoes, dpi=150)

plt.close()


# ============================================================
# GRÁFICO 3
# EMOÇÕES POR PARTICIPANTE
# ============================================================

print("📈 Gerando gráfico de emoções por participante...")

plt.figure(figsize=(12, 7))

emocao_por_pessoa.plot(
    kind="bar"
)

plt.title("Emoções identificadas por participante")

plt.xlabel("Participante")

plt.ylabel("Quantidade de segmentos")

plt.xticks(rotation=0)

plt.legend(
    title="Emoção"
)

plt.tight_layout()

grafico_emocoes_pessoa = (
    GRAFICOS_DIR /
    "emocoes_por_participante.png"
)

plt.savefig(
    grafico_emocoes_pessoa,
    dpi=150
)

plt.close()


# ============================================================
# GRÁFICO 4
# PERCENTUAL GERAL DAS EMOÇÕES
# ============================================================

print("📈 Gerando gráfico percentual das emoções...")

plt.figure(figsize=(8, 8))

plt.pie(
    quantidade_emocoes.values,
    labels=quantidade_emocoes.index,
    autopct="%1.1f%%"
)

plt.title("Distribuição percentual das emoções")

plt.tight_layout()

grafico_pizza = (
    GRAFICOS_DIR /
    "percentual_emocoes.png"
)

plt.savefig(
    grafico_pizza,
    dpi=150
)

plt.close()


# ============================================================
# GRÁFICO 5
# CONFIANÇA MÉDIA
# ============================================================

print("📈 Gerando gráfico de confiança média...")

confiancas = pd.Series(
    {
        "Pessoa": conf_pessoa_media * 100,
        "Emoção": conf_emocao_media * 100,
    }
)


plt.figure(figsize=(8, 6))

confiancas.plot(
    kind="bar"
)

plt.title("Confiança média das classificações")

plt.xlabel("Tipo de classificação")

plt.ylabel("Confiança (%)")

plt.ylim(0, 100)

plt.xticks(rotation=0)

plt.tight_layout()

grafico_confianca = (
    GRAFICOS_DIR /
    "confianca_media.png"
)

plt.savefig(
    grafico_confianca,
    dpi=150
)

plt.close()


# ============================================================
# RELATÓRIO TXT
# ============================================================

print("\n📝 Gerando relatório...")


linhas = []

linhas.append("=" * 70)
linhas.append("RELATÓRIO DE ANÁLISE DE ÁUDIOS")
linhas.append("=" * 70)

linhas.append("")

linhas.append(
    "Tipo de análise: segmentos de áudio gravados"
)

linhas.append(
    f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)

linhas.append("")

linhas.append("-" * 70)
linhas.append("1. RESUMO GERAL")
linhas.append("-" * 70)

linhas.append(
    f"Total de segmentos analisados: {total_segmentos}"
)

linhas.append(
    f"Duração total dos segmentos: {formatar_tempo(duracao_total)}"
)

linhas.append(
    f"Participantes identificados: {len(pessoas)}"
)

linhas.append(
    f"Emoções identificadas: {len(emocoes)}"
)

linhas.append("")

linhas.append("-" * 70)
linhas.append("2. PARTICIPANTES")
linhas.append("-" * 70)

for pessoa in tempo_por_pessoa.index:

    tempo = tempo_por_pessoa[pessoa]

    quantidade = segmentos_por_pessoa.get(
        pessoa,
        0
    )

    percentual = percentual_tempo_pessoa.get(
        pessoa,
        0
    )

    linhas.append(
        f"\nPessoa: {pessoa}"
    )

    linhas.append(
        f"  Segmentos: {quantidade}"
    )

    linhas.append(
        f"  Tempo de fala: {formatar_tempo(tempo)}"
    )

    linhas.append(
        f"  Percentual do tempo: {formatar_percentual(percentual)}"
    )


linhas.append("")

linhas.append("-" * 70)
linhas.append("3. DISTRIBUIÇÃO DAS EMOÇÕES")
linhas.append("-" * 70)

for emocao in quantidade_emocoes.index:

    quantidade = quantidade_emocoes[emocao]

    percentual = percentual_emocoes[emocao]

    tempo = tempo_por_emocao.get(
        emocao,
        0
    )

    linhas.append(
        f"\nEmoção: {emocao}"
    )

    linhas.append(
        f"  Segmentos: {quantidade}"
    )

    linhas.append(
        f"  Percentual: {formatar_percentual(percentual)}"
    )

    linhas.append(
        f"  Tempo total: {formatar_tempo(tempo)}"
    )


linhas.append("")

linhas.append("-" * 70)
linhas.append("4. EMOÇÕES POR PARTICIPANTE")
linhas.append("-" * 70)

for pessoa in emocao_por_pessoa.index:

    linhas.append(
        f"\n{pessoa}:"
    )

    for emocao in emocao_por_pessoa.columns:

        quantidade = emocao_por_pessoa.loc[
            pessoa,
            emocao
        ]

        linhas.append(
            f"  {emocao}: {quantidade} segmento(s)"
        )


linhas.append("")

linhas.append("-" * 70)
linhas.append("5. CONFIANÇA DAS CLASSIFICAÇÕES")
linhas.append("-" * 70)

linhas.append(
    f"Confiança média - pessoa: "
    f"{conf_pessoa_media * 100:.2f}%"
)

linhas.append(
    f"Confiança média - emoção: "
    f"{conf_emocao_media * 100:.2f}%"
)

linhas.append(
    f"Limite considerado baixa confiança: "
    f"{LIMITE_CONFIANCA * 100:.0f}%"
)


linhas.append("")

linhas.append("-" * 70)
linhas.append("6. SEGMENTOS COM BAIXA CONFIANÇA")
linhas.append("-" * 70)

linhas.append(
    f"Total: {len(segmentos_baixa_confianca)}"
)


if not segmentos_baixa_confianca.empty:

    for _, linha in segmentos_baixa_confianca.iterrows():

        arquivo = linha.get(
            "arquivo",
            "desconhecido"
        )

        pessoa = linha.get(
            "pessoa",
            "desconhecida"
        )

        emocao = linha.get(
            "emocao",
            "desconhecida"
        )

        conf_pessoa = linha.get(
            "conf_pessoa",
            0
        )

        conf_emocao = linha.get(
            "conf_emocao",
            0
        )

        linhas.append("")

        linhas.append(
            f"Arquivo: {arquivo}"
        )

        linhas.append(
            f"  Pessoa: {pessoa}"
        )

        linhas.append(
            f"  Confiança pessoa: "
            f"{conf_pessoa * 100:.2f}%"
        )

        linhas.append(
            f"  Emoção: {emocao}"
        )

        linhas.append(
            f"  Confiança emoção: "
            f"{conf_emocao * 100:.2f}%"
        )

else:

    linhas.append(
        "Nenhum segmento ficou abaixo do limite."
    )


linhas.append("")

linhas.append("-" * 70)
linhas.append("7. ARQUIVOS GERADOS")
linhas.append("-" * 70)

linhas.append(
    f"CSV: {CSV_PATH.name}"
)

linhas.append(
    f"Relatório: {RELATORIO_PATH.name}"
)

linhas.append(
    "Gráficos:"
)

linhas.append(
    "  - tempo_fala_por_participante.png"
)

linhas.append(
    "  - distribuicao_emocoes.png"
)

linhas.append(
    "  - emocoes_por_participante.png"
)

linhas.append(
    "  - percentual_emocoes.png"
)

linhas.append(
    "  - confianca_media.png"
)


linhas.append("")

linhas.append("=" * 70)

linhas.append(
    "FIM DO RELATÓRIO"
)

linhas.append("=" * 70)


# ============================================================
# SALVAR RELATÓRIO
# ============================================================

with open(
    RELATORIO_PATH,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(
        "\n".join(linhas)
    )


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n" + "=" * 70)

print("✅ RELATÓRIO GERADO COM SUCESSO!")

print("=" * 70)

print("\n📄 Relatório:")
print(f"   {RELATORIO_PATH}")

print("\n📊 Gráficos:")

print(
    f"   {grafico_tempo}"
)

print(
    f"   {grafico_emocoes}"
)

print(
    f"   {grafico_emocoes_pessoa}"
)

print(
    f"   {grafico_pizza}"
)

print(
    f"   {grafico_confianca}"
)

print("\n📌 Resumo:")

print(
    f"   Segmentos analisados: {total_segmentos}"
)

print(
    f"   Duração total: {formatar_tempo(duracao_total)}"
)

print(
    f"   Participantes: {len(pessoas)}"
)

print(
    f"   Confiança média pessoa: "
    f"{conf_pessoa_media * 100:.2f}%"
)

print(
    f"   Confiança média emoção: "
    f"{conf_emocao_media * 100:.2f}%"
)

print(
    f"   Baixa confiança: "
    f"{len(segmentos_baixa_confianca)}"
)

print("\n" + "=" * 70)