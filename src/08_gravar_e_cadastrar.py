from pathlib import Path
from datetime import datetime
import subprocess
import sys

import joblib
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

import comum  # ← ADICIONADO


SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PESSOAS_DIR = BASE_DIR / "dataset_pessoas"
MODELOS_DIR = BASE_DIR / "modelos"

MODELO_PESSOAS_PATH = MODELOS_DIR / "modelo_pessoas.pkl"
MODELO_EMOCOES_PATH = MODELOS_DIR / "modelo_emocoes.pkl"

GRAVACOES_DEMONSTRACAO_DIR = (
    BASE_DIR
    / "reunioes"
    / "reuniao_01"
    / "gravacoes_demonstracao"
)

TREINAR_PESSOAS_SCRIPT = (
    BASE_DIR
    / "src"
    / "03_treinar_pessoas.py"
)

SAMPLE_RATE = 16000
CANAIS = 1
DURACAO_PADRAO = 4

CONFIANCA_MINIMA_DESCONHECIDO = 0.70
LIMITE_CONFIANCA_EMOCAO = 0.55
LIMITE_DIFERENCA_EMOCOES = 0.10


def limpar_nome_pessoa(nome: str) -> str:
    nome = nome.strip().lower().replace(" ", "_")

    caracteres_permitidos = (
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "_-"
    )

    return "".join(
        caractere
        for caractere in nome
        if caractere in caracteres_permitidos
    )


def carregar_modelo(caminho: Path):
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
                f"O arquivo {caminho.name} não contém "
                "um modelo válido."
            )

        if classes is None and hasattr(modelo, "classes_"):
            classes = modelo.classes_

        if classes is None:
            raise ValueError(
                f"Não foi possível identificar as classes "
                f"do modelo {caminho.name}."
            )

        return modelo, np.array(classes)

    modelo = objeto

    if not hasattr(modelo, "predict"):
        raise ValueError(
            f"O arquivo {caminho.name} não contém "
            "um modelo válido."
        )

    classes = getattr(modelo, "classes_", None)

    if classes is None:
        raise ValueError(
            f"Não foi possível identificar as classes "
            f"de {caminho.name}."
        )

    return modelo, np.array(classes)

def gravar_audio(
    caminho_saida: Path,
    duracao: int = DURACAO_PADRAO
):
    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(" GRAVAÇÃO DE ÁUDIO")
    print("=" * 70)
    print(f"Duração: {duracao} segundos")
    print()
    print("Fale próximo ao microfone.")
    print("Evite ruídos externos.")
    print("Utilize uma frase completa.")
    print("Mantenha distância semelhante do microfone.")
    print()

    input(
        "Pressione ENTER para iniciar a gravação."
    )

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

    print()
    print("Gravação concluída.")
    print(f"Arquivo salvo em: {caminho_saida}")

    return caminho_saida


def gerar_nome_gravacao(
    prefixo: str = "audio"
) -> str:
    agora = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return f"{prefixo}_{agora}.wav"


def prever_pessoa(caminho_audio: Path):
    modelo, classes = carregar_modelo(
        MODELO_PESSOAS_PATH
    )

    features = comum.extrair_features_pessoa(
        caminho_audio
    )

    entrada = features.reshape(1, -1)

    if not hasattr(modelo, "predict_proba"):
        raise ValueError(
            "O modelo de pessoas não possui "
            "o método predict_proba."
        )

    probabilidades = modelo.predict_proba(
        entrada
    )[0]

    probabilidades_ordenadas = sorted(
        zip(classes, probabilidades),
        key=lambda item: item[1],
        reverse=True
    )

    pessoa_prevista = str(
        probabilidades_ordenadas[0][0]
    )

    confianca = float(
        probabilidades_ordenadas[0][1]
    )

    return (
        pessoa_prevista,
        confianca,
        probabilidades_ordenadas
    )


def prever_emocao(caminho_audio: Path):
    modelo, classes = carregar_modelo(
        MODELO_EMOCOES_PATH
    )

    features = comum.extrair_features_emocao(
        caminho_audio
    )

    features = np.asarray(
        features,
        dtype=np.float32
    ).flatten()

    entrada = features.reshape(1, -1)

    if not hasattr(modelo, "predict_proba"):
        raise ValueError(
            "O modelo de emoções não possui "
            "o método predict_proba."
        )

    probabilidades = modelo.predict_proba(
        entrada
    )[0]

    probabilidades_ordenadas = sorted(
        zip(classes, probabilidades),
        key=lambda item: item[1],
        reverse=True
    )

    emocao_prevista = str(
        probabilidades_ordenadas[0][0]
    )

    confianca = float(
        probabilidades_ordenadas[0][1]
    )

    if len(probabilidades_ordenadas) >= 2:
        segunda_confianca = float(
            probabilidades_ordenadas[1][1]
        )

        diferenca = (
            confianca - segunda_confianca
        )
    else:
        diferenca = confianca

    return (
        emocao_prevista,
        confianca,
        probabilidades_ordenadas,
        diferenca
    )


def exibir_probabilidades(
    titulo: str,
    probabilidades
):
    print()
    print(titulo)

    for classe, probabilidade in probabilidades:
        print(
            f"   {str(classe):<15}"
            f"{probabilidade * 100:6.2f}%"
        )


def classificar_confianca_emocao(
    confianca: float,
    diferenca: float
) -> str:
    if confianca < LIMITE_CONFIANCA_EMOCAO:
        return "INCERTA"

    if diferenca < LIMITE_DIFERENCA_EMOCOES:
        return "INCERTA"

    return "CONFIÁVEL"


def modo_identificar():
    print()
    print("=" * 70)
    print(" MODO 1 - IDENTIFICAR PESSOA E EMOÇÃO")
    print("=" * 70)

    print()
    print("Pessoas conhecidas pelo modelo:")

    _, classes_pessoas = carregar_modelo(
        MODELO_PESSOAS_PATH
    )

    for classe in classes_pessoas:
        print(f"   - {classe}")

    print()
    print("O áudio será utilizado apenas para teste.")
    print("Ele NÃO será adicionado ao dataset.")

    caminho_audio = (
        GRAVACOES_DEMONSTRACAO_DIR
        / gerar_nome_gravacao("identificacao")
    )

    gravar_audio(caminho_audio)

    print()
    print("Analisando áudio...")
    print("Extraindo características da voz...")

    (
        pessoa,
        confianca_pessoa,
        probabilidades_pessoas
    ) = prever_pessoa(caminho_audio)

    (
        emocao,
        confianca_emocao,
        probabilidades_emocoes,
        diferenca_emocoes
    ) = prever_emocao(caminho_audio)

    status_emocao = classificar_confianca_emocao(
        confianca_emocao,
        diferenca_emocoes
    )

    print()
    print("=" * 70)
    print(" RESULTADO DA IDENTIFICAÇÃO")
    print("=" * 70)

    print()
    print("PESSOA IDENTIFICADA")
    print("-" * 70)

    print(
        f"Pessoa: {pessoa}"
    )

    print(
        f"Confiança: {confianca_pessoa * 100:.2f}%"
    )

    print()
    print("ANÁLISE DA EMOÇÃO")
    print("-" * 70)

    if status_emocao == "CONFIÁVEL":

        print(
            f"Emoção detectada: {emocao}"
        )

        print(
            f"Confiança: "
            f"{confianca_emocao * 100:.2f}%"
        )

    else:

        print(
            f"Emoção mais provável: {emocao}"
        )

        print(
            f"Confiança: "
            f"{confianca_emocao * 100:.2f}%"
        )

    print()
    print(
        "Diferença entre as duas principais emoções: "
        f"{diferenca_emocoes * 100:.2f}%"
    )

    print()
    print("Áudio de teste preservado em:")
    print(caminho_audio)


def modo_cadastrar():
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
        print("Nome inválido.")
        return

    pasta_pessoa = (
        DATASET_PESSOAS_DIR
        / nome_pessoa
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

    try:
        quantidade = int(
            input(
                "Quantos áudios deseja gravar? "
                "(recomendado: 20 a 30): "
            )
        )
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
    print(f"Quantidade: {quantidade}")
    print(f"Duração: {duracao} segundos")
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
            pasta_pessoa
            / nome_arquivo
        )

        print()
        print(
            f"[{numero}/{quantidade}] "
            "Prepare-se para falar."
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
    print(" CADASTRO CONCLUÍDO")
    print("=" * 70)
    print(
        f"Áudios gravados: "
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
            "Execute: "
            "python .\\src\\03_treinar_pessoas.py"
        )
        return

    treinar_modelo_pessoas()


def treinar_modelo_pessoas():
    print()
    print("=" * 70)
    print(" TREINANDO MODELO DE PESSOAS")
    print("=" * 70)

    if not TREINAR_PESSOAS_SCRIPT.exists():
        print(
            "Script não encontrado:"
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
            "O arquivo modelo_pessoas.pkl "
            "foi recriado."
        )
    else:
        print("=" * 70)
        print(" ERRO AO TREINAR O MODELO")
        print("=" * 70)
        print(
            "Execute manualmente:"
        )
        print(
            "python .\\src\\03_treinar_pessoas.py"
        )


def exibir_menu():
    print()
    print("=" * 70)
    print(" SALA DE REUNIÃO INTELIGENTE")
    print(" GRAVAÇÃO, IDENTIFICAÇÃO E CADASTRO")
    print("=" * 70)
    print()
    print("1 - Gravar áudio e identificar pessoa e emoção")
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