import argparse
from collections import Counter
from pathlib import Path

import comum


# ============================================================
# EXIBIR RESULTADO
# ============================================================

def mostrar_resultado(resultado):

    print()

    if resultado["silencio"]:

        print(
            "  Áudio silencioso: "
            "nada para classificar."
        )

        return

    print(
        f"  Pessoa : "
        f"{resultado['pessoa']:<14} "
        f"confiança "
        f"{resultado['conf_pessoa']:.0%}"
    )

    print(
        f"  Top-3  : "
        f"{comum.formatar_ranking(resultado['ranking_pessoa'])}"
    )

    print(
        f"  Margem : "
        f"{resultado['margem_pessoa']:.0%}"
    )

    print()

    print(
        f"  Emoção : "
        f"{resultado['emocao']:<14} "
        f"confiança "
        f"{resultado['conf_emocao']:.0%}"
    )

    print(
        f"  Top-3  : "
        f"{comum.formatar_ranking(resultado['ranking_emocao'])}"
    )

    if (
        resultado["pessoa"]
        == comum.ROTULO_DESCONHECIDO
    ):

        print()

        print(
            "  Pessoa não reconhecida:"
        )

        print(
            f"    confiança mínima: "
            f"{comum.LIMIAR_PESSOA:.0%}"
        )

        print(
            f"    margem mínima: "
            f"{comum.MARGEM_PESSOA:.0%}"
        )

    if (
        resultado["emocao"]
        == comum.ROTULO_INCERTO
    ):

        print()

        print(
            "  Emoção incerta:"
        )

        print(
            f"    confiança mínima: "
            f"{comum.LIMIAR_EMOCAO:.0%}"
        )


# ============================================================
# TESTAR UM ARQUIVO
# ============================================================

def testar_arquivo(
    caminho,
    modelo_pessoas,
    modelo_emocoes
):

    caminho = Path(caminho)

    if not caminho.exists():

        print(
            f"ERRO: arquivo não encontrado:"
        )

        print(
            caminho
        )

        return

    print()
    print("=" * 70)
    print("TESTE DE UM ÁUDIO")
    print("=" * 70)

    print()
    print(
        f"Arquivo: {caminho}"
    )

    try:

        audio = comum.carregar_wav(
            caminho
        )

        resultado = comum.classificar_audio(
            audio,
            modelo_pessoas,
            modelo_emocoes
        )

        mostrar_resultado(
            resultado
        )

    except Exception as erro:

        print()
        print(
            f"ERRO ao processar áudio:"
        )

        print(
            erro
        )


# ============================================================
# TESTAR PASTA
# ============================================================

def testar_pasta(
    pasta,
    modelo_pessoas,
    modelo_emocoes,
    metadados
):

    pasta = Path(pasta)

    if not pasta.exists():

        print()
        print(
            f"ERRO: pasta não encontrada:"
        )

        print(
            pasta
        )

        return

    classes_pessoas = set(
        metadados
        .get("pessoas", {})
        .get("classes", [])
    )

    classes_emocoes = set(
        metadados
        .get("emocoes", {})
        .get("classes", [])
    )

    audios = sorted(
        pasta.rglob("*.wav")
    )

    if not audios:

        print()
        print(
            "Nenhum WAV encontrado."
        )

        return

    print()
    print("=" * 70)
    print("TESTE EM LOTE")
    print("=" * 70)

    print()
    print(
        f"Pasta: {pasta}"
    )

    print(
        f"Áudios encontrados: {len(audios)}"
    )

    print()

    # --------------------------------------------------------
    # CONTADORES
    # --------------------------------------------------------

    acertos_pessoa = 0
    total_pessoa = 0

    estatisticas_pessoas = {}

    # --------------------------------------------------------
    # PROCESSAMENTO
    # --------------------------------------------------------

    for indice, arquivo in enumerate(
        audios,
        start=1
    ):

        esperado = arquivo.parent.name

        print(
            "-" * 70
        )

        print(
            f"[{indice}/{len(audios)}] "
            f"{arquivo.name}"
        )

        try:

            audio = comum.carregar_wav(
                arquivo
            )

            resultado = comum.classificar_audio(
                audio,
                modelo_pessoas,
                modelo_emocoes
            )

        except Exception as erro:

            print(
                f"ERRO: {erro}"
            )

            continue

        if resultado["silencio"]:

            print(
                "  SILÊNCIO"
            )

            continue

        pessoa_prevista = (
            resultado["pessoa"]
        )

        emocao_prevista = (
            resultado["emocao"]
        )

        # ----------------------------------------------------
        # PESSOA
        # ----------------------------------------------------

        pessoa_esperada = esperado

        if esperado in classes_pessoas:

            total_pessoa += 1

            correto = (
                pessoa_prevista.lower()
                == pessoa_esperada.lower()
            )

            if correto:
                acertos_pessoa += 1

            if esperado not in estatisticas_pessoas:

                estatisticas_pessoas[
                    esperado
                ] = {
                    "total": 0,
                    "acertos": 0
                }

            estatisticas_pessoas[
                esperado
            ]["total"] += 1

            if correto:

                estatisticas_pessoas[
                    esperado
                ]["acertos"] += 1

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if esperado in classes_pessoas:

            status = (
                "OK"
                if pessoa_prevista.lower()
                == esperado.lower()
                else "ERR"
            )

        else:

            status = "INFO"

        print(
            f"  {status} "
            f"Pessoa: "
            f"{pessoa_prevista:<14} "
            f"{resultado['conf_pessoa']:.0%}"
        )

        print(
            f"     Esperado: "
            f"{esperado}"
        )

        print(
            f"  Emoção: "
            f"{emocao_prevista:<14} "
            f"{resultado['conf_emocao']:.0%}"
        )

        print(
            f"  Top pessoa: "
            f"{comum.formatar_ranking(
                resultado['ranking_pessoa']
            )}"
        )

        print(
            f"  Top emoção: "
            f"{comum.formatar_ranking(
                resultado['ranking_emocao']
            )}"
        )

    # ========================================================
    # RESULTADO
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)

    print()

    print(
        f"Áudios encontrados : "
        f"{len(audios)}"
    )

    print(
        f"Testes de pessoa   : "
        f"{total_pessoa}"
    )

    print(
        f"Acertos            : "
        f"{acertos_pessoa}"
    )

    if total_pessoa > 0:

        taxa = (
            acertos_pessoa
            / total_pessoa
        )

        print(
            f"Taxa de acerto     : "
            f"{taxa:.2%}"
        )

    # ========================================================
    # POR PESSOA
    # ========================================================

    if estatisticas_pessoas:

        print()
        print(
            "=" * 70
        )

        print(
            "RESULTADO POR PESSOA"
        )

        print(
            "=" * 70
        )

        print()

        for pessoa, dados in sorted(
            estatisticas_pessoas.items()
        ):

            total = dados["total"]
            acertos = dados["acertos"]

            taxa = (
                acertos / total
                if total > 0
                else 0
            )

            print(
                f"{pessoa:<15} "
                f"{acertos}/{total} "
                f"({taxa:.2%})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Testa os modelos de pessoas "
            "e emoções."
        )
    )

    parser.add_argument(
        "--arquivo",
        help=(
            "Classifica um único arquivo WAV."
        )
    )

    parser.add_argument(
        "--pasta",
        help=(
            "Classifica todos os WAVs "
            "encontrados na pasta."
        )
    )

    parser.add_argument(
        "--listar-dispositivos",
        action="store_true",
        help=(
            "Lista dispositivos de áudio."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DISPOSITIVOS
    # --------------------------------------------------------

    if args.listar_dispositivos:

        comum.listar_dispositivos()

        return

    # --------------------------------------------------------
    # CARREGAR MODELOS
    # --------------------------------------------------------

    try:

        (
            modelo_pessoas,
            modelo_emocoes,
            metadados
        ) = comum.carregar_modelos()

    except Exception as erro:

        print()
        print(
            f"ERRO ao carregar modelos:"
        )

        print(
            erro
        )

        return

    print()
    print("=" * 70)
    print("MODELOS CARREGADOS")
    print("=" * 70)

    print()

    print(
        "Pessoas:"
    )

    for classe in modelo_pessoas.classes_:

        print(
            f"   - {classe}"
        )

    print()

    print(
        "Emoções:"
    )

    for classe in modelo_emocoes.classes_:

        print(
            f"   - {classe}"
        )

    # --------------------------------------------------------
    # ARQUIVO
    # --------------------------------------------------------

    if args.arquivo:

        testar_arquivo(
            args.arquivo,
            modelo_pessoas,
            modelo_emocoes
        )

        return

    # --------------------------------------------------------
    # PASTA
    # --------------------------------------------------------

    if args.pasta:

        testar_pasta(
            args.pasta,
            modelo_pessoas,
            modelo_emocoes,
            metadados
        )

        return

    # --------------------------------------------------------
    # SEM ARGUMENTO
    # --------------------------------------------------------

    print()
    print(
        "Nenhum arquivo ou pasta foi informado."
    )

    print()
    print(
        "Exemplo:"
    )

    print(
        "python src/05_testar_modelos.py "
        "--pasta reunioes/reuniao_01/testes"
    )


if __name__ == "__main__":

    main()