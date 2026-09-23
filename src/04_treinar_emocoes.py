import argparse

import comum


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Treina o modelo de reconhecimento "
            "de emoções."
        )
    )

    parser.add_argument(
        "--forcar",
        action="store_true",
        help=(
            "Treina mesmo com classes "
            "abaixo do mínimo."
        )
    )

    parser.add_argument(
        "--so-contar",
        action="store_true",
        help=(
            "Apenas mostra a quantidade "
            "de áudios por emoção."
        )
    )

    parser.add_argument(
        "--arvores",
        type=int,
        default=300,
        help=(
            "Quantidade de árvores da Random Forest."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CONTAGEM
    # --------------------------------------------------------

    contagem = comum.contar_wavs(
        comum.PASTA_DATASET_EMOCOES
    )

    if args.so_contar:

        comum.imprimir_contagem(
            "CONTAGEM DO DATASET DE EMOÇÕES",
            contagem,
            minimo=comum.MINIMO_POR_EMOCAO
        )

        return

    # --------------------------------------------------------
    # VERIFICAR EMOÇÕES
    # --------------------------------------------------------

    faltando = [
        emocao
        for emocao in comum.EMOCOES
        if emocao not in contagem
    ]

    extras = [
        emocao
        for emocao in contagem
        if emocao not in comum.EMOCOES
    ]

    if faltando:

        print(
            f"AVISO: emoções obrigatórias "
            f"sem pasta/áudios: {faltando}"
        )

    if extras:

        print(
            f"AVISO: pastas fora das emoções "
            f"obrigatórias: {extras}"
        )

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    comum.treinar(
        pasta_dataset=(
            comum.PASTA_DATASET_EMOCOES
        ),

        nome_modelo=(
            "modelo_emocoes.pkl"
        ),

        chave_meta="emocoes",

        titulo=(
            "TREINO DO MODELO DE EMOÇÕES "
            "(COMO A FALA SOOU?)"
        ),

        minimo_por_classe=(
            comum.MINIMO_POR_EMOCAO
        ),

        minimo_classes=len(
            comum.EMOCOES
        ),

        forcar=args.forcar,

        n_arvores=args.arvores,

        extrator=(
            comum.extrair_features_emocao
        )
    )

    print()
    print(
        "Próximo passo:"
    )

    print(
        "python src/05_testar_modelos.py "
        "--pasta reunioes/reuniao_01/testes"
    )


if __name__ == "__main__":
    main()