import argparse

import comum


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Treina o modelo de identificação "
            "de pessoas."
        )
    )

    parser.add_argument(
        "--forcar",
        action="store_true",
        help=(
            "Treina mesmo quando alguma classe "
            "está abaixo do mínimo."
        )
    )

    parser.add_argument(
        "--so-contar",
        action="store_true",
        help=(
            "Apenas mostra a quantidade de "
            "áudios por pessoa."
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
    # APENAS CONTAGEM
    # --------------------------------------------------------

    if args.so_contar:

        comum.imprimir_contagem(
            "CONTAGEM DO DATASET DE PESSOAS",
            comum.contar_wavs(
                comum.PASTA_DATASET_PESSOAS
            ),
            minimo=comum.MINIMO_POR_PESSOA
        )

        return

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    comum.treinar(
        pasta_dataset=(
            comum.PASTA_DATASET_PESSOAS
        ),

        nome_modelo=(
            "modelo_pessoas.pkl"
        ),

        chave_meta="pessoas",

        titulo=(
            "TREINO DO MODELO DE PESSOAS "
            "(QUEM ESTÁ FALANDO?)"
        ),

        minimo_por_classe=(
            comum.MINIMO_POR_PESSOA
        ),

        minimo_classes=(
            comum.MINIMO_PESSOAS
        ),

        forcar=args.forcar,

        n_arvores=args.arvores,

        extrator=(
            comum.extrair_features_pessoa
        )
    )

    print()
    print(
        "Próximo passo:"
    )

    print(
        "python src/04_treinar_emocoes.py"
    )


if __name__ == "__main__":
    main()