# Sala de Reunião Inteligente

Sistema local de análise de áudios desenvolvido em Python para identificar participantes de uma reunião e estimar a emoção vocal presente em suas falas.

O projeto utiliza processamento digital de sinais de áudio e aprendizado de máquina para executar duas tarefas independentes:

1. **Identificação do participante:** estima quem está falando.
2. **Classificação da emoção vocal:** estima se a fala apresenta características de alegria, neutralidade, tristeza ou irritação.

> **Importante:** os resultados são estimativas produzidas pelos modelos treinados. A classificação de emoção não representa um diagnóstico psicológico nem determina com certeza o estado emocional de uma pessoa.

---

## Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Objetivos](#objetivos)
- [Funcionalidades](#funcionalidades)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Funcionamento](#funcionamento)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Organização dos datasets](#organização-dos-datasets)
- [Treinamento dos modelos](#treinamento-dos-modelos)
- [Testes](#testes)
- [Interpretação dos resultados](#interpretação-dos-resultados)
- [Arquivos gerados](#arquivos-gerados)
- [Fluxo completo](#fluxo-completo)
- [Solução de problemas](#solução-de-problemas)
- [Limitações](#limitações)
- [Possíveis melhorias](#possíveis-melhorias)

---

## Sobre o projeto

O **Sala de Reunião Inteligente** é um projeto acadêmico de Inteligência Artificial que analisa gravações de voz e utiliza características acústicas para realizar previsões.

O sistema possui dois modelos independentes:

| Modelo | Pergunta respondida | Classes |
|---|---|---|
| Modelo de pessoas | Quem está falando? | Participantes cadastrados |
| Modelo de emoções | Como a fala soou? | `alegre`, `neutro`, `triste`, `irritado` |

A aplicação foi planejada para funcionar localmente, utilizando arquivos de áudio fornecidos pelo usuário, sem depender de APIs externas durante a análise.

---

## Objetivos

### Objetivo geral

Desenvolver uma aplicação local capaz de analisar gravações de voz e aplicar técnicas de aprendizado de máquina para identificar participantes e estimar emoções vocais.

### Objetivos específicos

- Organizar gravações em datasets estruturados.
- Extrair características acústicas dos arquivos.
- Treinar um modelo para identificação de pessoas.
- Treinar um modelo para classificação de emoções.
- Avaliar o desempenho dos modelos.
- Gerar matrizes de confusão.
- Exibir níveis de confiança.
- Identificar previsões com confiança insuficiente.
- Manter o processamento local.

---

## Funcionalidades

### Identificação de pessoas

- Reconhecimento de participantes cadastrados.
- Exibição da pessoa prevista.
- Exibição da confiança.
- Ranking das três classes mais prováveis.
- Retorno `desconhecido` quando os critérios de confiança não forem atendidos.

### Classificação de emoções

- Classificação em quatro emoções:
  - `alegre`
  - `neutro`
  - `triste`
  - `irritado`
- Exibição da confiança.
- Ranking das três emoções mais prováveis.
- Retorno `incerta` quando a confiança for insuficiente.

### Avaliação

Durante o treinamento, são gerados:

- Acurácia.
- Relatório de classificação.
- Matriz de confusão.
- Modelo treinado.
- Metadados em JSON.

### Testes

É possível testar:

- Um único arquivo WAV.
- Uma pasta com vários WAVs.
- Arquivos organizados em subpastas por participante.
- Pessoas cadastradas.
- Pessoas não cadastradas.

---

## Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3.12+ | Linguagem principal |
| NumPy | Operações numéricas |
| Librosa | Leitura e processamento de áudio |
| Scikit-learn | Aprendizado de máquina e métricas |
| Random Forest | Classificação |
| Joblib | Persistência dos modelos |
| Matplotlib | Gráficos e matrizes de confusão |
| JSON | Metadados |
| pathlib | Manipulação de caminhos |

### Dependências

```text
numpy
librosa
scikit-learn
joblib
matplotlib
```

---

## Funcionamento

```text
Arquivo WAV
    |
    v
Carregamento do áudio
    |
    v
Extração de características
    |
    +--------------------------+
    |                          |
    v                          v
Modelo de pessoas         Modelo de emoções
    |                          |
    v                          v
Quem está falando?       Como a fala soou?
    |                          |
    v                          v
Pessoa ou desconhecido   Emoção ou incerta
```

### Etapas

1. O áudio é carregado com taxa de amostragem de 16.000 Hz.
2. O áudio é convertido para mono.
3. As características acústicas são extraídas.
4. O vetor é enviado ao modelo correspondente.
5. O Random Forest calcula as probabilidades.
6. Regras de confiança são aplicadas.
7. O resultado é exibido no terminal.

---

## Extração de características

### Modelo de pessoas

O modelo de pessoas utiliza **76 características**, baseadas em:

- MFCC.
- Energia RMS.
- Zero Crossing Rate.
- Spectral Centroid.
- Spectral Bandwidth.
- Spectral Rolloff.
- Pitch.

Para os indicadores são calculadas estatísticas como média, desvio padrão, mínimo e máximo.

### Modelo de emoções

O modelo de emoções utiliza **80 características**:

```text
76 características acústicas
+
4 características temporais
=
80 características
```

As características temporais são:

1. Taxa de silêncio.
2. Energia no início.
3. Energia no meio.
4. Energia no final.

> Os modelos utilizam vetores diferentes. O extrator usado no treinamento deve ser o mesmo utilizado durante os testes.

---

## Estrutura do projeto

```text
sala_reuniao_ia/
│
├── dataset_pessoas/
│   ├── caio/
│   ├── lorenzo/
│   └── lucascas/
│
├── dataset_emocoes/
│   ├── alegre/
│   ├── neutro/
│   ├── triste/
│   └── irritado/
│
├── modelos/
│   ├── modelo_pessoas.pkl
│   ├── modelo_emocoes.pkl
│   └── metadados.json
│
├── reunioes/
│   └── reuniao_01/
│       ├── testes/
│       │   ├── caio/
│       │   ├── lorenzo/
│       │   └── lucascas/
│       └── graficos/
│
├── src/
│   ├── comum.py
│   ├── 03_treinar_pessoas.py
│   ├── 04_treinar_emocoes.py
│   └── 05_testar_modelos.py
│
├── .gitignore
└── README.md
```

---

## Requisitos

- Windows 10 ou Windows 11.
- Python 3.12 ou versão compatível.
- Visual Studio Code.
- PowerShell.
- Arquivos WAV organizados por classe.

A internet é necessária para instalar as dependências, mas não é necessária durante a análise local dos arquivos.

---

## Instalação

### 1. Abrir a pasta do projeto

```powershell
cd "C:\Users\SEU_USUARIO\Desktop\sala_reuniao_ia"
```

### 2. Criar o ambiente virtual

```powershell
python -m venv .venv
```

### 3. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Atualizar o pip

```powershell
python -m pip install --upgrade pip
```

### 5. Instalar dependências

```powershell
pip install numpy librosa scikit-learn joblib matplotlib
```

### 6. Verificar a instalação

```powershell
python -c "import numpy, librosa, sklearn, joblib, matplotlib; print('Dependências instaladas com sucesso.')"
```

---

## Organização dos datasets

### Dataset de pessoas

Cada participante deve possuir uma pasta própria:

```text
dataset_pessoas/
├── caio/
├── lorenzo/
└── lucascas/
```

Exemplo:

```text
dataset_pessoas/caio/caio_01.wav
dataset_pessoas/caio/caio_02.wav
dataset_pessoas/lorenzo/lorenzo_01.wav
```

### Dataset de emoções

```text
dataset_emocoes/
├── alegre/
├── neutro/
├── triste/
└── irritado/
```

Exemplo:

```text
dataset_emocoes/alegre/alegre_01.wav
dataset_emocoes/neutro/neutro_01.wav
dataset_emocoes/triste/triste_01.wav
dataset_emocoes/irritado/irritado_01.wav
```

### Recomendações

- Utilizar arquivos WAV.
- Evitar ruídos excessivos.
- Evitar áudios silenciosos.
- Variar as frases.
- Utilizar gravações em condições diferentes.
- Manter equilíbrio entre as classes.
- Evitar que todos os áudios sejam gravados exatamente no mesmo ambiente.

### Quantidade recomendada

| Dataset | Mínimo recomendado por classe |
|---|---:|
| Pessoas | 20 áudios |
| Emoções | 25 áudios |

Esses valores são referências e não garantem uma acurácia específica.

---

## Treinamento dos modelos

Execute os comandos na raiz do projeto.

### Treinar pessoas

```powershell
python src/03_treinar_pessoas.py
```

O script:

1. Verifica o dataset.
2. Conta os áudios por participante.
3. Extrai as características.
4. Divide os dados em treinamento e teste.
5. Treina o Random Forest.
6. Calcula as métricas.
7. Gera a matriz de confusão.
8. Salva o modelo e os metadados.

### Apenas contar áudios de pessoas

```powershell
python src/03_treinar_pessoas.py --so-contar
```

### Forçar treinamento de pessoas

```powershell
python src/03_treinar_pessoas.py --forcar
```

### Alterar a quantidade de árvores

```powershell
python src/03_treinar_pessoas.py --arvores 500
```

O padrão é 300 árvores.

---

### Treinar emoções

```powershell
python src/04_treinar_emocoes.py
```

### Apenas contar áudios de emoções

```powershell
python src/04_treinar_emocoes.py --so-contar
```

### Forçar treinamento de emoções

```powershell
python src/04_treinar_emocoes.py --forcar
```

### Alterar a quantidade de árvores

```powershell
python src/04_treinar_emocoes.py --arvores 500
```

---

## Testes

O sistema permite testar arquivos já gravados. Não é obrigatório utilizar gravação ao vivo.

### Testar uma pasta completa

Considerando a estrutura:

```text
reunioes/reuniao_01/testes/
├── caio/
├── lorenzo/
└── lucascas/
```

Execute:

```powershell
python src/05_testar_modelos.py --pasta reunioes/reuniao_01/testes
```

O sistema percorre os arquivos WAV e utiliza o nome da subpasta como referência para calcular a taxa de acerto da pessoa.

### Testar um único arquivo

```powershell
python src/05_testar_modelos.py --arquivo "caminho\para\audio.wav"
```

Exemplo:

```powershell
python src/05_testar_modelos.py --arquivo "reunioes/reuniao_01/testes/caio/audio_01.wav"
```

### Listar dispositivos de áudio

```powershell
python src/05_testar_modelos.py --listar-dispositivos
```

Essa opção é útil caso a gravação pelo microfone seja adicionada futuramente.

---

## Interpretação dos resultados

### Confiança

A confiança representa a maior probabilidade estimada pelo modelo entre as classes conhecidas.

Exemplo:

```text
Pessoa: caio
Confiança: 87%
```

Isso significa que `caio` recebeu a maior probabilidade estimada pelo modelo.

> A confiança do Random Forest não é uma garantia de acerto nem necessariamente uma probabilidade perfeitamente calibrada.

### Pessoa desconhecida

O sistema pode retornar:

```text
Pessoa: desconhecido
```

São utilizados os seguintes critérios:

```python
LIMIAR_PESSOA = 0.45
MARGEM_PESSOA = 0.20
```

O sistema verifica:

1. A probabilidade da primeira classe.
2. A diferença entre a primeira e a segunda maior probabilidade.

Exemplo:

```text
caio: 48%
lorenzo: 42%
```

Embora `caio` seja a primeira opção, a margem é de apenas 6%. Como a margem mínima é 20%, o sistema pode retornar `desconhecido`.

### Emoção incerta

O sistema pode retornar:

```text
Emoção: incerta
```

Isso ocorre quando a maior probabilidade é menor que:

```python
LIMIAR_EMOCAO = 0.40
```

Exemplo:

```text
neutro: 31%
alegre: 28%
triste: 23%
irritado: 18%
```

Nesse caso, `neutro` é a opção mais provável, mas a confiança não atingiu o limite definido.

### Silêncio

Áudios com energia muito baixa podem ser classificados como silenciosos.

O limite utilizado é:

```python
LIMIAR_SILENCIO = 0.004
```

Esse valor pode precisar de ajustes dependendo do microfone e das condições de gravação.

---

## Arquivos gerados

### Modelo de pessoas

```text
modelos/modelo_pessoas.pkl
```

Random Forest treinado para identificar os participantes.

### Modelo de emoções

```text
modelos/modelo_emocoes.pkl
```

Random Forest treinado para classificar emoções vocais.

### Metadados

```text
modelos/metadados.json
```

Contém:

- Classes.
- Quantidade de amostras.
- Quantidade de características.
- Amostras de treinamento.
- Amostras de teste.
- Acurácia.
- Quantidade de árvores.

### Matrizes de confusão

```text
reunioes/reuniao_01/graficos/matriz_confusao_pessoas.png
reunioes/reuniao_01/graficos/matriz_confusao_emocoes.png
```

As matrizes mostram quais classes foram reconhecidas corretamente e quais foram confundidas.

---

## Fluxo completo

### 1. Ativar o ambiente

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Conferir os datasets

```powershell
python src/03_treinar_pessoas.py --so-contar
python src/04_treinar_emocoes.py --so-contar
```

### 3. Treinar pessoas

```powershell
python src/03_treinar_pessoas.py
```

### 4. Treinar emoções

```powershell
python src/04_treinar_emocoes.py
```

### 5. Testar os arquivos

```powershell
python src/05_testar_modelos.py --pasta reunioes/reuniao_01/testes
```

### Fluxo resumido

```text
Organizar áudios
      |
      v
Conferir quantidade
      |
      v
Treinar pessoas
      |
      v
Treinar emoções
      |
      v
Testar arquivos
      |
      v
Analisar métricas e resultados
```

---

## Solução de problemas

### Python não encontrado

Verifique:

```powershell
python --version
```

Também é possível verificar o Python do ambiente virtual:

```powershell
.\.venv\Scripts\python.exe --version
```

### Ambiente virtual não ativa

Execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Módulo não encontrado

Exemplo:

```text
ModuleNotFoundError: No module named 'librosa'
```

Instale a dependência:

```powershell
pip install librosa
```

Ou reinstale todas:

```powershell
pip install numpy librosa scikit-learn joblib matplotlib
```

### Modelo não encontrado

Treine os modelos antes de testar:

```powershell
python src/03_treinar_pessoas.py
python src/04_treinar_emocoes.py
```

### Quantidade incorreta de características

Esse erro geralmente ocorre quando o vetor utilizado no teste é diferente do vetor utilizado no treinamento.

Verifique:

- Se os modelos antigos foram substituídos.
- Se treinamento e teste utilizam o mesmo `comum.py`.
- Se pessoas utiliza 76 características.
- Se emoções utiliza 80 características.

### Acurácia baixa

Possíveis causas:

- Dataset pequeno.
- Ruído.
- Classes desbalanceadas.
- Frases muito semelhantes.
- Vozes parecidas.
- Diferenças de microfone.
- Dados de teste diferentes dos dados de treinamento.

Possíveis melhorias:

- Adicionar mais gravações.
- Variar frases.
- Melhorar a qualidade dos áudios.
- Balancear as classes.
- Avaliar a matriz de confusão.
- Ajustar os hiperparâmetros.

### Emoção frequentemente aparece como `incerta`

Isso pode ocorrer quando:

- As emoções possuem características acústicas semelhantes.
- O dataset é pequeno.
- As gravações apresentam pouca expressividade.
- Há ruído.
- A maior probabilidade fica abaixo de 40%.

---

## Limitações

### Identificação baseada em características acústicas

O sistema não compreende semanticamente o conteúdo da frase. A identificação utiliza padrões acústicos da voz.

### Emoção vocal não é diagnóstico psicológico

A classificação estima características presentes na fala e não determina o estado psicológico real da pessoa.

### Dependência do dataset

A qualidade das previsões depende da quantidade, diversidade e qualidade das gravações utilizadas no treinamento.

### Possibilidade de confusão

Pessoas com vozes semelhantes podem ser confundidas.

### Pessoa desconhecida

O Random Forest não é originalmente um detector perfeito de classes desconhecidas. As regras de confiança e margem são uma estratégia de rejeição, mas não garantem detectar todas as pessoas não cadastradas.

### Confiança não é garantia

Mesmo uma previsão com alta confiança pode estar incorreta.

### Testes por pasta

Para calcular a taxa de acerto de pessoas, os arquivos precisam estar em subpastas cujo nome represente a classe real.

---

## Possíveis melhorias

- Criar uma interface gráfica para envio de arquivos.
- Gerar relatórios em PDF.
- Exportar resultados para CSV.
- Criar gráficos de distribuição de emoções.
- Analisar gravações longas divididas em blocos.
- Detectar mudanças de participante durante a reunião.
- Utilizar validação cruzada.
- Calibrar probabilidades.
- Comparar Random Forest com SVM ou redes neurais.
- Adicionar normalização das características.
- Melhorar a detecção de pessoas desconhecidas.
- Adicionar redução de ruído.
- Criar testes automatizados.
- Registrar versões dos datasets e modelos.

---

## Boas práticas

### `.gitignore`

Recomenda-se ignorar arquivos grandes ou gerados:

```gitignore
.venv/
__pycache__/
*.pyc

modelos/*.pkl

.vscode/

*.wav
*.mp3
*.ogg
*.opus

reunioes/*/testes/
```

### Recriar modelos após alterar as features

Sempre que a extração de características for modificada, treine novamente os modelos.

Não utilize um modelo antigo com um extrator incompatível.

### Manter os datasets organizados

Use nomes claros e mantenha cada áudio dentro da classe correta.

---

## Considerações finais

O projeto **Sala de Reunião Inteligente** demonstra a aplicação prática de processamento de sinais e aprendizado de máquina em análise de voz.

A arquitetura separa as responsabilidades:

| Arquivo | Responsabilidade |
|---|---|
| `comum.py` | Funções compartilhadas, features, treinamento e classificação |
| `03_treinar_pessoas.py` | Treinamento do modelo de pessoas |
| `04_treinar_emocoes.py` | Treinamento do modelo de emoções |
| `05_testar_modelos.py` | Testes e apresentação dos resultados |

O sistema foi desenvolvido para execução local, com foco em organização dos dados, reprodutibilidade dos treinamentos e transparência dos resultados.
