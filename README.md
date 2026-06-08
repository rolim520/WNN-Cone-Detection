# Detecção de Cones utilizando Redes Neurais sem Pesos

Este repositório contém o código-fonte para um sistema de detecção de objetos focado em cones de sinalização. O pipeline combina técnicas clássicas de Visão Computacional com Redes Neurais sem Pesos (Weightless Neural Networks - WNNs), especificamente as arquiteturas **WiSARD** e **ClusWiSARD**.

O objetivo primário é fornecer uma alternativa eficiente, de rápido treinamento e com baixo custo computacional quando comparada a arquiteturas de Deep Learning baseadas em convolução.

## Índice

1. [Visão Geral do Sistema](#visão-geral-do-sistema)
2. [Estrutura do Repositório](#estrutura-do-repositório)
3. [Pré-requisitos e Instalação](#pré-requisitos-e-instalação)
4. [Estrutura do Dataset](#estrutura-do-dataset)
5. [Guia de Execução](#guia-de-execução)
6. [Resultados e Comparações](#resultados-e-comparações)

---

## Visão Geral do Sistema

O pipeline de detecção atua em duas frentes principais:

1. **Extração de Candidatos (Heurística):** Utiliza limiares de cor (laranja e branco), operações morfológicas e análise de densidade para isolar "Regiões de Interesse" (RoIs) na imagem original.
2. **Classificação (WNN):** As RoIs são binarizadas (usando cor ou bordas Canny) e enviadas aos modelos WiSARD/ClusWiSARD, que classificam o recorte como "cone" ou "não cone".

---

## Estrutura do Repositório

Arquivos principais e suas funções:

* `calibration.py`: Interface gráfica para ajuste fino dos limiares de HSV, filtros morfológicos e bordas Canny.
* `utils.py`: Funções utilitárias centrais (cálculo de IoU, mAP, processamento de imagem, binarização e manipulação do modelo).
* `grid_search_wisard.py` / `grid_search_cluswisard.py`: Scripts de busca em grade  para encontrar a melhor combinação de hiperparâmetros.
* `wisard.py` / `cluswisard.py`: Scripts de validação final. Treinam os modelos múltiplas vezes para extrair a incerteza estatística (precisão, recall, F1, mAP) e exportam o melhor modelo.
* `realtime_detect.py`: Script para captura de vídeo via webcam e detecção de cones em tempo real utilizando os modelos salvos.
* `config.json`: Arquivo de configuração persistente gerado pela etapa de calibração.

---

## Pré-requisitos e Instalação

Recomenda-se a utilização de um ambiente virtual para isolar as dependências do projeto.

### 1. Clonar o repositório

```bash
git clone https://github.com/rolim520/WNN-Cone-Detection.git
cd WNN-Cone-Detection
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

No Linux, caso utilize distribuições baseadas em Debian/Ubuntu, verifique se possui o pacote do Tkinter instalado no sistema para que a interface gráfica do script de calibração funcione:

```bash
sudo apt-get install python3-tk
```

Em seguida, instale as dependências listadas no arquivo `requirements.txt` em qualquer sistema operacional:

```bash
pip install -r requirements.txt
```

---

## Estrutura do Dataset

O sistema espera que as imagens e as anotações sigam o formato padrão do YOLO (arquivos `.txt` contendo `classe x_centro y_centro largura altura` normalizados).

Crie a seguinte hierarquia de diretórios na raiz do projeto e insira seus dados:

```text
Weightless-Cone-Detection/
├── images/
│   ├── train/       # Imagens para extração de recortes e treinamento
│   └── test/        # Imagens para inferência e cálculo de mAP
└── labels/
    ├── train/       # Anotações YOLO correspondentes ao treino
    └── test/        # Anotações YOLO correspondentes ao teste
```

---

## Guia de Execução

Siga a ordem abaixo para reproduzir o pipeline completo.

### 1. Calibração

Antes de treinar as redes sem peso, é necessário calibrar os extratores de características de acordo com as condições de iluminação do seu dataset.

```bash
python calibration.py
```

*Utilização:* Utilize os sliders na interface gráfica para isolar os cones da melhor forma possível. Ao clicar em "SALVAR E SAIR", os limiares otimizados serão gravados no arquivo `config.json`.

### 2. Otimização de Hiperparâmetros

Para encontrar os parâmetros ideais (tamanho da tupla, resolução, limites de discriminação) para a WNN, execute os scripts de busca em grade. Estes scripts utilizam processamento paralelo nativo.

```bash
python grid_search_wisard.py
# ou
python grid_search_cluswisard.py
```

O script exibirá no terminal os 15 melhores resultados e os salvará em um arquivo JSON correspondente.

### 3. Treinamento e Benchmark

Após atualizar os arquivos principais com os melhores parâmetros encontrados no passo anterior, execute os avaliadores. Eles realizarão múltiplas rodadas de treinamento para fornecer uma média de desempenho (mAP@50, F1-Score, tempos de inferência).

```bash
python wisard.py
# ou
python cluswisard.py
```

*Saídas geradas:* Pasta `resultados_wisard` (ou `resultados_cluswisard`) contendo imagens com bounding boxes de teste.

* O modelo vencedor exportado em formato JSON (`melhor_modelo.json`).
* Imagens mentais (`mental_images/`) que representam os padrões visuais aprendidos pela rede.

### 4. Inferência em Tempo Real

Com o melhor modelo salvo, você pode testar a detecção utilizando uma webcam.

```bash
python realtime_detect.py
```

*Nota:* Você pode alterar a variável `TIPO_MODELO` dentro do script para alternar entre os pesos do `wisard` e `cluswisard`. Pressione a tecla `q` com a janela de exibição focada para encerrar a captura.

---

## Resultados e Comparações

As métricas geradas pelas arquiteturas sem pesos podem ser comparadas com o baseline do modelo YOLOv5 armazenado em `yolo_results.txt`. Os indicadores calculados incluem:

* **mAP@50 e mAP@50-95.**
* **F1-Score, Precisão e Recall.**
* **Média de Interseção sobre União (IoU).**
* **Tempo de Inferência (ms/imagem)** e **Tempo de Treinamento (s/ciclo)**.

Os relatórios detalhados contendo a incerteza de múltiplas execuções estão disponíveis nos arquivos JSON gerados pelos scripts de benchmark.
