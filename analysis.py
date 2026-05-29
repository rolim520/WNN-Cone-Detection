import cv2
import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt

def analisar_dataset_treino(pasta_imagens, amostras_max=1000):
    caminhos = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        caminhos.extend(glob.glob(os.path.join(pasta_imagens, ext)))
    caminhos.sort()
    
    # Limita o número de imagens para não demorar uma eternidade na primeira vez
    caminhos = caminhos[:amostras_max]
    
    print(f"Iniciando análise de {len(caminhos)} imagens...")

    # Listas para guardar as estatísticas
    stats_densidade_laranja = []
    stats_densidade_total = []
    stats_aspect_ratio = []
    stats_area_caixa = []

    for idx, caminho_img in enumerate(caminhos):
        img = cv2.imread(caminho_img)
        if img is None: continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Máscaras
        mask1 = cv2.inRange(hsv, np.array([0, 87, 100]), np.array([21, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([177, 87, 100]), np.array([179, 255, 255]))
        mask_laranja = cv2.bitwise_or(mask1, mask2)
        mask_laranja_limpa = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        mask_branco = cv2.inRange(hsv, np.array([0, 0, 152]), np.array([179, 74, 255]))
        mask_branco_limpa = cv2.morphologyEx(mask_branco, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask_branco_limpa = cv2.morphologyEx(mask_branco_limpa, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

        # 2. Extração Básica (Apenas os blocos laranjas brutos, sem K-NN por enquanto)
        contornos, _ = cv2.findContours(mask_laranja_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contornos:
            area_contorno = cv2.contourArea(cnt)
            if area_contorno > 5: # Pegando quase tudo para ver a distribuição real
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Ignora linhas bizarras (1px de espessura)
                if w < 3 or h < 3: continue
                
                area_total_caixa = w * h
                
                # Densidades
                roi_laranja = mask_laranja_limpa[y:y+h, x:x+w]
                pixels_laranja = cv2.countNonZero(roi_laranja)
                densidade_laranja = pixels_laranja / float(area_total_caixa)

                roi_branco = mask_branco_limpa[y:y+h, x:x+w]
                pixels_branco = cv2.countNonZero(roi_branco)
                densidade_total = (pixels_laranja + pixels_branco) / float(area_total_caixa)

                # Aspect Ratio (Largura / Altura). Cones < 1.0 (Mais altos que largos)
                aspect_ratio = float(w) / float(h)

                # Guardando os dados
                stats_densidade_laranja.append(densidade_laranja)
                stats_densidade_total.append(densidade_total)
                stats_aspect_ratio.append(aspect_ratio)
                stats_area_caixa.append(area_total_caixa)

        if (idx + 1) % 50 == 0:
            print(f"Processadas {idx + 1}/{len(caminhos)} imagens...")

    print(f"\nExtraídos dados de {len(stats_densidade_laranja)} blocos/caixas.")
    print("Gerando painel de gráficos...")

    # ==========================================
    # PLOTAGEM DOS HISTOGRAMAS (MATPLOTLIB)
    # ==========================================
    plt.style.use('ggplot') # Estilo mais bonito
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análise Estatística dos Candidatos (Cones/Blocos Laranjas)', fontsize=16, fontweight='bold')

    # Grafico 1: Densidade Laranja
    axs[0, 0].hist(stats_densidade_laranja, bins=50, color='orange', edgecolor='black', alpha=0.7)
    axs[0, 0].set_title('1. Densidade do Laranja')
    axs[0, 0].set_xlabel('Porcentagem da Caixa (0.0 a 1.0)')
    axs[0, 0].set_ylabel('Frequência (Qtd. de Caixas)')
    axs[0, 0].axvline(np.mean(stats_densidade_laranja), color='red', linestyle='dashed', linewidth=2, label='Média')
    axs[0, 0].legend()

    # Grafico 2: Densidade Total (Laranja + Branco)
    axs[0, 1].hist(stats_densidade_total, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0, 1].set_title('2. Densidade Total (Laranja + Branco)')
    axs[0, 1].set_xlabel('Porcentagem da Caixa (0.0 a 1.0)')
    axs[0, 1].set_ylabel('Frequência (Qtd. de Caixas)')
    axs[0, 1].axvline(np.mean(stats_densidade_total), color='blue', linestyle='dashed', linewidth=2, label='Média')
    axs[0, 1].legend()

    # Grafico 3: Aspect Ratio
    # Limitamos o eixo X a 3.0 porque valores maiores que isso são linhas esticadas irrelevantes
    razoes_filtradas = [r for r in stats_aspect_ratio if r <= 3.0] 
    axs[1, 0].hist(razoes_filtradas, bins=50, color='green', edgecolor='black', alpha=0.7)
    axs[1, 0].set_title('3. Razão de Aspecto (Largura / Altura)')
    axs[1, 0].set_xlabel('Razão (< 1 = Alto | > 1 = Largo)')
    axs[1, 0].set_ylabel('Frequência')
    axs[1, 0].axvline(1.0, color='black', linestyle='dotted', linewidth=2, label='Quadrado (1.0)')
    axs[1, 0].legend()

    # Grafico 4: Área da Caixa
    # Tiramos os 5% maiores valores para o gráfico não ficar esmagado por caixas gigantes
    if stats_area_caixa:
        area_percentil_95 = np.percentile(stats_area_caixa, 95)
        areas_filtradas = [a for a in stats_area_caixa if a <= area_percentil_95]
        axs[1, 1].hist(areas_filtradas, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axs[1, 1].set_title('4. Área da Bounding Box (Tamanho)')
        axs[1, 1].set_xlabel('Área em Pixels Quadrados')
        axs[1, 1].set_ylabel('Frequência')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Dá um espaço para o título principal
    
    # Salva o gráfico em PNG e abre na tela
    plt.savefig("00_ESTATISTICAS_TREINO.png", dpi=300)
    plt.show()

# --- Execução ---
pasta_treino = "images/train" 
analisar_dataset_treino(pasta_treino)