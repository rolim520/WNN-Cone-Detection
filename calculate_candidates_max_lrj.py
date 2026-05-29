import cv2
import numpy as np
import os
import time
import math
from itertools import combinations

def calcular_centroide(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

def calcular_bounding_box_global(caixas):
    x_min = min([box[0] for box in caixas])
    y_min = min([box[1] for box in caixas])
    x_max = max([box[0] + box[2] for box in caixas])
    y_max = max([box[1] + box[3] for box in caixas])
    return x_min, y_min, x_max - x_min, y_max - y_min

# ADICIONADO: limiar_laranja_max (Por padrão 1.0, ou seja, 100%)
def extrair_candidatos_knn_dinamico(caminho_img, pasta_saida, limiar_laranja_min=0.25, limiar_laranja_max=1.0, limiar_total=0.40, k_vizinhos=6, fator_raio=2.5):
    if not os.path.exists(pasta_saida): 
        os.makedirs(pasta_saida)

    img = cv2.imread(caminho_img)
    if img is None: 
        print(f"Erro ao carregar a imagem: {caminho_img}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 1A. MÁSCARA LARANJA ---
    mask1 = cv2.inRange(hsv, np.array([0, 87, 100]), np.array([21, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([177, 87, 100]), np.array([179, 255, 255]))
    mask_laranja = cv2.bitwise_or(mask1, mask2)
    
    kernel_abertura_laranja = np.ones((3, 3), np.uint8)
    mask_laranja_limpa = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, kernel_abertura_laranja, iterations=1)

    # --- 1B. MÁSCARA BRANCA ---
    limite_inf_branco = np.array([0, 0, 152])
    limite_sup_branco = np.array([179, 74, 255])
    mask_branco = cv2.inRange(hsv, limite_inf_branco, limite_sup_branco)
    
    kernel_abert_branco = np.ones((3, 3), np.uint8)
    mask_branco_limpa = cv2.morphologyEx(mask_branco, cv2.MORPH_OPEN, kernel_abert_branco, iterations=1)
    kernel_fech_branco = np.ones((3, 3), np.uint8)
    mask_branco_limpa = cv2.morphologyEx(mask_branco_limpa, cv2.MORPH_CLOSE, kernel_fech_branco, iterations=2)

    # --- 2. ENCONTRAR BLOCOS E PRÉ-CALCULAR ---
    contornos, _ = cv2.findContours(mask_laranja_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caixas_validas = []
    pixels_por_caixa = []
    centroides = []
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > 15:
            x, y, w, h = cv2.boundingRect(cnt)
            caixas_validas.append((x, y, w, h))
            
            roi = mask_laranja_limpa[y:y+h, x:x+w]
            pixels_por_caixa.append(cv2.countNonZero(roi))
            centroides.append(calcular_centroide((x, y, w, h)))

    num_caixas = len(caixas_validas)
    print(f"Encontrados {num_caixas} blocos laranjas base.")

    inicio_tempo = time.perf_counter()
    
    # --- 3. K-VIZINHOS COM RAIO DINÂMICO ---
    combos_unicos = set()
    
    for i in range(num_caixas):
        distancias_validas = []
        x_i, y_i, w_i, h_i = caixas_validas[i]
        
        raio_dinamico_i = max(w_i, h_i) * fator_raio
        
        for j in range(num_caixas):
            dist = math.hypot(centroides[i][0] - centroides[j][0], centroides[i][1] - centroides[j][1])
            if dist <= raio_dinamico_i:
                distancias_validas.append((dist, j))
        
        distancias_validas.sort()
        
        vizinhos_k = [idx for dist, idx in distancias_validas[:k_vizinhos]]
        
        limite_comb = min(len(vizinhos_k), 4)
        for tamanho in range(1, limite_comb + 1):
            for combo in combinations(vizinhos_k, tamanho):
                combos_unicos.add(tuple(sorted(combo)))

    # --- 4. FILTRO DE DENSIDADE DUPLA (COM LIMITE SUPERIOR) ---
    candidatos_finais = []
    
    for combo_indices in combos_unicos:
        caixas_do_combo = [caixas_validas[idx] for idx in combo_indices]
        x_g, y_g, w_g, h_g = calcular_bounding_box_global(caixas_do_combo)
        
        if w_g <= 5 or h_g <= 5: 
            continue
            
        area_total_caixa = w_g * h_g
        
        pixels_combo_estrito = sum([pixels_por_caixa[idx] for idx in combo_indices])
        densidade_laranja = pixels_combo_estrito / float(area_total_caixa)

        roi_branco = mask_branco_limpa[y_g:y_g+h_g, x_g:x_g+w_g]
        pixels_branco_total = cv2.countNonZero(roi_branco)
        
        densidade_total = (pixels_combo_estrito + pixels_branco_total) / float(area_total_caixa)

        # MUDANÇA AQUI: Verifica se a densidade laranja está entre o Mínimo e o Máximo
        if limiar_laranja_min <= densidade_laranja <= limiar_laranja_max and densidade_total >= limiar_total:
            candidatos_finais.append((x_g, y_g, w_g, h_g, densidade_laranja, densidade_total))

    fim_tempo = time.perf_counter()
    tempo_logico_ms = (fim_tempo - inicio_tempo) * 1000

    print(f"Avaliadas {len(combos_unicos)} combinações únicas (K-NN com Raio Dinâmico).")
    print(f"Sobreviveram {len(candidatos_finais)} caixas aos Filtros Laranja ({limiar_laranja_min} a {limiar_laranja_max}) e Total (>= {limiar_total}).")
    print(f"-> Tempo Lógico de Processamento: {tempo_logico_ms:.2f} ms")
    print("-" * 50)

    # --- 5. SALVAR RECORTES ---
    img_debug = img.copy()
    for idx, (x, y, w, h, d_laranja, d_total) in enumerate(candidatos_finais):
        recorte_cone = img[y:y+h, x:x+w]
        nome_arquivo = os.path.join(pasta_saida, f"cand_{idx}_L_{d_laranja:.2f}_T_{d_total:.2f}.png")
        cv2.imwrite(nome_arquivo, recorte_cone)
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(pasta_saida, "00_VISUALIZACAO_GERAL.png"), img_debug)
    print(f"Sucesso! Salvos em '{pasta_saida}'.")

# --- Execução ---
caminho_imagem = "images/train/cone (58).jpg" 
pasta_destino = "candidatos_wisard_baseline"

# Ajuste o limiar_laranja_max conforme necessário (ex: 0.85 para ignorar caixas com mais de 85% de laranja)
extrair_candidatos_knn_dinamico(
    caminho_imagem, 
    pasta_destino, 
    limiar_laranja_min=0.25,  #0.25
    limiar_laranja_max=0.70, 
    limiar_total=0.35,  #3.5
    k_vizinhos=6, 
    fator_raio=2.8 #2
)