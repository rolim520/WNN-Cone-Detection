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

def extrair_candidatos_knn(caminho_img, pasta_saida, limiar_densidade=0.12, k_vizinhos=6):
    if not os.path.exists(pasta_saida): 
        os.makedirs(pasta_saida)

    img = cv2.imread(caminho_img)
    if img is None: 
        print(f"Erro ao carregar a imagem: {caminho_img}")
        return

    # 1. Máscara
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 87, 100]), np.array([21, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([177, 87, 100]), np.array([179, 255, 255]))
    mask_cor = cv2.bitwise_or(mask1, mask2)
    
    kernel_abertura = np.ones((3, 3), np.uint8)
    mask_limpa = cv2.morphologyEx(mask_cor, cv2.MORPH_OPEN, kernel_abertura, iterations=1)

    # 2. Encontrar blocos e PRÉ-CALCULAR pixels laranjas e centroides
    contornos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caixas_validas = []
    pixels_por_caixa = []
    centroides = []
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > 15:
            x, y, w, h = cv2.boundingRect(cnt)
            caixas_validas.append((x, y, w, h))
            
            roi = mask_limpa[y:y+h, x:x+w]
            pixels_por_caixa.append(cv2.countNonZero(roi))
            centroides.append(calcular_centroide((x, y, w, h)))

    num_caixas = len(caixas_validas)
    print(f"Encontrados {num_caixas} blocos laranjas base.")

    inicio_tempo = time.perf_counter()
    
    # 3. K-Vizinhos e Combinações
    k_real = min(k_vizinhos, num_caixas)
    combos_unicos = set()
    
    for i in range(num_caixas):
        distancias = []
        for j in range(num_caixas):
            dist = math.hypot(centroides[i][0] - centroides[j][0], centroides[i][1] - centroides[j][1])
            distancias.append((dist, j))
        
        distancias.sort()
        # Pega apenas os índices dos K vizinhos mais próximos
        vizinhos_k = [idx for dist, idx in distancias[:k_real]]
        
        # Faz as combinações (limitado a 4 elementos, como você definiu)
        limite_comb = min(len(vizinhos_k), 3)
        for tamanho in range(1, limite_comb + 1):
            for combo in combinations(vizinhos_k, tamanho):
                combos_unicos.add(tuple(sorted(combo)))

    # 4. Filtrar por Densidade Estrita
    candidatos_finais = []
    
    for combo_indices in combos_unicos:
        caixas_do_combo = [caixas_validas[idx] for idx in combo_indices]
        x_g, y_g, w_g, h_g = calcular_bounding_box_global(caixas_do_combo)
        
        # Ignora caixas inválidas ou puramente lineares
        if w_g <= 5 or h_g <= 5: 
            continue
            
        area_total_caixa = w_g * h_g
        
        # Soma estrita apenas dos pixels dos blocos combinados
        pixels_combo_estrito = sum([pixels_por_caixa[idx] for idx in combo_indices])
        densidade = pixels_combo_estrito / float(area_total_caixa)

        if densidade >= limiar_densidade:
            candidatos_finais.append((x_g, y_g, w_g, h_g, densidade))

    fim_tempo = time.perf_counter()
    tempo_logico_ms = (fim_tempo - inicio_tempo) * 1000

    print(f"Avaliadas {len(combos_unicos)} combinações únicas pelo método K-NN (K={k_real}).")
    print(f"Sobreviveram {len(candidatos_finais)} caixas ao Filtro de Densidade.")
    print(f"-> Tempo Lógico de Processamento: {tempo_logico_ms:.2f} ms")
    print("-" * 50)

    # 5. Salvar Recortes e Imagem de Visualização
    img_debug = img.copy()
    for idx, (x, y, w, h, densidade) in enumerate(candidatos_finais):
        recorte_cone = img[y:y+h, x:x+w]
        nome_arquivo = os.path.join(pasta_saida, f"candidato_{idx}_dens_{densidade:.2f}.png")
        cv2.imwrite(nome_arquivo, recorte_cone)
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(pasta_saida, "00_VISUALIZACAO_GERAL.png"), img_debug)
    print(f"Sucesso! Salvos em '{pasta_saida}'.")

# --- Execução ---
caminho_imagem = "images/train/cone (134).jpg" 
pasta_destino = "candidatos_wisard_baseline"

extrair_candidatos_knn(caminho_imagem, pasta_destino, limiar_densidade=0.20, k_vizinhos=6)