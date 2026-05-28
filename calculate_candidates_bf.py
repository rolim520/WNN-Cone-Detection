import cv2
import numpy as np
import os
import time
from itertools import combinations

def calcular_bounding_box_global(caixas):
    """Calcula a Bounding Box que engloba um conjunto de sub-caixas."""
    x_min = min([box[0] for box in caixas])
    y_min = min([box[1] for box in caixas])
    x_max = max([box[0] + box[2] for box in caixas])
    y_max = max([box[1] + box[3] for box in caixas])
    return x_min, y_min, x_max - x_min, y_max - y_min

def extrair_candidatos_combinacoes(caminho_img, pasta_saida, limiar_densidade=0.25):
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    img = cv2.imread(caminho_img)
    if img is None:
        print(f"Erro ao carregar a imagem: {caminho_img}")
        return

    # 1. Pré-processamento e Máscara
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv, np.array([0, 87, 100]), np.array([21, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([177, 87, 100]), np.array([179, 255, 255]))
    mask_cor = cv2.bitwise_or(mask1, mask2)

    kernel_abertura = np.ones((3, 3), np.uint8)
    mask_limpa = cv2.morphologyEx(mask_cor, cv2.MORPH_OPEN, kernel_abertura, iterations=1)

    # 2. Encontrar todos os blocos isolados
    contornos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caixas_validas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 15:
            caixas_validas.append(cv2.boundingRect(cnt))

    print(f"Encontrados {len(caixas_validas)} blocos laranjas base.")

    caixas_candidatas_unicas = set()
    limite_combinacoes = min(len(caixas_validas), 4) 
    
    # ==========================================
    # INÍCIO DO CRONÔMETRO LÓGICO
    # ==========================================
    inicio_tempo = time.perf_counter()
    
    # Passo A: Gerar combinações
    for tamanho_grupo in range(1, limite_combinacoes + 1):
        for combo in combinations(caixas_validas, tamanho_grupo):
            x, y, w, h = calcular_bounding_box_global(combo)
            
            if w > 5 and h > 5:
                caixas_candidatas_unicas.add((x, y, w, h))

    # Passo B: Filtrar por Densidade de Laranja
    candidatos_aprovados = [] # Lista apenas com as coordenadas (sem imagem ainda)
    
    for x, y, w, h in caixas_candidatas_unicas:
        area_total_caixa = w * h
        mascara_roi = mask_limpa[y:y+h, x:x+w]
        
        pixels_laranjas = cv2.countNonZero(mascara_roi)
        densidade = pixels_laranjas / float(area_total_caixa)

        if densidade >= limiar_densidade:
            # Salva apenas a tupla de coordenadas e a densidade para uso posterior
            candidatos_aprovados.append((x, y, w, h, densidade))

    # ==========================================
    # FIM DO CRONÔMETRO LÓGICO
    # ==========================================
    fim_tempo = time.perf_counter()
    tempo_logico_ms = (fim_tempo - inicio_tempo) * 1000

    print(f"Geradas {len(caixas_candidatas_unicas)} Bounding Boxes únicas (Combinações).")
    print(f"Sobreviveram {len(candidatos_aprovados)} caixas ao filtro de densidade.")
    print(f"-> Tempo Lógico de Processamento (Sem I/O de disco): {tempo_logico_ms:.2f} ms")
    print("-" * 50)

    # 4. Escrita no Disco (Fora do Cronômetro)
    img_debug = img.copy()
    for idx, (x, y, w, h, densidade) in enumerate(candidatos_aprovados):
        recorte_cone = img[y:y+h, x:x+w]
        
        nome_arquivo = os.path.join(pasta_saida, f"candidato_{idx}_dens_{densidade:.2f}.png")
        cv2.imwrite(nome_arquivo, recorte_cone)
        
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(pasta_saida, "00_VISUALIZACAO_GERAL.png"), img_debug)
    print(f"Sucesso! Recortes salvos em '{pasta_saida}'.")

# --- Execução ---
caminho_imagem = "images/train/cone (152).jpg"
pasta_destino = "candidatos_wisard_baseline"

extrair_candidatos_combinacoes(caminho_imagem, pasta_destino, limiar_densidade=0.20)