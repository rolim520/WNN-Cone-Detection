import cv2
import numpy as np
import os
import math
import json
from itertools import combinations

ARQUIVO_CONFIG = "config.json"

def carregar_configuracoes():
    """Carrega o JSON de configuração e injeta os parâmetros mais novos caso não existam."""
    if not os.path.exists(ARQUIVO_CONFIG):
        raise FileNotFoundError(f"[!] O arquivo {ARQUIVO_CONFIG} não foi encontrado.")
    
    with open(ARQUIVO_CONFIG, 'r') as f:
        config = json.load(f)
        
        # Retrocompatibilidade com configurações antigas
        if 'razao_aspecto_max' not in config: config['razao_aspecto_max'] = 4.0
        if 'max_regioes_internas' not in config: config['max_regioes_internas'] = 5
            
        return config

def ler_gabarito_yolo(img_path, img_w, img_h):
    base_nome = os.path.splitext(os.path.basename(img_path))[0]
    dir_label = os.path.dirname(img_path).replace('images', 'labels')
    txt_path = os.path.join(dir_label, base_nome + '.txt')
    caixas = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) >= 5:
                    _, x_c, y_c, w, h = map(float, partes[:5])
                    x_px, y_px = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                    caixas.append((x_px, y_px, int(w * img_w), int(h * img_h)))
    return caixas

def calcular_iou(boxA, boxB):
    x_inter_min = max(boxA[0], boxB[0])
    y_inter_min = max(boxA[1], boxB[1])
    x_inter_max = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    y_inter_max = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min: return 0.0
    area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    return area_inter / float((boxA[2]*boxA[3]) + (boxB[2]*boxB[3]) - area_inter)

def calcular_centroide(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

def calcular_bounding_box_global(caixas):
    x_min = min([box[0] for box in caixas])
    y_min = min([box[1] for box in caixas])
    x_max = max([box[0] + box[2] for box in caixas])
    y_max = max([box[1] + box[3] for box in caixas])
    return x_min, y_min, x_max - x_min, y_max - y_min

def gerar_mascaras(img_bgr, config):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([config['h1_min'], config['s_min'], config['v_min']]), np.array([config['h1_max'], 255, 255]))
    m2 = cv2.inRange(hsv, np.array([config['h2_min'], config['s_min'], config['v_min']]), np.array([config['h2_max'], 255, 255]))
    mask_laranja = cv2.bitwise_or(m1, m2)
    
    k_abert = config['k_abertura']
    if k_abert % 2 == 0: k_abert += 1 
    if config['iter_abertura'] > 0 and k_abert > 0:
        kernel_a = np.ones((k_abert, k_abert), np.uint8)
        mask_laranja = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, kernel_a, iterations=config['iter_abertura'])
        
    mb = cv2.inRange(hsv, np.array([config['h_b_min'], config['s_b_min'], config['v_b_min']]), np.array([config['h_b_max'], config['s_b_max'], config['v_b_max']]))
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, np.ones((config['k_abert_b'], config['k_abert_b']), np.uint8), iterations=config['iter_abert_b'])
    mask_branco = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((config['k_fech_b'], config['k_fech_b']), np.uint8), iterations=config['iter_fech_b'])
    
    return mask_laranja, mask_branco

def augmentar_recorte_mascaras(m_lar, m_br):
    aug_crops = [(m_lar, m_br), (cv2.flip(m_lar, 1), cv2.flip(m_br, 1))]
    h, w = m_lar.shape[:2]
    centro = (w // 2, h // 2)
    for angulo in [-5,-4,-3, -2, -1, 1, 2, 3,4,5]:
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        rot_lar = cv2.warpAffine(m_lar, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        rot_br = cv2.warpAffine(m_br, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        aug_crops.extend([(rot_lar, rot_br), (cv2.flip(rot_lar, 1), cv2.flip(rot_br, 1))])
    return aug_crops

def extrair_candidatos_multiplos(mask_laranja_limpa, mask_branco_limpa, img_w, img_h, config):
    contornos, _ = cv2.findContours(mask_laranja_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas_validas, pixels_por_caixa, centroides = [], [], []
    
    areas = [cv2.contourArea(cnt) for cnt in contornos]
    area_maxima_img = max(areas) if areas else 0
    limite_area_final = max(config['area_minima'], area_maxima_img * config['fator_proporcao'])
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > limite_area_final:
            x, y, w, h = cv2.boundingRect(cnt)
            caixas_validas.append((x, y, w, h))
            pixels_por_caixa.append(cv2.countNonZero(mask_laranja_limpa[y:y+h, x:x+w]))
            centroides.append(calcular_centroide((x, y, w, h)))
            
    num_caixas = len(caixas_validas)
    combos_unicos = set()
    
    for i in range(num_caixas):
        distancias_validas = []
        raio_dinamico_i = max(caixas_validas[i][2], caixas_validas[i][3]) * config['fator_raio']
        
        for j in range(num_caixas):
            dist = math.hypot(centroides[i][0] - centroides[j][0], centroides[i][1] - centroides[j][1])
            if dist <= raio_dinamico_i:
                distancias_validas.append((dist, j))
        
        distancias_validas.sort()
        vizinhos_k = [idx for dist, idx in distancias_validas[:config['k_vizinhos']]]
        
        for tamanho in range(1, min(len(vizinhos_k), 4) + 1):
            for combo in combinations(vizinhos_k, tamanho):
                combos_unicos.add(tuple(sorted(combo)))

    caixas_base = []
    
    # Aplica as regras de Densidade, Geometria e Overlap
    for combo_indices in combos_unicos:
        caixas_do_combo = [caixas_validas[idx] for idx in combo_indices]
        x_g, y_g, w_g, h_g = calcular_bounding_box_global(caixas_do_combo)
        
        if w_g <= 5 or h_g <= 5: continue
            
        if (max(w_g, h_g) / float(min(w_g, h_g))) > config['razao_aspecto_max']: continue
            
        regioes_contidas = sum(1 for cx, cy in centroides if x_g <= cx <= (x_g + w_g) and y_g <= cy <= (y_g + h_g))
        if regioes_contidas > config['max_regioes_internas']: continue

        area_total_caixa = w_g * h_g
        pix_lar = sum([pixels_por_caixa[idx] for idx in combo_indices])
        pix_br = cv2.countNonZero(mask_branco_limpa[y_g:y_g+h_g, x_g:x_g+w_g])
        d_lar = pix_lar / float(area_total_caixa)
        d_tot = (pix_lar + pix_br) / float(area_total_caixa)

        if config['limiar_laranja_min'] <= d_lar <= config['limiar_laranja_max'] and d_tot >= config['limiar_total']:
            bbox = (x_g, y_g, w_g, h_g)
            qtd_elementos = len(combo_indices)
            
            # Adiciona apenas se não existe ou se é um agrupamento mais rico
            encontrou_melhor = False
            for c_b, n_elem in caixas_base:
                if c_b == bbox:
                    encontrou_melhor = True
                    if qtd_elementos > n_elem:
                        caixas_base.remove((c_b, n_elem))
                        caixas_base.append((bbox, qtd_elementos))
                    break
            if not encontrou_melhor:
                caixas_base.append((bbox, qtd_elementos))

    # Filtro NMS simplificado para limpar sobreposições gritantes de extração
    bboxes_finais_limpas = []
    caixas_base.sort(key=lambda x: x[0][2]*x[0][3], reverse=True)
    for (bbox, _) in caixas_base:
        if not any(calcular_iou(bbox, b_apr) > 0.8 for b_apr in bboxes_finais_limpas):
            bboxes_finais_limpas.append(bbox)

    finais_multiplos = []
    for c in bboxes_finais_limpas:
        xg, yg, wg, hg = c
        finais_multiplos.append(c) # Âncora Justa
        # Âncora Base Preta
        xn2 = max(0, xg - int(wg * 0.20))
        wn2 = min(img_w - xn2, wg + int(wg * 0.40))
        yn2 = max(0, yg - int(hg * 0.00))
        hn2 = min(img_h - yn2, hg + int(hg * 0.1)) 
        finais_multiplos.append((xn2, yn2, wn2, hn2))
            
    return finais_multiplos

def binarizar_para_resolucao(m_lar, m_br, res):
    ml_res = cv2.resize(m_lar, (res, res), interpolation=cv2.INTER_NEAREST)
    mb_res = cv2.resize(m_br, (res, res), interpolation=cv2.INTER_NEAREST)
    return np.where(ml_res.ravel() > 0, 1, 0).tolist() + np.where(mb_res.ravel() > 0, 1, 0).tolist()

def calcular_iom(boxA, boxB):
    """
    Calcula o Intersection over Minimum (IoM).
    Perfeito para deletar caixas pequenas contidas dentro de caixas maiores.
    """
    x_inter_min = max(boxA[0], boxB[0])
    y_inter_min = max(boxA[1], boxB[1])
    x_inter_max = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    y_inter_max = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min: 
        return 0.0
        
    area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    
    return area_inter / float(min(areaA, areaB))