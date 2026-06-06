import cv2
import numpy as np
import os
import glob
import time
import math
import random
import json
from itertools import combinations
import wisardpkg as wp

# ==========================================================
# 1. PARÂMETROS E CONFIGURAÇÕES DO ARQUIVO JSON
# ==========================================================
ARQUIVO_CONFIG = "config.json"

def carregar_configuracoes():
    if not os.path.exists(ARQUIVO_CONFIG):
        raise FileNotFoundError(f"[!] O arquivo {ARQUIVO_CONFIG} não foi encontrado. Por favor, certifique-se de que o Complete Calibration gerou o JSON na mesma pasta.")
    
    with open(ARQUIVO_CONFIG, 'r') as f:
        print(f"[*] Parâmetros carregados estritamente de {ARQUIVO_CONFIG}")
        return json.load(f)

ESTADO = carregar_configuracoes()

# Parâmetros Fixos e de Controle
RESOLUCAO = 64
MODO = 'mascaras'
TUPLA = 16
IGN_ZERO = False
EXIBIR_TODAS_CAIXAS_WISARD = False

# ==========================================================
# 2. GEOMETRIA, OPENCV E DATA AUGMENTATION OTIMIZADOS
# ==========================================================
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

# NOVO: Gera as máscaras apenas UMA vez por imagem
def gerar_mascaras(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    m1 = cv2.inRange(hsv, np.array([ESTADO['h1_min'], ESTADO['s_min'], ESTADO['v_min']]), np.array([ESTADO['h1_max'], 255, 255]))
    m2 = cv2.inRange(hsv, np.array([ESTADO['h2_min'], ESTADO['s_min'], ESTADO['v_min']]), np.array([ESTADO['h2_max'], 255, 255]))
    mask_laranja = cv2.bitwise_or(m1, m2)
    
    if ESTADO['iter_abertura'] > 0:
        kernel_a = np.ones((ESTADO['k_abertura'], ESTADO['k_abertura']), np.uint8)
        mask_laranja = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, kernel_a, iterations=ESTADO['iter_abertura'])
        
    mb = cv2.inRange(hsv, np.array([ESTADO['h_b_min'], ESTADO['s_b_min'], ESTADO['v_b_min']]), np.array([ESTADO['h_b_max'], ESTADO['s_b_max'], ESTADO['v_b_max']]))
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, np.ones((ESTADO['k_abert_b'], ESTADO['k_abert_b']), np.uint8), iterations=ESTADO['iter_abert_b'])
    mask_branco = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, np.ones((ESTADO['k_fech_b'], ESTADO['k_fech_b']), np.uint8), iterations=ESTADO['iter_fech_b'])
    
    return mask_laranja, mask_branco

# OTIMIZADO: Aplica as rotações diretamente nas máscaras binárias
def augmentar_recorte_mascaras(m_lar, m_br):
    aug_crops = [(m_lar, m_br), (cv2.flip(m_lar, 1), cv2.flip(m_br, 1))]
    h, w = m_lar.shape[:2]
    centro = (w // 2, h // 2)
    for angulo in [-3, -2, -1, 1, 2, 3]:
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        # Rotação de imagens binárias deve usar interpolacao nearest para manter as bordas limpas
        rot_lar = cv2.warpAffine(m_lar, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        rot_br = cv2.warpAffine(m_br, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        aug_crops.extend([(rot_lar, rot_br), (cv2.flip(rot_lar, 1), cv2.flip(rot_br, 1))])
    return aug_crops

# OTIMIZADO: Recebe as máscaras prontas em vez da imagem BGR
def extrair_candidatos_multiplos(mask_laranja, mask_branco, img_w, img_h):
    contornos, _ = cv2.findContours(mask_laranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas, pixels, centroides = [], [], []
    lim_area = max(ESTADO['area_minima'], (max([cv2.contourArea(c) for c in contornos]) if contornos else 0) * ESTADO['fator_proporcao'])
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > lim_area:
            x, y, w, h = cv2.boundingRect(cnt)
            caixas.append((x, y, w, h))
            pixels.append(cv2.countNonZero(mask_laranja[y:y+h, x:x+w]))
            centroides.append((x + w/2.0, y + h/2.0))
            
    bboxes_dna = {}
    for i in range(len(caixas)):
        r = max(caixas[i][2], caixas[i][3]) * ESTADO['fator_raio']
        viz_k = [j for j in range(len(caixas)) if math.hypot(centroides[i][0]-centroides[j][0], centroides[i][1]-centroides[j][1]) <= r][:ESTADO['k_vizinhos']]
        for tam in range(1, min(len(viz_k), 4) + 1):
            for indices in combinations(viz_k, tam):
                c_combo = [caixas[idx] for idx in indices]
                xg, yg = min([b[0] for b in c_combo]), min([b[1] for b in c_combo])
                wg, hg = max([b[0]+b[2] for b in c_combo]) - xg, max([b[1]+b[3] for b in c_combo]) - yg
                if wg <= 5 or hg <= 5: continue
                pix_lar = sum([pixels[idx] for idx in indices])
                pix_br = cv2.countNonZero(mask_branco[yg:yg+hg, xg:xg+wg])
                d_lar, d_tot = pix_lar / float(wg * hg), (pix_lar + pix_br) / float(wg * hg)
                if ESTADO['limiar_laranja_min'] <= d_lar <= ESTADO['limiar_laranja_max'] and d_tot >= ESTADO['limiar_total']:
                    box = (xg, yg, wg, hg)
                    if box not in bboxes_dna or len(indices) > len(bboxes_dna.get(box, set())):
                        bboxes_dna[box] = set(indices)

    finais_multiplos = []
    caixas_base = []
    for c, dna in sorted(list(bboxes_dna.items()), key=lambda x: x[0][2]*x[0][3], reverse=True):
        if not any(calcular_iou(c, a[0]) > 0.8 and dna.issubset(a[1]) for a in caixas_base):
            caixas_base.append((c, dna))
            
    for c, _ in caixas_base:
        xg, yg, wg, hg = c
        finais_multiplos.append(c) # Âncora Justa
        
        # Âncora Base Preta
        xn2 = max(0, xg - int(wg * 0.20))
        wn2 = min(img_w - xn2, wg + int(wg * 0.40))
        yn2 = max(0, yg - int(hg * 0.00))
        hn2 = min(img_h - yn2, hg + int(hg * 0.1)) 
        finais_multiplos.append((xn2, yn2, wn2, hn2))
            
    return finais_multiplos

# OTIMIZADO: Não converte cores. Apenas redimensiona a máscara de forma rápida e segura para a memória.
def binarizar_mascaras(m_lar, m_br):
    ml_res = cv2.resize(m_lar, (RESOLUCAO, RESOLUCAO), interpolation=cv2.INTER_NEAREST)
    mb_res = cv2.resize(m_br, (RESOLUCAO, RESOLUCAO), interpolation=cv2.INTER_NEAREST)
    
    # Casting super-rápido para listas de 0s e 1s
    lista_laranja = np.where(ml_res.ravel() > 0, 1, 0).tolist()
    lista_branco = np.where(mb_res.ravel() > 0, 1, 0).tolist()
    return lista_laranja + lista_branco

# ==========================================================
# 3. EXECUÇÃO PRINCIPAL DO BENCHMARK
# ==========================================================
if __name__ == "__main__":
    print("="*60)
    print(" BENCHMARK FINAL: PIPELINE SEM PESOS vs TRACON (YOLOv5)")
    print("="*60)
    
    print("\n[Fase 1] Carregando e Augmentando Dados de Treino/Validação...")
    cones_X, fundos_X = [], []
    for arq in glob.glob("images/train/*.*") + glob.glob("images/val/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        
        # Gera máscaras uma vez por imagem de treino
        mask_lar, mask_br = gerar_mascaras(img)
        
        for gab in gabaritos:
            x, y, w, h = gab
            x, y = max(0, int(x)), max(0, int(y))
            crop_lar = mask_lar[y:y+h, x:x+w]
            crop_br = mask_br[y:y+h, x:x+w]
            
            if crop_lar.size > 0:
                for v_lar, v_br in augmentar_recorte_mascaras(crop_lar, crop_br):
                    cones_X.append(binarizar_mascaras(v_lar, v_br))
        
        for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img):
            x, y, w, h = cand
            x, y = max(0, int(x)), max(0, int(y))
            crop_lar = mask_lar[y:y+h, x:x+w]
            crop_br = mask_br[y:y+h, x:x+w]
            
            if crop_lar.size == 0: continue
            
            iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
            if iou <= ESTADO['iou_negativo']:
                fundos_X.append(binarizar_mascaras(crop_lar, crop_br))
            elif iou >= ESTADO['iou_positivo']:
                cones_X.append(binarizar_mascaras(crop_lar, crop_br))
            
    random.seed(42)
    random.shuffle(fundos_X)
    fundos_X = fundos_X[:len(cones_X)]
    
    X_train = cones_X + fundos_X
    y_train = ['cone'] * len(cones_X) + ['nao_cone'] * len(fundos_X)
    
    print(f" -> Extração Concluída: {len(cones_X)} Cones Perfeitos vs {len(fundos_X)} Não-Cones.")
    
    print(f"\n[Fase 2] Treinando WiSARD (Res={RESOLUCAO}, Tupla={TUPLA}, IgnZero={IGN_ZERO})...")
    modelo = wp.Wisard(TUPLA, ignoreZero=IGN_ZERO)
    dataset_treino = wp.DataSet(X_train, y_train)
    modelo.train(dataset_treino)

    print("\n[Fase 3] Rodando Inferência no Conjunto de Teste End-to-End...")
    pasta_saida = "resultados_finais"
    os.makedirs(pasta_saida, exist_ok=True)
    
    tempos = []
    tp_oficial, tp_relaxado, fp, tot_gabaritos = 0, 0, 0, 0
    
    for arq in glob.glob("images/test/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        tot_gabaritos += len(gabaritos)
        
        # INÍCIO DO CRONÔMETRO
        t0 = time.perf_counter()
        
        mask_lar, mask_br = gerar_mascaras(img)
        candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img)
        caixas_filtradas = []
        
        if candidatos:
            recortes = []
            candidatos_validos = [] # Mapeia os candidatos que não estouraram as bordas da imagem
            
            for (x, y, w, h) in candidatos:
                x, y = max(0, int(x)), max(0, int(y))
                crop_lar = mask_lar[y:y+h, x:x+w]
                crop_br = mask_br[y:y+h, x:x+w]
                
                if crop_lar.size > 0:
                    recortes.append(binarizar_mascaras(crop_lar, crop_br))
                    candidatos_validos.append((x, y, w, h))
                    
            if recortes:
                preds = modelo.classify(wp.DataSet(recortes))
                
                aprovadas_raw = [candidatos_validos[i] for i, p in enumerate(preds) if p == 'cone']
                for box_raw in sorted(aprovadas_raw, key=lambda b: b[2]*b[3], reverse=True):
                    if not any(calcular_iou(box_raw, b_apr) > 0.6 for b_apr in caixas_filtradas):
                        caixas_filtradas.append(box_raw)

        # FIM DO CRONÔMETRO
        tempos.append((time.perf_counter() - t0) * 1000)
        
        gab_detectados_ofc = set()
        gab_detectados_rel = set()
        
        for box in caixas_filtradas:
            melhor_iou, idx_gab = 0.0, -1
            for i, gab in enumerate(gabaritos):
                iou = calcular_iou(box, gab)
                if iou > melhor_iou:
                    melhor_iou = iou
                    idx_gab = i
            
            is_tp = False
            if melhor_iou >= 0.50:
                if idx_gab not in gab_detectados_ofc:
                    tp_oficial += 1
                    gab_detectados_ofc.add(idx_gab)
                is_tp = True
                
            if melhor_iou >= 0.30 and idx_gab not in gab_detectados_rel:
                tp_relaxado += 1
                gab_detectados_rel.add(idx_gab)
                
            if not is_tp:
                fp += 1
                
        # ==========================================================
        # ETAPA VISUAL
        # ==========================================================
        if EXIBIR_TODAS_CAIXAS_WISARD:
            if candidatos and recortes:
                for idx, box in enumerate(candidatos_validos):
                    x, y, w, h = box
                    if preds[idx] == 'cone':
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        else:
            for box in caixas_filtradas:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Cone", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(pasta_saida, os.path.basename(arq)), img)

    # Cálculos PASCAL VOC
    precisao = (tp_oficial / (tp_oficial + fp)) * 100 if (tp_oficial + fp) > 0 else 0
    recall_oficial = (tp_oficial / tot_gabaritos) * 100 if tot_gabaritos > 0 else 0
    f1_oficial = 2 * (precisao * recall_oficial) / (precisao + recall_oficial) if (precisao + recall_oficial) > 0 else 0
    
    recall_relax = (tp_relaxado / tot_gabaritos) * 100 if tot_gabaritos > 0 else 0

    print("\n" + "="*60)
    print(" RELATÓRIO CIENTÍFICO FINAL: PIPELINE SEM PESOS vs YOLOv5")
    print("="*60)
    print(f" Total de Cones no Teste: {tot_gabaritos}")
    print(f" Falsos Positivos da Rede: {fp}")
    print("-" * 60)
    print(" AVALIAÇÃO OFICIAL (PASCAL VOC - IoU > 0.50)")
    print(f" -> Precisão : {precisao:.2f}%")
    print(f" -> Recall   : {recall_oficial:.2f}% (Teto era 61.02%)")
    print(f" -> F1-Score : {f1_oficial:.2f}%")
    print("-" * 60)
    print(" AVALIAÇÃO RELAXADA (Aplicação em Robótica - IoU > 0.30)")
    print(f" -> Recall Relaxado: {recall_relax:.2f}%")
    print("-" * 60)
    print(" MÉTRICA DE PERFORMANCE (Velocidade)")
    print(f" -> YOLOv5 (Artigo): 65.0 ms (Placa de Vídeo Tesla K80)")
    print(f" -> WiSARD + OpenCV: {np.mean(tempos):.2f} ms (Processador Comum)")
    print("="*60)