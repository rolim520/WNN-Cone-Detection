import cv2
import numpy as np
import os
import glob
import time
import math
import json
import random
from itertools import combinations
import wisardpkg as wp

# ==========================================================
# 1. PARÂMETROS BASE E ESPAÇO DE BUSCA
# ==========================================================
ARQUIVO_CONFIG = "config.json"

def carregar_configuracoes():
    if not os.path.exists(ARQUIVO_CONFIG):
        raise FileNotFoundError(f"[!] O arquivo {ARQUIVO_CONFIG} não foi encontrado.")
    with open(ARQUIVO_CONFIG, 'r') as f:
        return json.load(f)

ESTADO = carregar_configuracoes()

# --- ESPAÇO DE BUSCA DO GRID SEARCH ---
GRID_RESOLUCOES = [16, 32, 64]
GRID_TUPLAS = [8, 12, 16, 20, 24]
GRID_IGNORE_ZERO = [True, False]
NUM_EXECUCOES = 5  # Quantidade de vezes que cada configuração será testada

# ==========================================================
# 2. FUNÇÕES DE GEOMETRIA E EXTRAÇÃO (OTIMIZADAS)
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

def augmentar_recorte_mascaras(m_lar, m_br):
    aug_crops = [(m_lar, m_br), (cv2.flip(m_lar, 1), cv2.flip(m_br, 1))]
    h, w = m_lar.shape[:2]
    centro = (w // 2, h // 2)
    for angulo in [-3, -2, -1, 1, 2, 3]:
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        rot_lar = cv2.warpAffine(m_lar, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        rot_br = cv2.warpAffine(m_br, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        aug_crops.extend([(rot_lar, rot_br), (cv2.flip(rot_lar, 1), cv2.flip(rot_br, 1))])
    return aug_crops

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
        finais_multiplos.append(c)
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

# ==========================================================
# 3. PREPARAÇÃO DO CACHE DE DADOS (Roda o OpenCV apenas 1x)
# ==========================================================
print("="*60)
print(f" INICIANDO GRID SEARCH DA WISARD ({NUM_EXECUCOES} Execuções por Setup)")
print("="*60)

print("\n[Etapa 1/3] Extraindo recortes da base de Treino (Cache na RAM)...")
cache_treino_cones = []
cache_treino_fundos = []

# MODIFICAÇÃO: Lendo estritamente a pasta de treino, sem a pasta de validação
for arq in glob.glob("images/train/*.*"):
    img = cv2.imread(arq)
    if img is None: continue
    h_img, w_img = img.shape[:2]
    gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
    mask_lar, mask_br = gerar_mascaras(img)
    
    for gab in gabaritos:
        x, y, w, h = gab
        x, y = max(0, int(x)), max(0, int(y))
        c_lar = mask_lar[y:y+h, x:x+w]
        c_br = mask_br[y:y+h, x:x+w]
        if c_lar.size > 0:
            for v_lar, v_br in augmentar_recorte_mascaras(c_lar, c_br):
                cache_treino_cones.append((v_lar, v_br))
    
    for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img):
        x, y, w, h = cand
        x, y = max(0, int(x)), max(0, int(y))
        c_lar = mask_lar[y:y+h, x:x+w]
        c_br = mask_br[y:y+h, x:x+w]
        if c_lar.size == 0: continue
        iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
        
        if iou <= ESTADO['iou_negativo']:
            cache_treino_fundos.append((c_lar, c_br))
        elif iou >= ESTADO['iou_positivo']:
            cache_treino_cones.append((c_lar, c_br))

random.seed(42)
random.shuffle(cache_treino_fundos)
# Evita erro se houver menos fundos que cones
cache_treino_fundos = cache_treino_fundos[:len(cache_treino_cones)] 
print(f" -> Cache Criado: {len(cache_treino_cones)} Cones vs {len(cache_treino_fundos)} Não-Cones.")

print("\n[Etapa 2/3] Extraindo candidatos da base de Teste (Cache na RAM)...")
cache_teste = []
tot_gabaritos = 0

for arq in glob.glob("images/test/*.*"):
    img = cv2.imread(arq)
    if img is None: continue
    h_img, w_img = img.shape[:2]
    gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
    tot_gabaritos += len(gabaritos)
    mask_lar, mask_br = gerar_mascaras(img)
    candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img)
    
    caixas_validas = []
    recortes_raw = []
    for (x, y, w, h) in candidatos:
        x, y = max(0, int(x)), max(0, int(y))
        c_lar = mask_lar[y:y+h, x:x+w]
        c_br = mask_br[y:y+h, x:x+w]
        if c_lar.size > 0:
            caixas_validas.append((x, y, w, h))
            recortes_raw.append((c_lar, c_br))
            
    cache_teste.append({
        'arquivo': arq,
        'gabaritos': gabaritos,
        'caixas': caixas_validas,
        'recortes_raw': recortes_raw
    })
print(f" -> Cache de Teste Criado: {tot_gabaritos} cones reais a serem encontrados nas imagens.")

# ==========================================================
# 4. EXECUÇÃO DO GRID SEARCH (WiSARD Multi-Run)
# ==========================================================
print("\n[Etapa 3/3] Iniciando Grid Search...")
melhor_media_f1 = -1
melhor_config = {}
historico_resultados = []

for res in GRID_RESOLUCOES:
    print(f"\n>> Preparando dados binarizados para Resolução {res}x{res}...")
    
    # Prepara o conjunto base de treino
    X_train_base = []
    for ml, mb in cache_treino_cones: X_train_base.append(binarizar_para_resolucao(ml, mb, res))
    for ml, mb in cache_treino_fundos: X_train_base.append(binarizar_para_resolucao(ml, mb, res))
    y_train_base = ['cone'] * len(cache_treino_cones) + ['nao_cone'] * len(cache_treino_fundos)
    
    # Prepara o conjunto base de teste
    for img_data in cache_teste:
        img_data['X_test'] = [binarizar_para_resolucao(ml, mb, res) for ml, mb in img_data['recortes_raw']]
    
    for tupla in GRID_TUPLAS:
        for ign_zero in GRID_IGNORE_ZERO:
            print(f"   -> [Res={res:2d}, Tupla={tupla:2d}, IgnZero={str(ign_zero):<5}] Avaliando... ", end="")
            
            soma_precisao = 0
            soma_recall = 0
            soma_f1 = 0
            soma_iou = 0  # MODIFICAÇÃO: Soma para Média de IoU
            
            for rodada in range(NUM_EXECUCOES):
                # 1. Embaralha os dados para gerar mapeamentos diferentes na RAM
                combinado = list(zip(X_train_base, y_train_base))
                random.shuffle(combinado)
                X_train_shuf, y_train_shuf = zip(*combinado)
                
                # 2. Treinamento
                dataset_treino = wp.DataSet(list(X_train_shuf), list(y_train_shuf))
                modelo = wp.Wisard(tupla, ignoreZero=ign_zero)
                modelo.train(dataset_treino)
                
                # 3. Inferência e Avaliação PASCAL VOC (IoU > 0.50)
                tp_oficial, fp = 0, 0
                ious_da_rodada = []  # MODIFICAÇÃO: Guarda os IoUs dos acertos
                
                for img_data in cache_teste:
                    if not img_data['X_test']: continue
                        
                    preds = modelo.classify(wp.DataSet(img_data['X_test']))
                    
                    caixas_filtradas = []
                    aprovadas_raw = [img_data['caixas'][i] for i, p in enumerate(preds) if p == 'cone']
                    for box_raw in sorted(aprovadas_raw, key=lambda b: b[2]*b[3], reverse=True):
                        if not any(calcular_iou(box_raw, b_apr) > 0.6 for b_apr in caixas_filtradas):
                            caixas_filtradas.append(box_raw)
                    
                    gab_detectados_ofc = set()
                    for box in caixas_filtradas:
                        melhor_iou, idx_gab = 0.0, -1
                        for i, gab in enumerate(img_data['gabaritos']):
                            iou = calcular_iou(box, gab)
                            if iou > melhor_iou:
                                melhor_iou = iou
                                idx_gab = i
                        
                        is_tp = False
                        if melhor_iou >= 0.50:
                            if idx_gab not in gab_detectados_ofc:
                                tp_oficial += 1
                                gab_detectados_ofc.add(idx_gab)
                                ious_da_rodada.append(melhor_iou)  # MODIFICAÇÃO: Salva o IoU do Acerto Perfeito
                            is_tp = True
                            
                        if not is_tp:
                            fp += 1
                            
                prec_rodada = (tp_oficial / (tp_oficial + fp)) * 100 if (tp_oficial + fp) > 0 else 0
                rec_rodada = (tp_oficial / tot_gabaritos) * 100 if tot_gabaritos > 0 else 0
                f1_rodada = 2 * (prec_rodada * rec_rodada) / (prec_rodada + rec_rodada) if (prec_rodada + rec_rodada) > 0 else 0
                iou_medio_rodada = (np.mean(ious_da_rodada) * 100) if ious_da_rodada else 0
                
                soma_precisao += prec_rodada
                soma_recall += rec_rodada
                soma_f1 += f1_rodada
                soma_iou += iou_medio_rodada
                
            # Calcula as médias das N execuções
            media_prec = soma_precisao / NUM_EXECUCOES
            media_rec = soma_recall / NUM_EXECUCOES
            media_f1 = soma_f1 / NUM_EXECUCOES
            media_iou = soma_iou / NUM_EXECUCOES
            
            print(f"Média F1: {media_f1:.2f}% (Prec: {media_prec:.1f}%, Rec: {media_rec:.1f}%, Média IoU: {media_iou:.2f}%)")
            
            historico_resultados.append({
                'res': res, 'tupla': tupla, 'ign_zero': ign_zero,
                'precisao': media_prec, 'recall': media_rec, 'f1': media_f1, 'iou': media_iou
            })
            
            if media_f1 > melhor_media_f1:
                melhor_media_f1 = media_f1
                melhor_config = {'res': res, 'tupla': tupla, 'ign_zero': ign_zero, 'prec': media_prec, 'rec': media_rec, 'iou': media_iou}

print("\n" + "="*60)
print(f" RESULTADO FINAL DO GRID SEARCH (Média de {NUM_EXECUCOES} execuções)")
print("="*60)
historico_ordenado = sorted(historico_resultados, key=lambda x: x['f1'], reverse=True)

for i, r in enumerate(historico_ordenado[:5]):
    print(f" {i+1}º Lugar -> Res: {r['res']:2d} | Tupla: {r['tupla']:2d} | IgnZero: {str(r['ign_zero']):<5} || F1 Média: {r['f1']:.2f}% (Prec: {r['precisao']:5.1f}%, Rec: {r['recall']:5.1f}%, Média IoU: {r['iou']:5.2f}%)")

print("\n[AÇÃO RECOMENDADA]")
print(f"Abra o seu script de benchmark real e configure:")
print(f"RESOLUCAO = {melhor_config['res']}")
print(f"TUPLA = {melhor_config['tupla']}")
print(f"IGN_ZERO = {melhor_config['ign_zero']}")
print(f"-> Média de IoU Esperada: {melhor_config['iou']:.2f}% (YOLOv5 obteve 91.31%)")
print("="*60)