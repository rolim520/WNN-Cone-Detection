import cv2
import numpy as np
import os
import glob
import time
import random
import wisardpkg as wp
from utils import *

# ==========================================================
# 1. PARÂMETROS BASE E ESPAÇO DE BUSCA
# ==========================================================
ESTADO = carregar_configuracoes()

GRID_MODOS = ['cor', 'canny', 'hibrido']
GRID_RESOLUCOES = [16, 32, 64]
GRID_TUPLAS = [8, 12, 16, 20, 24]
GRID_IGNORE_ZERO = [True, False]
NUM_EXECUCOES = 5
LIMIAR_AR_CONE = 1.25

# ==========================================================
# 2. PREPARAÇÃO DO CACHE DE DADOS (Roda o OpenCV apenas 1x)
# ==========================================================
print("="*60)
print(f" INICIANDO GRID SEARCH DA WISARD ({NUM_EXECUCOES} Execuções por Setup)")
print("="*60)

print("\n[Etapa 1/3] Extraindo recortes da base de Treino (Cache na RAM)...")
cache_treino_cones = []
cache_treino_fundos = []

for arq in glob.glob("images/train/*.*"):
    img = cv2.imread(arq)
    if img is None: continue
    h_img, w_img = img.shape[:2]
    gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
    
    mask_lar, mask_br = gerar_mascaras(img, ESTADO)
    mask_canny = gerar_canny(img, ESTADO) # Usando a função unificada
    
    for gab in gabaritos:
        x, y, w, h = gab
        x, y = max(0, int(x)), max(0, int(y))
        c_lar, c_br, c_canny = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w], mask_canny[y:y+h, x:x+w]
        
        if c_lar.size > 0:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            for v_lar, v_br, v_canny in augmentar_recorte_mascaras(c_lar, c_br, c_canny):
                cache_treino_cones.append((v_lar, v_br, v_canny))
    
    for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO):
        x, y, w, h = cand
        x, y = max(0, int(x)), max(0, int(y))
        c_lar, c_br, c_canny = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w], mask_canny[y:y+h, x:x+w]
        if c_lar.size == 0: continue
        
        iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
        
        if iou <= ESTADO['iou_negativo']:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            cache_treino_fundos.append((c_lar, c_br, c_canny))
        elif iou >= ESTADO['iou_positivo']:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            cache_treino_cones.append((c_lar, c_br, c_canny))

random.seed(42)
random.shuffle(cache_treino_fundos)
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
    
    mask_lar, mask_br = gerar_mascaras(img, ESTADO)
    mask_canny = gerar_canny(img, ESTADO)
    candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO)
    
    caixas_validas = []
    recortes_raw = []
    for (x, y, w, h) in candidatos:
        x, y = max(0, int(x)), max(0, int(y))
        c_lar, c_br, c_canny = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w], mask_canny[y:y+h, x:x+w]
        if c_lar.size > 0:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            caixas_validas.append((x, y, w, h))
            recortes_raw.append((c_lar, c_br, c_canny))
            
    cache_teste.append({
        'arquivo': arq,
        'gabaritos': gabaritos,
        'caixas': caixas_validas,
        'recortes_raw': recortes_raw
    })
print(f" -> Cache de Teste Criado: {tot_gabaritos} cones reais a serem encontrados nas imagens.")

# ==========================================================
# 3. EXECUÇÃO DO GRID SEARCH (WiSARD Multi-Run)
# ==========================================================
print("\n[Etapa 3/3] Iniciando Grid Search...")
historico_resultados = []

for modo in GRID_MODOS:
    print(f"\n" + "="*50)
    print(f" AVALIANDO MODO DE BINARIZAÇÃO: {modo.upper()}")
    print("="*50)
    
    for res in GRID_RESOLUCOES:
        X_train_base = []
        for ml, mb, mc in cache_treino_cones: 
            X_train_base.append(binarizar_para_resolucao(ml, mb, res, mc, modo))
        for ml, mb, mc in cache_treino_fundos: 
            X_train_base.append(binarizar_para_resolucao(ml, mb, res, mc, modo))
            
        y_train_base = ['cone'] * len(cache_treino_cones) + ['nao_cone'] * len(cache_treino_fundos)
        
        for img_data in cache_teste:
            img_data['X_test'] = [binarizar_para_resolucao(ml, mb, res, mc, modo) for ml, mb, mc in img_data['recortes_raw']]
        
        for tupla in GRID_TUPLAS:
            for ign_zero in GRID_IGNORE_ZERO:
                print(f"   -> [Res={res:2d}, Tupla={tupla:2d}, IgnZero={str(ign_zero):<5}] Avaliando... ", end="")
                
                soma_precisao, soma_recall, soma_f1, soma_iou = 0, 0, 0, 0
                
                for rodada in range(NUM_EXECUCOES):
                    combinado = list(zip(X_train_base, y_train_base))
                    random.shuffle(combinado)
                    X_train_shuf, y_train_shuf = zip(*combinado)
                    
                    dataset_treino = wp.DataSet(list(X_train_shuf), list(y_train_shuf))
                    modelo = wp.Wisard(tupla, ignoreZero=ign_zero)
                    modelo.train(dataset_treino)
                    
                    tp_oficial, fp = 0, 0
                    ious_da_rodada = []
                    
                    for img_data in cache_teste:
                        if not img_data['X_test']: continue
                            
                        preds = modelo.classify(wp.DataSet(img_data['X_test']))
                        
                        caixas_filtradas = []
                        aprovadas_raw = [img_data['caixas'][i] for i, p in enumerate(preds) if p == 'cone']
                        for box_raw in sorted(aprovadas_raw, key=lambda b: b[2]*b[3], reverse=True):
                            if not any(calcular_iom(box_raw, b_apr) > 0.6 for b_apr in caixas_filtradas):
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
                                    ious_da_rodada.append(melhor_iou)
                                is_tp = True
                                
                            if not is_tp: fp += 1
                                
                    prec_rodada = (tp_oficial / (tp_oficial + fp)) * 100 if (tp_oficial + fp) > 0 else 0
                    rec_rodada = (tp_oficial / tot_gabaritos) * 100 if tot_gabaritos > 0 else 0
                    f1_rodada = 2 * (prec_rodada * rec_rodada) / (prec_rodada + rec_rodada) if (prec_rodada + rec_rodada) > 0 else 0
                    iou_medio_rodada = (np.mean(ious_da_rodada) * 100) if ious_da_rodada else 0
                    
                    soma_precisao += prec_rodada
                    soma_recall += rec_rodada
                    soma_f1 += f1_rodada
                    soma_iou += iou_medio_rodada
                    
                media_prec = soma_precisao / NUM_EXECUCOES
                media_rec = soma_recall / NUM_EXECUCOES
                media_f1 = soma_f1 / NUM_EXECUCOES
                media_iou = soma_iou / NUM_EXECUCOES
                
                # --- AQUI ESTÁ O PRINT CORRIGIDO ---
                print(f"Média F1: {media_f1:.2f}% (Prec: {media_prec:.1f}%, Rec: {media_rec:.1f}%, Média IoU: {media_iou:.2f}%)")
                
                historico_resultados.append({
                    'modo': modo, 'res': res, 'tupla': tupla, 'ign_zero': ign_zero,
                    'precisao': media_prec, 'recall': media_rec, 'f1': media_f1, 'iou': media_iou
                })

print("\n" + "="*70)
print(f" TOP 10 RESULTADOS DO GRID SEARCH (Média de {NUM_EXECUCOES} execuções)")
print("="*70)
historico_ordenado = sorted(historico_resultados, key=lambda x: x['f1'], reverse=True)

# --- AQUI ESTÁ O RANKING FINAL COM IOU DE VOLTA ---
for i, r in enumerate(historico_ordenado[:10]):
    print(f" {i+1:2d}º -> Modo: {r['modo']:7s} | Res: {r['res']:2d} | Tupla: {r['tupla']:2d} | IgnZero: {str(r['ign_zero']):<5} || F1 Média: {r['f1']:.2f}% (Prec: {r['precisao']:5.1f}%, Rec: {r['recall']:5.1f}%, Média IoU: {r['iou']:5.2f}%)")