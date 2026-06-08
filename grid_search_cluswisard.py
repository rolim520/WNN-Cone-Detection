import cv2
import numpy as np
import gc
import os
import glob
import time
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import wisardpkg as wp
from utils import *

# ==========================================================
# 1. PARÂMETROS BASE E ESPAÇO DE BUSCA (CLUSWISARD)
# ==========================================================
ESTADO = carregar_configuracoes()

GRID_MODOS = ['cor'] 
GRID_RESOLUCOES = [64] 
GRID_ADDRESS_SIZE = [16, 20, 24, 28]
GRID_IGNORE_ZERO = [False]

# Novos Parâmetros Exclusivos da ClusWisard
GRID_MIN_SCORE = [0.5, 0.7, 0.8]
GRID_THRESHOLD = [786, 1573, 2500]      # Valores atualizados!
GRID_DISC_LIMIT = [5, 15]          

NUM_EXECUCOES = 5
LIMIAR_AR_CONE = 1.25

# ==========================================================
# VARIÁVEIS GLOBAIS (Compartilhamento de Memória para CPUs)
# ==========================================================
GLOBAL_DATA = {}
GLOBAL_TOT_GAB = 0

# ==========================================================
# FUNÇÃO DO WORKER (Roda em cada núcleo da CPU)
# ==========================================================
# ==========================================================
# FUNÇÃO DO WORKER (Atualizada com Proteção de Memória)
# ==========================================================
def worker_avaliar_configuracao(params):
    modo, res, tupla, m_score, thresh, d_limit, ign_zero = params
    
    # Recupera os dados pré-binarizados da memória compartilhada
    X_train_base, y_train_base, cache_teste_pronto = GLOBAL_DATA[(modo, res)]
    
    soma_precisao, soma_recall, soma_f1, soma_iou = 0, 0, 0, 0
    
    for rodada in range(NUM_EXECUCOES):
        # 1. Cria a cópia embaralhada (Isso consome muita RAM)
        combinado = list(zip(X_train_base, y_train_base))
        random.shuffle(combinado)
        X_train_shuf, y_train_shuf = zip(*combinado)
        
        modelo = wp.ClusWisard(
            tupla, m_score, thresh, d_limit, ignoreZero=ign_zero, verbose=False
        )
        
        # 2. Transfere os dados para o C++
        dataset_treino = wp.DataSet(list(X_train_shuf), list(y_train_shuf))
        modelo.train(dataset_treino)
        
        # 3. CRÍTICO: Deleta as cópias gigantescas do Python imediatamente!
        del combinado
        del X_train_shuf
        del y_train_shuf
        del dataset_treino
        gc.collect() # Força a limpeza
        
        tp_oficial, fp = 0, 0
        ious_da_rodada = []
        
        for img_data in cache_teste_pronto:
            if not img_data['X_test']: continue
                
            # Cria e treina
            dataset_teste = wp.DataSet(img_data['X_test'])
            preds = modelo.classify(dataset_teste)
            
            # Limpa o dataset de teste
            del dataset_teste
            
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
        rec_rodada = (tp_oficial / GLOBAL_TOT_GAB) * 100 if GLOBAL_TOT_GAB > 0 else 0
        f1_rodada = 2 * (prec_rodada * rec_rodada) / (prec_rodada + rec_rodada) if (prec_rodada + rec_rodada) > 0 else 0
        iou_medio_rodada = (np.mean(ious_da_rodada) * 100) if ious_da_rodada else 0
        
        soma_precisao += prec_rodada
        soma_recall += rec_rodada
        soma_f1 += f1_rodada
        soma_iou += iou_medio_rodada
        
        # 4. Deleta o modelo da memória RAM ao final do loop
        del modelo
        gc.collect() 
        
    resultado = {
        'modo': modo, 'res': res, 'tupla': tupla, 
        'm_score': m_score, 'thresh': thresh, 'd_limit': d_limit, 'ign_zero': ign_zero,
        'precisao': soma_precisao / NUM_EXECUCOES, 
        'recall': soma_recall / NUM_EXECUCOES, 
        'f1': soma_f1 / NUM_EXECUCOES, 
        'iou': soma_iou / NUM_EXECUCOES
    }
    return resultado


# ==========================================================
# CÓDIGO PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    
    print("="*70)
    print(f" GRID SEARCH PARALELO DA CLUSWISARD ({NUM_EXECUCOES} Execuções por Setup)")
    print(f" -> Núcleos de CPU detectados: {multiprocessing.cpu_count()}")
    print("="*70)

    print("\n[Etapa 1/3] Extraindo recortes das bases de imagens...")
    cache_treino_cones, cache_treino_fundos = [], []
    
    # 1. Cache Treino
    for arq in glob.glob("images/train/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO)
        
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
    
    # 2. Cache Teste
    cache_teste = []
    for arq in glob.glob("images/test/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        GLOBAL_TOT_GAB += len(gabaritos)
        
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO)
        candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO)
        
        caixas_validas, recortes_raw = [], []
        for (x, y, w, h) in candidatos:
            x, y = max(0, int(x)), max(0, int(y))
            c_lar, c_br, c_canny = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w], mask_canny[y:y+h, x:x+w]
            if c_lar.size > 0:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                caixas_validas.append((x, y, w, h))
                recortes_raw.append((c_lar, c_br, c_canny))
                
        cache_teste.append({
            'gabaritos': gabaritos,
            'caixas': caixas_validas,
            'recortes_raw': recortes_raw
        })

    print(f" -> Extração Concluída.")

    print("\n[Etapa 2/3] Binarizando e montando dados na Memória Compartilhada...")
    # Binariza todas as imagens uma única vez para todos os workers poderem ler
    for modo in GRID_MODOS:
        for res in GRID_RESOLUCOES:
            X_train_base = [binarizar_para_resolucao(ml, mb, res, mc, modo) for ml, mb, mc in cache_treino_cones]
            X_train_base += [binarizar_para_resolucao(ml, mb, res, mc, modo) for ml, mb, mc in cache_treino_fundos]
            y_train_base = ['cone'] * len(cache_treino_cones) + ['nao_cone'] * len(cache_treino_fundos)
            
            cache_teste_pronto = []
            for img_data in cache_teste:
                x_test_bin = [binarizar_para_resolucao(ml, mb, res, mc, modo) for ml, mb, mc in img_data['recortes_raw']]
                cache_teste_pronto.append({
                    'gabaritos': img_data['gabaritos'],
                    'caixas': img_data['caixas'],
                    'X_test': x_test_bin
                })
            
            # Salva na memória global
            GLOBAL_DATA[(modo, res)] = (X_train_base, y_train_base, cache_teste_pronto)

    print("\n[Etapa 3/3] Disparando Threads do Grid Search (Proteção Anti-Leak Ativada)...")
    tarefas = []
    for modo in GRID_MODOS:
        for res in GRID_RESOLUCOES:
            for tupla in GRID_ADDRESS_SIZE:
                for m_score in GRID_MIN_SCORE:
                    for thresh in GRID_THRESHOLD:
                        for d_limit in GRID_DISC_LIMIT:
                            for ign_zero in GRID_IGNORE_ZERO:
                                tarefas.append((modo, res, tupla, m_score, thresh, d_limit, ign_zero))

    historico_resultados = []
    tempo_inicial = time.perf_counter()
    
    # Reduzir núcleos para garantir estabilidade com imagens 64x64
    max_workers = min(6, multiprocessing.cpu_count())
    
    print(f" -> Iniciando Pool com {max_workers} workers.")
    print(" -> maxtasksperchild=1 ativado: A memória C++ será destruída a cada iteração.\n")
    
    # =========================================================================
    # A MÁGICA ACONTECE AQUI: maxtasksperchild=1
    # =========================================================================
    with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        concluidos = 0
        total = len(tarefas)
        
        # imap_unordered vai retornando os resultados conforme os núcleos terminam
        for res in pool.imap_unordered(worker_avaliar_configuracao, tarefas):
            historico_resultados.append(res)
            concluidos += 1
            
            cfg = f"Tup:{res['tupla']:2d}|mSc:{res['m_score']}|th:{res['thresh']:4d}|dLim:{res['d_limit']:2d}"
            print(f"[{concluidos:3d}/{total}] {cfg} -> F1: {res['f1']:5.2f}% (Prec: {res['precisao']:.1f}%, Rec: {res['recall']:.1f}%)")
    # =========================================================================

    tempo_total = (time.perf_counter() - tempo_inicial) / 60
    
    print("\n" + "="*80)
    print(f" TOP 15 RESULTADOS DA CLUSWISARD (Tempo de Grid: {tempo_total:.1f} min)")
    print("="*80)
    historico_ordenado = sorted(historico_resultados, key=lambda x: x['f1'], reverse=True)

    for i, r in enumerate(historico_ordenado[:15]):
        config = f"Res:{r['res']}|Tup:{r['tupla']:2d}|mSc:{r['m_score']}|th:{r['thresh']:4d}|dLim:{r['d_limit']:2d}"
        print(f" {i+1:2d}º -> {config} || F1: {r['f1']:5.2f}% (Prec: {r['precisao']:5.1f}%, Rec: {r['recall']:5.1f}%, IoU: {r['iou']:5.2f}%)")

    # =========================================================================
    # NOVO: SALVAR RESULTADOS EM JSON
    # =========================================================================
    nome_arquivo_json = f"resultados_grid_search_res{GRID_RESOLUCOES[0]}.json"
    
    with open(nome_arquivo_json, 'w', encoding='utf-8') as f:
        # indent=4 deixa o arquivo formatado e legível para humanos
        json.dump(historico_ordenado, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*80)
    print(f"[*] Todos os resultados salvos com sucesso em: {nome_arquivo_json}")
    print("="*80)