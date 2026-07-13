# Guia de Datasets para Análise de Vídeo - Tech Challenge Fase 4

## Dataset já baixado

### 1. Sitting Posture Classification (Roboflow Universe)
- **Fonte:** https://universe.roboflow.com/leonardo-sabino/sitting-posture-classification-ccvao-31o25
- **Licença:** CC BY 4.0
- **Formato:** Classificação (folder format)
- **Imagens:** 2.347 imagens
- **Classes:** Posturas sentadas (good posture, bad postures)
- **Uso no projeto:** Treinar modelo YOLOv8-cls para classificação postural
- **Local:** `data/datasets/video/Sitting Posture Classification.folder/`

---

## Datasets Recomendados para Baixar

### ANÁLISE POSTURAL E FISIOTERAPIA

### 2. KIMORE - KiMoRe Dataset (PhysioNet)
- **Fonte:** https://physionet.org/content/kimore/
- **Descrição:** Dataset italiano de exercícios de reabilitação com captura de movimento 3D (Kinect v2).
  Contém 5 exercícios diferentes realizados por 78 sujeitos (44 saudáveis + 34 com problemas motores).
- **Formato:** CSV com coordenadas 3D de articulações + vídeos RGB-D
- **Exercícios:** Levantar braço frontal, levantar braço lateral, rotação de tronco, etc.
- **Uso no projeto:**
  - Detecção de desvios posturais durante fisioterapia
  - OpenPose para extrair keypoints e comparar com referência
  - Classificação de exercícios corretos vs incorretos

**Download:**
```bash
# Requer cadastro gratuito no PhysioNet
wget -r -N -c -np --user YOUR_USER --password YOUR_PASS \
  https://physionet.org/files/kimore/latest/
```

### 3. UI-PRMD - University of Idaho PRMD (PhysioNet)
- **Fonte:** https://physionet.org/content/ui-prmd/
- **Descrição:** Movimentos de reabilitação capturados com Kinect e Vicon (motion capture profissional).
  10 exercícios realizados por 10 sujeitos.
- **Formato:** Coordenadas 3D (Kinect) + ground truth (Vicon)
- **Uso no projeto:** Ground truth para validação de OpenPose, baseline de movimento correto
- **Exercícios:** Deep squat, hurdle step, inline lunge, shoulder mobility, etc.

**Download:**
```bash
wget -r -N -c -np https://physionet.org/files/ui-prmd/latest/
```

### 4. NTU RGB+D 120
- **Fonte:** https://rose1.ntu.edu.sg/dataset/actionRecognition/
- **Descrição:** 120 classes de ações humanas, incluindo ações médicas e de saúde.
  114.480 vídeos RGB+D com anotações de esqueleto 3D.
- **Classes relevantes:** sitting down, standing up, falling, walking, stretching,
  wearing jacket, taking off jacket, etc.
- **Uso no projeto:**
  - Reconhecimento de ações em vídeos hospitalares
  - Detecção de quedas (fall detection)
  - Modelo base para anomalias de movimento

**Download:** (requer solicitação no site)

### 5. COCO Keypoints (MS COCO)
- **Fonte:** https://cocodataset.org/#keypoints-2020
- **Descrição:** Dataset de referência para pose estimation. 250K+ instâncias de pessoas
  com 17 keypoints anotados.
- **Uso no projeto:** Treinar/pré-treinar modelos de pose estimation (OpenPose/YOLOv8-pose)
- **Formato:** JSON com keypoints + bounding boxes

### 6. MPII Human Pose Dataset
- **Fonte:** http://human-pose.mpi-inf.mpg.de/
- **Descrição:** 25K imagens com 40K pessoas anotadas com 16 keypoints.
  Inclui atividades diversas do dia a dia.
- **Uso no projeto:** Dataset complementar para pose estimation

---

### ANÁLISE DE VÍDEOS CIRÚRGICOS

### 7. Cholec80 / CholecT50
- **Fonte:** http://camma.u-strasbg.fr/datasets
- **Descrição:** 80 vídeos de colecistectomia laparoscópica.
  Anotado com fases cirúrgicas (7 fases) e presença de ferramentas.
- **Formato:** Vídeos + anotações de fase cirúrgica e tool detection
- **Uso no projeto:**
  - Detecção de fases anômalas em cirurgia
  - YOLOv8 para detecção de instrumentos cirúrgicos
  - Identificação de desvios do protocolo cirúrgico

### 8. m2caiSeg / m2cai16-tool
- **Fonte:** http://camma.u-strasbg.fr/datasets
- **Descrição:** Segmentação semântica e detecção de ferramentas em vídeos cirúrgicos.
  Anotações de bounding box para instrumentos cirúrgicos.
- **Uso no projeto:**
  - Detecção de instrumentos fora de posição
  - Alertas de uso incorreto de ferramentas
  - Validação de protocolo cirúrgico

### 9. JIGSAWS - JHU-ISI Gesture and Skill Assessment
- **Fonte:** https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **Descrição:** Banco de dados de habilidades cirúrgicas com movimentos de sutura,
  passagem de agulha e nós.
- **Uso no projeto:** Análise de qualidade de movimentos cirúrgicos

---

### DETECÇÃO DE QUEDAS E ANOMALIAS DE MOVIMENTO

### 10. UR Fall Detection Dataset
- **Fonte:** http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- **Descrição:** 70 sequências (30 quedas + 40 atividades normais) gravadas com
  2 câmeras Kinect e acelerômetro.
- **Uso no projeto:** Detecção de quedas em ambiente hospitalar

### 11. Le2i Fall Detection Dataset
- **Fonte:** http://le2i.cnrs.fr/Fall-detection-Dataset
- **Descrição:** Vídeos de quedas simuladas e atividades normais. Múltiplos cenários
  (casa, escritório, sala de café).
- **Uso no projeto:** Treinar modelo de detecção de quedas para monitoramento hospitalar

### 12. Multiple Cameras Fall Dataset
- **Fonte:** http://www.iro.umontreal.ca/~labimage/Dataset/
- **Descrição:** 24 cenários com 8 câmeras sincronizadas, mostrando quedas e
  atividades normais.
- **Uso no projeto:** Validação multi-ângulo de detecção de anomalias de movimento

---

### DATASETS NO KAGGLE (recomendados)

### 13. Human Activity Recognition (Kaggle)
- **Fonte:** https://www.kaggle.com/datasets?search=human+activity+recognition+video
- **Descrição:** Diversos datasets de reconhecimento de atividades humanas disponíveis
- **Uso:** Classificação de atividades em ambiente hospitalar

### 14. Fall Detection Dataset (Kaggle)
- **Fonte:** https://www.kaggle.com/datasets?search=fall+detection+dataset
- **Descrição:** Conjuntos de dados de detecção de quedas para ambientes assistidos

---

## Como Baixar os Datasets

### Opção 1: Roboflow Universe (recomendado para YOLOv8)
1. Acessar https://universe.roboflow.com/
2. Buscar por "sitting posture", "physiotherapy", "exercise", "surgical", "fall detection"
3. Exportar no formato YOLOv8 (txt annotations) ou folder format
4. Datasets populares no Roboflow:
   - Sitting Posture Detection
   - Fall Detection
   - Person Detection
   - Surgical Instrument Detection

### Opção 2: PhysioNet (sinais vitais + movimento)
```bash
# Instalar dependências
pip install wfdb

# Exemplo: baixar dados de sinais vitais MIMIC-III
wget -r -N -c -np https://physionet.org/files/mimiciii/1.4/
```

### Opção 3: Kaggle
```bash
# Instalar kaggle CLI
pip install kaggle

# Autenticar (requer kaggle.json da conta Kaggle)
mkdir -p ~/.kaggle
# Colocar kaggle.json em ~/.kaggle/

# Baixar dataset específico
kaggle datasets download <dataset-path>
```

---

## Resumo para o Projeto

| Dataset | Tipo | Tamanho | Uso Principal |
|---------|------|---------|--------------|
| Sitting Posture (Roboflow) | Imagens | 2.347 | Classificação postural (YOLOv8-cls) |
| KIMORE (PhysioNet) | Vídeo 3D | 78 sujeitos | Análise de fisioterapia (OpenPose) |
| UI-PRMD (PhysioNet) | Vídeo 3D | 10 sujeitos | Ground truth de movimento |
| COCO Keypoints | Imagens | 250K+ | Pose estimation (YOLOv8-pose) |
| Cholec80 | Vídeos | 80 vídeos | Análise cirúrgica (YOLOv8) |
| UR Fall Detection | Vídeos | 70 seq. | Detecção de quedas |
| MIMIC-III | Sinais | 40K+ pacientes | Detecção de anomalias em sinais vitais |

---

## Próximos Passos Recomendados

1. **Já temos:** Sitting Posture Classification → treinar YOLOv8-cls
2. **Baixar:** COCO Keypoints → treinar/fine-tune YOLOv8-pose
3. **Baixar:** KIMORE → análise de fisioterapia com OpenPose
4. **Baixar:** UR Fall Detection → detecção de quedas em hospital
5. **Integrar:** Sinais vitais do PhysioNet → módulo de anomalias
