# 🏃 Text–Motion Retrieval — Data Challenge

> Étant donné une description textuelle (*"a person walks forward and sits down"*), retrouver parmi N motions candidates celle qui correspond.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Installation](#installation)
4. [Format des données](#format-des-données)
5. [Architecture du modèle](#architecture-du-modèle)
6. [Pipeline complet](#pipeline-complet)
7. [Entraînement](#entraînement)
8. [Évaluation](#évaluation)
9. [Prédiction & Soumission](#prédiction--soumission)
10. [Hyperparamètres](#hyperparamètres)
11. [Résultats & Visualisation](#résultats--visualisation)

---

## Vue d'ensemble

Le problème est un **retrieval cross-modal** : aligner un espace textuel et un espace de mouvements humains dans un espace vectoriel commun, de façon à pouvoir mesurer leur similarité par produit scalaire.

```
Texte  ──[CLIP text encoder]──────────────────────────┐
                                                        ├──► cosine similarity ──► ranking
Motion ──[MotionEncoder (Transformer 1D)]──────────────┘
```

Les deux encodeurs projettent dans le **même espace de dimension 512**. La loss **InfoNCE** les force à aligner les paires correctes (texte_i ↔ motion_i) tout en éloignant les paires incorrectes.

---

## Structure du projet

```
.
├── data/
│   ├── motions/          ← fichiers .npy  (shape : T × 384)
│   ├── texts/            ← fichiers .txt  (1 à 3 descriptions par motion)
│   ├── train.txt         ← liste des noms de fichiers d'entraînement
│   └── val/              ← batches de validation générés automatiquement
│       ├── 1/
│       │   ├── text.txt
│       │   ├── motion_1.npy
│       │   └── ...
│       ├── gt.csv        ← vérités terrain
│       └── ...
├── text-motion_retrieval.ipynb
├── motion_encoder.pt           ← sauvegarde du MotionEncoder après entraînement
├── clip_model_finetuned.pt     ← sauvegarde de CLIP fine-tuné
├── submission.csv              ← fichier de soumission final
└── tsne_latent_space.png       ← visualisation de l'espace latent
```

---

## Installation

```bash
pip install torch torchvision open-clip-torch diffusers \
            scikit-learn matplotlib pandas tqdm info-nce
```

> **GPU recommandé.** Le code fonctionne aussi en CPU mais l'entraînement sera très lent.

---

## Format des données

### Motions
Chaque fichier `.npy` contient un array NumPy de shape `(T, 384)` :
- `T` = nombre de frames (variable selon la séquence)
- `384` = vecteur de pose par frame, encodant les positions et rotations des articulations des **2 personnes** (format HumanML3D multi-personne)

```python
motion = np.load('data/motions/00001.npy')
# motion.shape → (184, 384)
```

La motion encode pour chaque personne :
- `22 × 3` positions 3D des joints
- `21 × 6` rotations des joints (représentation 6D)
- vitesses et positions globales

### Textes
Chaque fichier `.txt` contient **1 à 3 descriptions** de la motion correspondante, une par ligne :

```
a person walks forward and sits down on a chair.
the human walks ahead and takes a seat.
someone walks to a chair and sits.
```

---

## Architecture du modèle

### Encodeur texte : CLIP (ViT-B/32)

CLIP est pré-entraîné sur des milliards de paires texte-image. Son encodeur textuel comprend naturellement le langage décrivant des actions physiques.

- **Partie visuelle** : gelée (inutilisée, économise la mémoire)
- **Partie textuelle** : fine-tunée à `lr × 0.1` pour s'adapter au vocabulaire du mouvement
- **Sortie** : vecteur de dim `512`, normalisé L2

```python
text_emb = F.normalize(clip_model.encode_text(tokens), dim=-1)  # (B, 512)
```

### Encodeur motion : Transformer 1D

Les motions sont des **séquences temporelles**, pas des images. Un Transformer 1D peut faire de l'attention entre les frames, capturant la dynamique du mouvement.

```
(B, T, 384)
    │
    ▼  Linear(384 → 512) + positional embedding
(B, T, 512)
    │
    ▼  TransformerEncoder (4 layers, 8 heads)
(B, T, 512)
    │
    ▼  Mean pooling sur la dimension temporelle T
(B, 512)
    │
    ▼  LayerNorm
(B, 512)  ──► normalize L2 ──► embedding final
```

> **Pourquoi pas un VAE image ?** Un `AutoencoderKL` est conçu pour des images RGB 2D. Passer une motion en image 2D n'a pas de sens physique : il n'y a aucune localité spatiale à exploiter, et l'encodage détruirait la structure temporelle.

---

## Pipeline complet

### 1. Preprocessing des motions (`preprocess_motion`)

```python
def preprocess_motion(motion, max_len=512):
    # Normalisation : mean=0, std=1 par feature
    motion = (motion - motion.mean(0)) / (motion.std(0) + 1e-6)

    # Troncature si T > max_len
    if T >= max_len:
        motion = motion[:max_len]
    # Padding avec des zéros si T < max_len
    else:
        pad = np.zeros((max_len - T, F))
        motion = np.vstack([motion, pad])

    return torch.tensor(motion)   # (max_len, 384)
```

### 2. Dataset (`TextMotionDataset`)

Chaque entrée du dataset est une paire `(texte, motion)` alignée par nom de fichier. À chaque accès, **une des 3 descriptions est choisie aléatoirement** — c'est une forme d'augmentation de données qui améliore la généralisation.

```python
# fnames = ['00001', '00002', ...]  ← noms communs aux .npy et .txt
dataset = TextMotionDataset(fnames, data_root)
# dataset[0] → ("a person walks forward...", tensor(512, 384))
```

### 3. DataLoader & collate

La tokenisation CLIP est faite dans la `collate_fn` (plus efficace que dans `__getitem__`) :

```python
def collate_fn(batch):
    texts, motions = zip(*batch)
    text_tokens  = tokenizer(list(texts))     # (B, 77)  — 77 tokens max CLIP
    motion_batch = torch.stack(motions)       # (B, 512, 384)
    return text_tokens, motion_batch
```

---

## Entraînement

### Loss : InfoNCE

Pour un batch de N paires, la matrice de similarité cosinus est :

```
               motion_1  motion_2  ...  motion_N
  texte_1    [  0.95      0.12     ...   0.08  ]   ← on veut maximiser la diagonale
  texte_2    [  0.11      0.91     ...   0.05  ]
  ...
  texte_N    [  0.07      0.09     ...   0.88  ]
```

La loss est une **cross-entropy symétrique** : elle maximise les valeurs diagonales (paires correctes) et minimise les hors-diagonales (N-1 négatifs implicites par exemple). Plus le batch est grand, plus il y a de négatifs durs → apprentissage plus riche. C'est exactement la loss de CLIP.

```python
loss = InfoNCE(temperature=0.07)(text_emb, motion_emb)
```

La température `0.07` contrôle la netteté de la distribution : plus elle est basse, plus la loss est "dure" (pénalise fortement les mauvais classements).

### Optimiseur

```python
optimizer = AdamW([
    {'params': clip_model.transformer.parameters(), 'lr': 1e-5},  # CLIP : doucement
    {'params': motion_encoder.parameters(),         'lr': 1e-4},  # Motion enc : normal
])
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

Le scheduler **cosinus** diminue le learning rate progressivement pour stabiliser la convergence en fin d'entraînement.

### Boucle d'entraînement (résumé)

```
Pour chaque époque :
  Pour chaque batch (text_tokens, motion_batch) :
    1. text_emb   = normalize(CLIP.encode_text(text_tokens))
    2. motion_emb = normalize(MotionEncoder(motion_batch))
    3. loss = InfoNCE(text_emb, motion_emb)
    4. loss.backward() + optimizer.step()
  
  Valider sur val_loader (sans gradient)
  scheduler.step()
  Sauvegarder les modèles
```

---

## Évaluation

### Génération des batches de validation

Pour évaluer localement sans le test set officiel, `generate_val_batches` simule le protocole du challenge :

1. Tire **30 groupes de 32 motions** depuis le train set
2. Pour chaque groupe, choisit **une motion** comme requête (texte)
3. Les 32 motions sont les candidates — une seule est la bonne réponse
4. Sauvegarde les paires et les vérités terrain dans `gt.csv`

### Métrique : Recall@K pondéré

```
Score = Σ (1/k × Recall@k)  /  Σ (1/k)     pour k = 1..10
```

| k | Poids | Interprétation |
|---|-------|----------------|
| 1 | 1.000 | La bonne motion est-elle en 1ère position ? |
| 2 | 0.500 | Est-elle dans le top-2 ? |
| 5 | 0.200 | Est-elle dans le top-5 ? |
| 10| 0.100 | Est-elle dans le top-10 ? |

Bien classer en position 1 est **10× plus important** que de la trouver en position 10.

```python
score = eval_recall(gt_df, submission_df, verbose=True)
# k=1 => recall@1=0.45
# k=2 => recall@2=0.61
# ...
# Score pondéré : 0.52
```

---

## Prédiction & Soumission

Pour chaque query (texte) face à ses N motions candidates :

```python
# 1. Encoder le texte
text_emb = encode_text(tokenizer([query_text]))         # (1, 512)

# 2. Encoder toutes les motions candidates
motion_embs = encode_motion(motions)                    # (N, 512)

# 3. Similarité cosinus → classement
sims    = (text_emb @ motion_embs.T).squeeze(0)         # (N,)
top_idx = torch.topk(sims, k=10).indices                # top-10

# 4. Construire la ligne de soumission
row = {'query_id': id, 'candidate_1': ..., ..., 'candidate_10': ...}
```

Le fichier `submission.csv` a le format :

```
query_id,candidate_1,candidate_2,...,candidate_10
1,15,3,27,8,...
2,4,21,9,12,...
```

---

## Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `EMBED_DIM` | 512 | Même dimension que CLIP text |
| `MOTION_DIM` | 384 | Dimension d'une frame brute |
| `MAX_SEQ_LEN` | 512 | Couvre ~99% des séquences |
| `BATCH_SIZE` | 64 | Plus grand = plus de négatifs InfoNCE |
| `EPOCHS` | 30 | Convergence observée empiriquement |
| `LR` (motion enc) | 1e-4 | Standard AdamW |
| `LR` (CLIP) | 1e-5 | Fine-tuning doux |
| `TEMPERATURE` | 0.07 | Valeur CLIP originale |
| `nhead` | 8 | Standard Transformer |
| `num_layers` | 4 | Compromis capacité/vitesse |

---

## Résultats & Visualisation

### Courbe de loss

La loss InfoNCE diminue au fil des époques sur train et val. Une divergence entre les deux courbes indique de l'overfitting → réduire les epochs ou augmenter le dropout.

### t-SNE de l'espace latent

Après entraînement, on encode N paires (texte, motion) et on projette en 2D via t-SNE. Les lignes grises relient chaque texte à sa motion correspondante.

- **Avant entraînement** : lignes longues et aléatoires — les deux modalités sont dans des régions séparées de l'espace
- **Après entraînement** : lignes courtes — textes et motions correspondants sont proches

```python
combined = np.vstack([text_arr, motion_arr])   # (2N, 512)
proj_2d  = TSNE(n_components=2).fit_transform(combined)
```

> Le t-SNE est calculé sur la **concaténation** texte+motion pour que les deux modalités soient dans le même espace de projection 2D, permettant une comparaison visuelle directe.

---

## Références

- [CLIP — Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., OpenAI 2021)
- [InfoNCE — Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (van den Oord et al., DeepMind 2018)
- [TMR — Text-to-Motion Retrieval](https://arxiv.org/abs/2305.00976) (Petrovich et al., 2023)
- [HumanML3D Dataset](https://github.com/EricGuo5513/HumanML3D) (Guo et al., 2022)