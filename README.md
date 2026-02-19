# Text-to-Motion Retrieval — Documentation Projet

## Vue d'ensemble

Ce projet implémente un système de **retrieval text-to-motion** : étant donné une description textuelle d'une interaction humaine (ex : *"A person approaches and hands an object to another"*), le modèle retrouve la séquence de mouvement 3D correspondante parmi des candidats.

L'architecture s'inspire du papier **TMR** (*Text-to-Motion Retrieval*) et utilise un alignement cross-modal dans un espace embedding partagé.

---

## Dataset

Le dataset contient des **motions d'interaction à deux personnes** (donner un objet, pousser, saluer, etc.).

| Caractéristique | Détail |
|---|---|
| Motions uniques | ~6 000 |
| Format motion | `.npy` — shape `(T, 396)` — `T` frames, 44 joints × 9 dims |
| Textes | 2–4 descriptions par motion (paraphrases) |
| Samples effectifs | ~18 000 (après duplication par texte) |
| Évaluation | 30 batches × 32 candidats, 1 correct par batch |

Chaque motion encode la position 3D + rotations 6D des deux personnes à chaque frame. Les descriptions couvrent les rôles, l'intention et l'outcome de l'interaction.

---

## Architecture

```
Texte  ──→ [DistilBERT] ──→ [Projection MLP] ──→ embedding (256d) ──┐
                                                                      ├──→ Similarity Score
Motion ──→ [HumanML3D Transformer] ──────────────→ embedding (256d) ──┘
```

### Text Encoder — DistilBERT
- Modèle pré-entraîné `distilbert-base-uncased` (66M paramètres)
- Extraction du token `[CLS]` comme représentation de la phrase
- Projection MLP : `768 → 512 → 256` avec LayerNorm + Dropout
- Fine-tuné end-to-end (non gelé)

**Pourquoi DistilBERT plutôt que CLIP ?**  
CLIP encode texte et image conjointement ; ici, le texte décrit du mouvement, pas des images. DistilBERT est pré-entraîné sur du langage général et capture mieux les nuances sémantiques des descriptions d'action.

### Motion Encoder — HumanML3DTransformer
Architecture du papier TMR adaptée à notre dataset :

| Composant | Choix | Raison |
|---|---|---|
| Positional encoding | **Sinusoïdal** (non-appris) | Plus stable, généralise mieux aux longueurs variables |
| Token spécial | **Action token** (comme [CLS] BERT) | Agrège l'information sans pooling manuel |
| Activation | **GELU** (vs ReLU) | Gradient plus lisse, convergence plus stable |
| Layers | 4–8 Transformer layers | Dépend de la taille du dataset |
| Architecture | Pre-LN (LayerNorm avant attention) | Entraînement plus stable |

Le motion encoder reçoit une séquence `(T, 384)` et produit un vecteur `256d`.

**Alternative disponible** : `TransformerMotionEncoder` (learnable pos encoding, mean pooling, ReLU) — sélectionnable via `Config.MOTION_MODEL_TYPE = 'transformer'`.

### Espace Embedding Partagé
Les deux embeddings sont normalisés L2 sur la sphère unitaire.  
La similarité entre un texte et une motion se calcule comme le **produit scalaire** (= cosinus similitude après normalisation).

---

## Loss Function

### AlignedContrastiveLoss (recommandée)

Combine trois termes pour résoudre la séparation modale text/motion :

```
L = L_contrastive + α × L_MSE + β × L_uniformity
```

**L_contrastive** (CLIP-style, bidirectionnel) : maximise la similarité des paires correctes, minimise les incorrectes dans un batch.

**L_MSE** sur paires positives : force les embeddings `text_i` et `motion_i` à être proches dans L2, résout la séparation modale observée sur le t-SNE (les deux nuages séparés).

**L_uniformity** : distribue les embeddings uniformément sur la sphère, évite le mode collapse.

```python
USE_ALIGNED_LOSS   = True
MSE_WEIGHT         = 0.15   # α
UNIFORMITY_WEIGHT  = 0.01   # β
TEMPERATURE        = 0.05
```

### Alternative : TEMOSLoss (papier original)
```
L = L_contrastive + λ_KL × L_KL
```
Utilisée si `USE_ALIGNED_LOSS = False`.

---

## Entraînement

### Pipeline
1. **Normalisation globale** des motions (mean/std calculés sur le train set)
2. **Augmentation** (train uniquement) : time masking, noise injection, temporal shift, random crop
3. **Batch** : 32 paires (text, motion) — les négatifs sont tous les autres éléments du batch
4. **Optimiseur** : AdamW avec warmup linéaire (5 epochs) + cosine decay
5. **Gradient clipping** à 0.5

### Early Stopping
Deux modes :
- `'recall'` ← recommandé : patience sur `recall@1` — arrête quand le recall ne monte plus même si la val loss stagne (votre cas)
- `'val_loss'` : comportement classique

### Tracking MLflow
Chaque run enregistre :
- Hyperparamètres complets
- Loss train/val par epoch
- `cos_correct`, `cos_incorrect`, `alignment_gap` (cosinus sur exemples viz)
- Métriques avancées (`centroid_dist`, `diagonal_sim`, `recall@1/5/10`, `gap`)
- Matrices de similarité (images, toutes les N epochs)
- Poids du meilleur modèle

```bash
# Lancer l'UI MLflow
mlflow ui --backend-store-uri ./mlruns
```

---

## Évaluation

### Métriques internes (val set pendant training)

| Métrique | Description | Cible |
|---|---|---|
| `alignment_gap` | cos(correct) − cos(incorrect) sur train viz | > 0.80 |
| `adv_centroid_dist` | Distance text centroid ↔ motion centroid | < 0.05 |
| `adv_diagonal_sim` | Similarité moyenne des paires correctes | > 0.60 |
| `adv_gap` | diagonal_sim − off_diagonal_sim | > 0.50 |
| `adv_recall@1` | Recall@1 sur val set (batchs de 32) | > 0.15 |

### Évaluation finale (test officiel)
30 batches de 32 candidats, 1 correct par batch. Score = **Weighted Recall@K** (K=1..10, poids 1/K).

### Prédiction par Ensemble (Section 16bis)
Plusieurs modèles indépendants votent pour chaque candidat. Le score final est une **moyenne pondérée des rangs** (Borda count) ou une **moyenne des similarités normalisées**. Plus robuste qu'un seul modèle, équivalent du random forest pour la retrieval.

---

## Structure du Notebook

| Section | Description |
|---|---|
| 0. Config | Tous les hyperparamètres centralisés |
| 1. Imports | Dépendances, device, MLflow setup |
| 2. Visualisation | t-SNE, dashboard, matrice similarité |
| 3. Évaluation | Fonction recall pondéré |
| 4. Val batches | Génération des batches de validation |
| 5. Text Encoders | DistilBERT + CLIP (sélectionnable) |
| 6. Motion Encoders | HumanML3D + Transformer + LSTM |
| 7. Dataset | Augmentation, preprocessing, DataLoader |
| 8. Loss Functions | AlignedContrastive + TEMOS + NTXent |
| 9. Encode functions | encode_text, encode_motion, compute_loss |
| 10. Optimiseur | AdamW + warmup + scheduler |
| 11. Dashboard | Visualisation live training |
| 12. Training Loop | Entraînement + MLflow + checkpoint |
| 13. Graphiques | Courbes finales |
| 14. Save/Load | Sauvegarde rapide |
| 15. Prédiction | Génération submission.csv |
| 16. Évaluation | Score final recall |
| **16bis. Ensemble** | **Vote multi-modèles (nouveau)** |
| 17. Reprise | Reprise depuis checkpoint |

---

## Résultats Obtenus

| Run | Architecture | Loss | Recall final |
|---|---|---|---|
| Baseline | Transformer (4L) | NTXent | ~0.77 |
| V2 | HumanML3D (8L) | AlignedContrastive | **0.89** |
| V3 | HumanML3D (4L) | TEMOS | En cours |

### Observations clés
- **t-SNE mélangé** (paires text/motion proches) ✅ — grâce à la MSE loss
- **Overfitting fort** (train loss → 0, val loss monte) — lié à la taille du dataset
- Le **recall final sur le test** (0.89) reste bon malgré l'overfitting car l'espace d'embedding est bien organisé

---

## Dépendances

```bash
pip install torch transformers open_clip_torch mlflow scikit-learn tqdm seaborn
```

| Librairie | Usage |
|---|---|
| `torch` | Modèles, training |
| `transformers` | DistilBERT |
| `open_clip_torch` | CLIP (alternative) |
| `mlflow` | Tracking expériences |
| `scikit-learn` | t-SNE, train/val split |
| `seaborn` | Heatmaps similarité |

---

## Fichiers Générés

```
checkpoints/
├── motion_encoder_best.pt    ← Meilleur motion encoder
├── text_encoder_best.pt      ← Meilleur text encoder
└── checkpoint_latest.pt      ← Checkpoint complet (reprise)

mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── params/           ← Config complète
        ├── metrics/          ← Toutes les métriques par epoch
        └── artifacts/
            ├── motion_encoder/       ← Modèle PyTorch MLflow
            ├── text_encoder/
            └── similarity_matrices/  ← Images heatmap par epoch

submission.csv                ← Prédictions finales
```
