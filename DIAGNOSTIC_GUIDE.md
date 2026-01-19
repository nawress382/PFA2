# Guide de Diagnostic et Correction - Authentification Gauche/Droite

## Problème
L'authentification échoue sur les poses **gauche** (yaw négatif) et **droite** (yaw positif).

## Causes Possibles
1. **Tolérance trop restrictive** — écart entre angle estimé et angle cible > tolérance
2. **Données d'entraînement incohérentes** — poses capturées varient trop lors de l'entraînement
3. **Imprécision de l'estimation de pose** — bruit dans les landmarks dlib
4. **Variabilité d'une session à l'autre** — angles moyens capturés ≠ angles réels

## Solution : Plan de Diagnostic et Test

### Étape 1 : Afficher les poses apprises
```powershell
python pose_diagnostic.py
```
Cela affichera toutes les poses moyennes apprises pour chaque utilisateur.
**Cherche les valeurs yaw pour gauche/droite** — sont-elles cohérentes avec ce que tu fais ?

**Exemple de sortie attendue :**
```
alice:
  frontale  : yaw=  -2.5°  pitch=   1.3°  roll=   -0.8°
  gauche    : yaw= -35.2°  pitch=   2.1°  roll=   -1.5°
  droite    : yaw=  38.7°  pitch=   0.9°  roll=    1.2°
  haut      : yaw=  -0.5°  pitch=  25.8°  roll=   -0.3°
  bas       : yaw=   1.2°  pitch= -22.4°  roll=    0.5°
```

### Étape 2 : Vérifier l'écart en temps réel
Le script `pose_diagnostic.py` à la fin démarre un **moniteur en temps réel**. Pendant qu'il tourne :
- **Regarde les valeurs yaw/pitch/roll actuelles** au fur et à mesure que tu bouges
- **Appuie sur 'g'** pour passer à la pose "gauche"
- **Appuie sur 'd'** pour passer à la pose "droite"
- Compare l'écart affiché (err=...) à la tolérance (tol=...)
- **Si err > tol** régulièrement, c'est le problème !

Exemple affichage en direct :
```
YAW:      -28.5° -> -35.2° (err= 6.7° tol=25°)   <- OK ✓
PITCH:     2.0° ->   2.1° (err= 0.1° tol=15°)    <- OK ✓
ROLL:     -1.2° ->  -1.5° (err= 0.3° tol=15°)    <- OK ✓
✓ MATCH
```

vs

```
YAW:      -15.2° -> -35.2° (err=20.0° tol=25°)   <- BORDERLINE ✓
YAW:      -8.5° -> -35.2° (err=26.8° tol=25°)    <- FAIL ✗ (too much variation!)
```

### Étape 3 : Test d'authentification avec diagnostic
```powershell
python test_auth_diagnostic.py
```

Ce script affiche **chaque tentative** avec écarts détaillés. Tu verras :
```
frontale | yaw: -2.1 vs -2.5 (err=0.4, tol=15) | pitch: 1.2 vs 1.3 (err=0.1, tol=15) | roll: -0.9 vs -0.8 (err=0.1, tol=15)
frontale | yaw: -2.0 vs -2.5 (err=0.5, tol=15) | pitch: 1.3 vs 1.3 (err=0.0, tol=15) | roll: -0.8 vs -0.8 (err=0.0, tol=15)
gauche | yaw: -35.0 vs -35.2 (err=0.2, tol=25) | pitch: 2.0 vs 2.1 (err=0.1, tol=15) | roll: -1.5 vs -1.5 (err=0.0, tol=15)
gauche | yaw: -34.9 vs -35.2 (err=0.3, tol=25) | pitch: 2.1 vs 2.1 (err=0.0, tol=15) | roll: -1.5 vs -1.5 (err=0.0, tol=15)
```

## Solutions Basées sur les Résultats

### Cas 1 : Écarts trop grands et variables
**Problème :** Les angles estimés varient beaucoup. Augmenter la tolérance n'aidera pas car il y a aussi de la variabilité lors de la capture.

**Solutions :**
- **Recommencer la capture** avec plus de stabilité :
  - Assure-toi que ta tête est bien stable
  - Prends les 40 images lentement et sans bouger
  - Augmente `MIN_TIME_BETWEEN_SAVES` dans `face_service.py` (ex: 1.0 au lieu de 0.7)

- **Ou : Augmente le nombre d'images par pose** dans la capture :
  - Au lieu de 40, essaye 60-80 images
  - Moyenne sur plus de données → plus stable

### Cas 2 : Écarts petits mais inconstants (≈ tolérance)
**Problème :** Les angles sautent d'un côté à l'autre de la tolérance (ex: 24.8° puis 25.2°).

**Solutions :**
- **Légèrement augmenter la tolérance** :
  - Dans `face_service.py`, augmente `POSE_TOLERANCES["gauche"]` de `(25, 15, 15)` à `(28, 15, 15)`
  - Même chose pour `"droite"`

- **Ou : Améliorer les landmarks** :
  - Assure-toi que dlib détecte 68 landmarks correctement
  - La détection de visage dlib fonctionne mieux en bonne luminosité

### Cas 3 : L'écart cible est mauvais (pose apprise ≠ pose réelle)
**Problème :** En étape 2, tu vois que la valeur cible (ex: yaw=-35.2°) ne correspond pas à ce que tu fais réellement (tu es à -20° max).

**Solutions :**
- **Supprimer l'utilisateur et recommencer** :
  ```powershell
  rm dataset/<nomutilisateur> -r
  ```
- **Puis recapturer** en faisant **exactement** les poses demandées :
  - Frontale : regard droit devant
  - Gauche : tourne la tête maximal à gauche
  - Droite : tourne la tête maximal à droite
  - Haut : lève la tête maximale
  - Bas : baisse la tête maximale

## Modification Apportée

Nous avons maintenant :
- **Tolérances adaptatives** par pose (gauche/droite ont yaw_tol=25° vs frontale yaw_tol=15°)
- **Affichage détaillé** des écarts dans le GUI et les scripts de test
- **Script de diagnostic** pour investiguer sans toucher à l'interface

## Étapes Recommandées

1. Lance `python pose_diagnostic.py` → affiche les valeurs et start le moniteur
2. Regarde les écarts en temps réel quand tu refais les poses
3. Si ok → lance `python test_auth_diagnostic.py` → teste l'authentification
4. Regarde les logs détaillés pour voir où ça échoue
5. Ajuste selon le cas ci-dessus

## Logs Détaillés Maintenant Affichés

Au démarrage du GUI (`python face_pose_estimation.py`) :
```
=== DIAGNOSTIC POSES APPRISES ===
alice:
  frontale: yaw=-2.5, pitch=1.3, roll=-0.8
  gauche: yaw=-35.2, pitch=2.1, roll=-1.5
  ...
```

Pendant l'authentification (voir console) :
```
gauche | Y:-34.8 vs -35.2 (err=0.4, tol=25) | P:2.1 vs 2.1 (err=0.0, tol=15) | R:-1.5 vs -1.5 (err=0.0, tol=15)
```
