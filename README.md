

A binary image classifier that takes a leaf photo and decides whether it is an **arecanut (areca palm) leaf**. Built with transfer learning (MobileNetV2) in Google Colab using TensorFlow/Keras.

> **Status:** works well on arecanut, banana, mango, neem, jackfruit, guava and date palm leaves. **It is not yet reliable on coconut leaves**, which look very similar to arecanut and were not in the training data. See [Known limitations](#known-limitations).

---

## How the model works

- **Base:** MobileNetV2 pretrained on ImageNet, frozen (feature extraction).
- **Head (trained):** GlobalAveragePooling2D, Dropout(0.3), Dense(1, sigmoid).
- **Built into the model:** data augmentation (active only in training) and pixel rescaling, so the model takes **raw images (0 to 255, 224×224)**.
- **Output:** one number, `P(Not_Arecanut)`.

| P(Not_Arecanut) | Decision |
|---|---|
| below 0.35 | Arecanut |
| above 0.65 | Not Arecanut |
| 0.35 to 0.65 | Uncertain (the model is not sure) |

**Training setup:** Adam (learning rate 1e-3), binary cross-entropy, batch size 32, class weights to balance the classes, up to 25 epochs with early stopping on validation loss. Augmentation (applied equally to both classes): flips, rotation with reflect fill, zoom, brightness and contrast.

---

## Data

The image dataset is **not included** in this repository. Class names and counts are described below.

| Class | Source folders | Images used |
|---|---|---|
| Arecanut | Project arecanut leaf set: 604 files, which are augmented copies of only **83 original photos** | 295 after cleaning |
| Not Arecanut | Banana (250), mango (100), neem (100), jackfruit (100), guava (100), date palm (150) sampled from the project's leaf collections | 754 after cleaning |

### Split (final model, v3)

| Split | Arecanut | Not Arecanut | Total |
|---|---|---|---|
| Train | 169 | 450 | 619 |
| Validation | 54 | 151 | 205 |
| Test | 72 | 153 | 225 |

Two precautions were needed to make this split trustworthy (see [Lessons learned](#lessons-learned)):

1. **Split by original photo.** All augmented copies of one photo stay in the same split, so no test image is a copy of a training image.
2. **Black-corner images removed from both classes.** Rotated Arecanut images had black corners that the model could use as a shortcut.

---

## Results

### Internal test set (225 images from the same source folders as training)

| Metric | Result |
|---|---|
| Accuracy | 1.00 (225 of 225) |
| Precision / recall / F1 (both classes) | 1.00 |
| Confusion matrix | `[[72, 0], [0, 153]]` |

These scores are **optimistic**: the test images come from the same source folders as the training images, so they are an internal check, not a measure of real-world accuracy. See `results/v3/` for the confusion matrix, ROC curve and classification report.

### External photos (not from any dataset folder)

| Photo | True class | Prediction | P(Not_Arecanut) |
|---|---|---|---|
| Banana leaves, black background | Not Arecanut | Not Arecanut | 0.97 |
| Same, background changed to white | Not Arecanut | Not Arecanut | 0.98 |
| Spotted arecanut leaf (web photo) | Arecanut | Arecanut | 0.02 |
| Arecanut leaves from above (web photo) | Arecanut | Arecanut | 0.0001 |
| Arecanut leaves from below (web photo) | Arecanut | Arecanut | 0.003 |
| Spotted arecanut leaf (test image) | Arecanut | Arecanut | 0.0000 |
| **Coconut leaf (test image)** | Not Arecanut | **Arecanut (wrong)** | **0.006** |

Only 7 external photos have been checked, so this is evidence that the model works, not a measured accuracy.

---

## Lessons learned

The first model (v1: Arecanut vs Banana, random split) scored **100% on its test set** and then labelled a clean banana-leaf photo as Arecanut. Investigation found two causes:

- **Data leakage:** the 604 Arecanut files were copies of 83 original photos, spread randomly across train, validation and test. The model was tested on photos it had effectively already seen.
- **Shortcut learning:** about 55% of Arecanut training images had black rotation corners against about 6% of the other class, and Arecanut images included white and studio backgrounds while all Banana images were messy field photos. The model could score perfectly by learning image style instead of leaf shape.

Fixes: split by original photo, remove black-corner images, use a wider "not arecanut" class, apply the same augmentation to both classes, add class weights and early stopping. The result is v3, the model in this repository.

---

## Known limitations

- **Coconut and other look-alike palms:** not in the training data. A coconut leaf was wrongly labelled Arecanut with high confidence. Next step: add 150 or more coconut and other palm leaf photos to the negative class and retrain.
- **Small Arecanut set:** only about 49 original Arecanut photos in training.
- **Closed two-class model:** it cannot answer "this is not a leaf". Any photo gets an Arecanut or Not Arecanut answer, and a high confidence number does not guarantee a correct one.
- **Best on a single leaf or frond filling most of the photo**, less so on whole trees or distant shots.

---

## Repository contents

```
.
├── README.md
├── arecanut_leaf_classifier.ipynb   # full workflow: data split, training, evaluation
├── models/
│   └── arecanut_vs_not_v3.keras     # final model (v3)
└── results/
    ├── v3/                          # confusion matrix, ROC curve, classification report
    └── v1/                          # curves and Grad-CAM from the first model (for comparison)
```

The files in `results/v1/` come from the first model and show its inflated scores. They are kept only to document the project history.

---

## Using the model

```python
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('models/arecanut_vs_not_v3.keras')

img = Image.open('leaf.jpg').convert('RGB').resize((224, 224))
arr = np.expand_dims(np.array(img, dtype=np.float32), 0)   # raw 0-255, the model rescales itself
p = float(model.predict(arr, verbose=0)[0][0])             # P(Not_Arecanut)

if p < 0.35:
    print('Arecanut', 1 - p)
elif p > 0.65:
    print('Not Arecanut', p)
else:
    print('Uncertain', p)
```

Requires `tensorflow`, `numpy` and `pillow`.
