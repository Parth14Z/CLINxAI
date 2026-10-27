import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from .config import IMAGE_SIZE

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def image_to_array(image: Image.Image):
    return np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.uint8)

def predict_batch(images, model, device):
    """Prediction callback used by LIME and SHAP. Input is RGB uint8 images."""
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensors = torch.stack([prep(np.asarray(im).astype(np.uint8)) for im in images])
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(tensors.to(device)), dim=1).cpu().numpy()

def make_lime_explanation(image_array, model, device, num_samples=500):
    explainer = lime_image.LimeImageExplainer()
    predict_fn = lambda batch: predict_batch(batch, model, device)
    exp = explainer.explain_instance(
        image_array, predict_fn, top_labels=1, hide_color=0,
        num_samples=num_samples
    )
    label = int(exp.top_labels[0])
    temp, mask = exp.get_image_and_mask(
        label, positive_only=False, num_features=8, hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask), label

def make_shap_explanation(image_array, model, device, class_names, max_evals=100):
    """Image SHAP can be slow. Increase max_evals only after the first test works."""
    predict_fn = lambda batch: predict_batch(batch, model, device)
    masker = shap.maskers.Image("blur(32,32)", image_array.shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=class_names)
    return explainer(np.expand_dims(image_array, axis=0), max_evals=max_evals, batch_size=8)
