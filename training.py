import json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from .config import SPLIT_DIR, MODEL_DIR, OUTPUT_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from .models import build_model, MODEL_NAMES

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }
    try:
        if num_classes == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, np.array(y_prob)[:, 1]))
        else:
            result["roc_auc_ovr_macro"] = float(roc_auc_score(
                y_true, np.array(y_prob), multi_class="ovr", average="macro"
            ))
    except ValueError:
        result["roc_auc_note"] = "Could not calculate AUC; check class coverage."
    return result

def train_one(name, train_loader, val_loader, class_names, device, epochs=EPOCHS):
    model = build_model(name, len(class_names), pretrained=(name != "custom_cnn")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
                count += x.size(0)
        val_loss /= max(count, 1)
        print(f"{name} | epoch {epoch+1}/{epochs} | "
              f"train_loss={running_loss/max(len(train_loader.dataset),1):.4f} "
              f"val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_name": name, "state_dict": model.state_dict(),
                "class_names": class_names, "image_size": IMAGE_SIZE,
            }, MODEL_DIR / f"{name}.pt")
    print(f"Saved best checkpoint: {MODEL_DIR / (name + '.pt')}")

def main():
    seed_everything()
    if not (SPLIT_DIR / "train").exists():
        raise FileNotFoundError("Run `python prepare_data.py` first.")
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(SPLIT_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(SPLIT_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(SPLIT_DIR / "test", transform=eval_tf)
    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train/validation/test class names do not match.")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Classes:", train_ds.classes)

    results = {}
    for name in MODEL_NAMES:
        train_one(name, train_loader, val_loader, train_ds.classes, device)
        checkpoint = torch.load(MODEL_DIR / f"{name}.pt", map_location=device, weights_only=False)
        model = build_model(name, len(train_ds.classes), pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        results[name] = evaluate(model, test_loader, device, len(train_ds.classes))
        print(f"TEST RESULTS — {name}: {json.dumps(results[name], indent=2)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {OUTPUT_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
