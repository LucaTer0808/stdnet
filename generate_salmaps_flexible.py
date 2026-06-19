"""
Universal Saliency Map Generator for SDNet
Author: Zhuo Su / Refactored for direct folder inference
Date: June 2026
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

import torch
import torch.nn.functional as F
from torchvision import transforms

import models
from utils import state_from_training

def main():
    parser = argparse.ArgumentParser(description='Universal SDNet Saliency Map Generator')
    
    # NUR noch source und target als Argumente
    parser.add_argument('--source', type=str, required=True, 
                        help="Pfad zum Ordner mit den Eingabebildern")
    parser.add_argument('--target', type=str, required=True, 
                        help="Pfad zum Ordner, in dem die Ergebnisse landen sollen")

    args = parser.parse_args()

    # Feste Parameter aus dem Tutorial im Hintergrund setzen
    args.model = "sdneta"
    args.bn = True
    args.inference_config = "baseline"
    args.train_config = "sdnet-a"
    args.gpu = "0"
    args.evaluate = "checkpoints/sdneta_from_pretrained.pth"
    img_size = 384  # Die Zielgröße für das Modell aus dem Tutorial

    # GPU/CPU Gerät zuweisen
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Gerät: {device}")

    ### 1. Modelle erstellen & auf Device schieben
    args.config = args.train_config
    model_training_time = getattr(models, args.model)(args).to(device)

    args.config = args.inference_config
    args.bn = False
    model_inference_time = getattr(models, args.model)(args).to(device)

    ### 2. Gewichte laden
    print(f"=> Lade Checkpoint aus '{args.evaluate}'")
    checkpoint_dict = torch.load(args.evaluate, map_location=device)
    model_training_time.load_state_dict(checkpoint_dict)

    ### 3. Reparametrisierung (DCR)
    if hasattr(model_training_time, 'module'):
        model_training_time.module.reparameterize()
    else:
        model_training_time.reparameterize()
        
    state_from_training(model_training_time, model_inference_time)
    print("=> Reparametrisierung erfolgreich abgeschlossen.")

    ### 4. Direkte Verarbeitung aller Bilder im Ordner
    process_folder(model_inference_time, device, args.source, args.target, img_size)


def process_folder(model, device, source_dir, target_dir, img_size):
    model.eval()
    
    src_path = Path(source_dir)
    tgt_path = Path(target_dir)

    shutil.rmtree(target_dir, ignore_errors=True)
    Path(target_dir).mkdir(parents=True, exist_ok=False)

    tgt_path.mkdir(parents=True, exists_ok=False)

    # Erlaubte Bildendungen definieren (Groß-/Kleinschreibung wird ignoriert)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Alle Dateien im Ordner finden, die ein Bild sind
    img_files = [f for f in src_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not img_files:
        print(f"❌ Keine unterstützten Bilder in '{source_dir}' gefunden!")
        return

    print(f"\n📸 {len(img_files)} Bilder gefunden. Starte Generierung...")
    print(f"📂 Ergebnisse werden gespeichert in: {tgt_path}\n")

    # PyTorch-Transformation für die Bilder (Normalisierung & Resize auf 384x384)
    # Hinweis: Da ich die originale dataloader_sod nicht kenne, nutzen wir Standard ImageNet-Werte.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.inference_mode(): # Moderner, schneller Modus
        for idx, img_path in enumerate(img_files):
            # Bild laden
            orig_img = Image.open(img_path).convert('RGB')
            orig_size = orig_img.size # (Breite, Höhe) für das spätere Zurückskalieren
            
            # Transformieren und Batch-Dimension hinzufügen [1, 3, 384, 384]
            image_tensor = transform(orig_img).unsqueeze(0).to(device)
            
            # Modell-Inferenz
            result = model(image_tensor)
            
            # Saliency Map wieder auf die Originalgröße des Bildes hochskalieren
            result = F.interpolate(result, mode='bilinear', size=(orig_size[1], orig_size[0]), align_corners=False)
            
            # Werte in numpy-Array umwandeln
            result = torch.squeeze(result).cpu().numpy()
            
            # Pixelwerte von [0, 1] auf [0, 255] bringen
            # Falls die Ausgabebilder zu dunkel/hell sind, hier evtl. "result = 1 / (1 + np.exp(-result))" (Sigmoid) einfügen
            result_scaled = (result * 255).astype(np.uint8)
            result_img = Image.fromarray(result_scaled)
            
            # Speichern unter exakt gleichem Namen im Zielordner (als .png)
            save_path = tgt_path / f"{img_path.stem}.png"
            result_img.save(save_path)
            
            if (idx + 1) % 20 == 0 or (idx + 1) == len(img_files):
                print(f"Fortschritt: [{idx + 1}/{len(img_files)}] Bilder verarbeitet.")

    print("\n🎉 Fertig! Alle Saliency Maps wurden erfolgreich erstellt.")

if __name__ == '__main__':
    main()
