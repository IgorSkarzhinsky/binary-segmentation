import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

def predict_segmentation(image_path, model_path='segmentation_model.pth'):
    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Загрузка и подготовка изображения
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        pred = model(img_tensor)
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    # Преобразование 0/1 в 0/255 для визуализации
    mask_vis = (mask * 255).astype(np.uint8)
    return mask_vis

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mask = predict_segmentation(sys.argv[1])
        cv2.imwrite('predicted_mask.png', mask)
        print("Маска сохранена как predicted_mask.png")