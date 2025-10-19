import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# --- AYARLAR ---
MODEL_FILE = "acacia_detector_balanced.h5"
INPUT_IMAGE_FILE = "deneme3.tiff"
OUTPUT_MASK_FILE = "acacia_mask_final.tiff"

def find_optimal_threshold(model, X_val, y_val):
    """Optimal threshold'u bul"""
    print("🎯 Optimal threshold hesaplanıyor...")
    y_pred_proba = model.predict(X_val, verbose=0)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"✅ Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

print("🧠 Eğitilmiş model yükleniyor...")
try:
    model = keras.models.load_model(MODEL_FILE)
    input_shape = model.input_shape
    print(f"✅ Model başarıyla yüklendi. Girdi şekli: {input_shape[1:]}")
except Exception as e:
    print(f"❌ Model yüklenirken hata: {e}")
    exit()

# Optimal threshold için validation seti oluştur
print("📊 Validation seti hazırlanıyor...")
try:
    data = np.load("image_chips_labels_50m_balanced.npz")
    X_val = data['X'][:1000].astype('float32')  # İlk 1000 örnek
    y_val = data['y'][:1000]
    
    if np.max(X_val) > 1.0:
        X_val = X_val / 10000.0
    
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)
except:
    print("⚠️ Validation seti bulunamadı, default threshold kullanılıyor: 0.5")
    optimal_threshold = 0.5

print(f"\n🖼️ İşlenecek görüntü açılıyor: {INPUT_IMAGE_FILE}")
try:
    with rasterio.open(INPUT_IMAGE_FILE) as src:
        img = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    
    print(f"📷 Görüntü Boyutları (Bant,Y,X): {img.shape}")
except Exception as e:
    print(f"❌ Görüntü yüklenirken hata: {e}")
    exit()

# Görüntüyü işle
img_processed = np.transpose(img, (1, 2, 0))  # (Y, X, Bant)
img_normalized = img_processed.astype('float32')

# Normalizasyon
if np.max(img_normalized) > 1.0:
    if np.max(img_normalized) <= 10000:
        img_normalized = img_normalized / 10000.0
    else:
        img_normalized = img_normalized / np.max(img_normalized)

print(f"🎛️  Normalizasyon - Min: {np.min(img_normalized):.3f}, Max: {np.max(img_normalized):.3f}")

# Model parametreleri
CHIP_SIZE = input_shape[1]
stride = CHIP_SIZE  # Overlap yok
height, width, bands = img_normalized.shape

print(f"🔍 Görüntü {height}x{width} -> {CHIP_SIZE}x{CHIP_SIZE} çiplere ayrılıyor...")
print(f"📐 Çip boyutu: {CHIP_SIZE}x{CHIP_SIZE}, Adım: {stride}")

# Hesaplamalar
num_chips_y = (height - CHIP_SIZE) // stride + 1
num_chips_x = (width - CHIP_SIZE) // stride + 1
total_chips = num_chips_y * num_chips_x

print(f"🔢 Toplam çip sayısı: {num_chips_y} x {num_chips_x} = {total_chips}")

# Maske ve tahmin dizileri
mask = np.zeros((height, width), dtype=np.uint8)
probability_map = np.zeros((height, width), dtype=np.float32)
count_map = np.zeros((height, width), dtype=np.uint8)  # Kaç kere tahmin yapıldığını takip et

print("🎯 TAHMİN YAPILIYOR...")

chip_count = 0
predictions_list = []

for i in tqdm(range(0, height - CHIP_SIZE + 1, stride), desc="Satırlar"):
    for j in range(0, width - CHIP_SIZE + 1, stride):
        try:
            # Çipi al
            chip = img_normalized[i:i+CHIP_SIZE, j:j+CHIP_SIZE, :]
            
            # Boyut kontrolü
            if chip.shape[0] != CHIP_SIZE or chip.shape[1] != CHIP_SIZE:
                continue
            
            # Batch dimension ekle ve tahmin yap
            chip_batch = np.expand_dims(chip, axis=0)
            prediction = model.predict(chip_batch, verbose=0)[0][0]
            
            predictions_list.append(prediction)
            
            # Probability map'i güncelle
            probability_map[i:i+CHIP_SIZE, j:j+CHIP_SIZE] += prediction
            count_map[i:i+CHIP_SIZE, j:j+CHIP_SIZE] += 1
            
            # Binary maske için threshold uygula
            if prediction > optimal_threshold:
                mask[i:i+CHIP_SIZE, j:j+CHIP_SIZE] = 255
                
            chip_count += 1
            
        except Exception as e:
            continue

# Ortalama probability hesapla
probability_map = np.divide(probability_map, count_map, where=count_map>0)

print(f"\n📊 TAHMİN İSTATİSTİKLERİ:")
predictions_array = np.array(predictions_list)
print(f"   Min Olasılık: {predictions_array.min():.4f}")
print(f"   Max Olasılık: {predictions_array.max():.4f}")
print(f"   Ortalama Olasılık: {predictions_array.mean():.4f}")
print(f"   Standart Sapma: {predictions_array.std():.4f}")
print(f"   {optimal_threshold}'dan büyük olanlar: {np.sum(predictions_array > optimal_threshold)} / {len(predictions_array)}")


#mask = 255 - mask
# Maske kaydet
print(f"\n💾 Maske kaydediliyor: {OUTPUT_MASK_FILE}")
meta.update({
    "count": 1,
    "dtype": "uint8",
    "compress": 'lzw'
})

try:
    with rasterio.open(OUTPUT_MASK_FILE, 'w', **meta) as dst:
        dst.write(mask, 1)
    print("✅ Maske başarıyla kaydedildi")
except Exception as e:
    print(f"❌ Maske kaydedilirken hata: {e}")

# Görselleştirme
print("📊 Görselleştirme oluşturuluyor...")
try:
    plt.figure(figsize=(20, 5))
    
    # 1. Orijinal Görüntü
    plt.subplot(1, 4, 1)
    # RGB kanallarını göster (ilk 3 band)
    display_img = img_normalized[..., :3]
    # Kontrast iyileştirme
    p2, p98 = np.percentile(display_img, (2, 98))
    display_img_enhanced = np.clip((display_img - p2) / (p98 - p2), 0, 1)
    plt.imshow(display_img_enhanced)
    plt.title('Orijinal Görüntü (RGB)')
    plt.axis('off')
    
    # 2. Probability Map
    plt.subplot(1, 4, 2)
    plt.imshow(probability_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Akasya Olasılığı')
    plt.title('Probability Map')
    plt.axis('off')
    
    # 3. Binary Maske
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Binary Maske (Threshold: {optimal_threshold:.2f})')
    plt.axis('off')
    
    # 4. Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(display_img_enhanced)
    plt.imshow(mask, cmap='Reds', alpha=0.3)
    plt.title('Overlay (Kırmızı = Akasya)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization_final1.png', dpi=150, bbox_inches='tight')
    print("✅ Görsel 'prediction_visualization_final1.png' kaydedildi")
    
except Exception as e:
    print(f"⚠️ Görselleştirme oluşturulamadı: {e}")

print(f"\n🎉 TAHMİN TAMAMLANDI!")
print(f"📁 Maske dosyası: {OUTPUT_MASK_FILE}")
print(f"📁 Görsel dosyası: prediction_visualization_final2.png")
print(f"🎯 Kullanılan threshold: {optimal_threshold:.2f}")
