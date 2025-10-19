import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

# --- AYARLAR ---
NPZ_FILE = "image_chips_labels_50m_balanced.npz"
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

print("🔄 DENGELENMİŞ VERİ İLE MODEL EĞİTİMİ...")

# --- 1. Veriyi Yükleme ---
print(f"📦 Veri yükleniyor: {NPZ_FILE}")
try:
    data = np.load(NPZ_FILE)
    X = data['X']
    y = data['y']
    print(f"✅ Veri şekli: {X.shape}")
    print(f"📊 Etiket dağılımı: {dict(zip(*np.unique(y, return_counts=True)))}")
except Exception as e:
    print(f"❌ HATA: NPZ dosyası yüklenirken: {e}")
    exit()

# --- 2. Veriyi Normalleştirme ---
print("\n🎛️  Normalizasyon uygulanıyor...")
X = X.astype('float32')

if np.max(X) > 1.0:
    if np.max(X) <= 10000:
        X = X / 10000.0
        print("✅ 10000'e bölünerek normalizasyon")
    else:
        X = X / np.max(X)
        print("✅ Max değere bölünerek normalizasyon")
else:
    print("✅ Zaten normalleştirilmiş")

print(f"📊 Normalize edilmiş Min: {np.min(X):.3f}, Max: {np.max(X):.3f}")

# --- 3. Veriyi Eğitim ve Test Setlerine Ayırma ---
print(f"\n🔀 Veri %{int((1-TEST_SPLIT_RATIO)*100)} eğitim, %{int(TEST_SPLIT_RATIO*100)} test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"📊 Eğitim seti: {X_train.shape}")
print(f"📊 Test seti: {X_test.shape}")

# --- 4. CNN Modelini Tanımlama ---
print("\n🧠 Model oluşturuluyor...")
input_shape = X_train.shape[1:]

model = keras.Sequential([
    keras.Input(shape=input_shape),
    
    # 1. Konvolüsyon Katmanı
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # 2. Konvolüsyon Katmanı
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # Global Pooling
    keras.layers.GlobalAveragePooling2D(),
    
    # Tam Bağlı Katmanlar
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    # Çıktı Katmanı
    keras.layers.Dense(1, activation='sigmoid')
])

# --- 5. Class Weight Hesaplama ---
print("\n⚖️ Sınıf ağırlıkları hesaplanıyor...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"📊 Class weights: {class_weight_dict}")

# --- 6. Modeli Derleme ---
print("\n🔧 Model derleniyor...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

model.summary()

# --- 7. Callbacks Tanımlama ---
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        monitor='val_loss'
    )
]

# --- 8. Modeli Eğitme ---
print("\n🎯 MODEL EĞİTİMİ BAŞLIYOR...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# --- 9. Model Performansını Değerlendirme ---
print("\n📊 MODEL DEĞERLENDİRİLİYOR...")
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Test Kaybı (Loss): {test_loss:.4f}")
print(f"✅ Test Doğruluğu (Accuracy): {test_accuracy:.4f}")
print(f"✅ Test Kesinlik (Precision): {test_precision:.4f}")
print(f"✅ Test Duyarlılık (Recall): {test_recall:.4f}")
print(f"✅ Test AUC: {test_auc:.4f}")

# --- 10. Detaylı Metrikler ---
print("\n📈 SINIFLANDIRMA RAPORU:")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype("int32")

print(classification_report(y_test, y_pred, target_names=["Akasya Yok (0)", "Akasya Var (1)"]))

print("\n🔢 KARMAŞIKLIK MATRİSİ:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- 11. PDF Skoru Hesaplama ---
print("\n📝 PDF PUAN HESAPLAMASI:")
TN, FP, FN, TP = cm.ravel()

pdf_score = 200 * (TP * 0.5 - FN * 0.3 - FP * 0.2) / (TP + FN + FP)
print(f"📊 PDF Skoru: {pdf_score:.2f}")

# --- 12. Eğitim Geçmişini Görselleştirme ---
try:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Doğruluk')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Validasyon Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Kayıp')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Eğitim Precision')
    plt.plot(history.history['val_precision'], label='Validasyon Precision')
    plt.plot(history.history['recall'], label='Eğitim Recall')
    plt.plot(history.history['val_recall'], label='Validasyon Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Değer')
    plt.legend()
    plt.title('Precision & Recall')
    
    plt.tight_layout()
    plt.savefig("training_history_balanced.png", dpi=150, bbox_inches='tight')
    print("📊 Eğitim geçmişi 'training_history_balanced.png' olarak kaydedildi.")
except Exception as e:
    print(f"⚠️ Grafik kaydedilemedi: {e}")

# --- 13. Modeli Kaydetme ---
model.save("acacia_detector_balanced.h5")
print("💾 Model 'acacia_detector_balanced.h5' olarak kaydedildi.")

print("\n🎉 TÜM İŞLEMLER TAMAMLANDI!")
