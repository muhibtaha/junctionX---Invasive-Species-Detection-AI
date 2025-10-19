import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np
import os
import warnings
from math import ceil
from tqdm import tqdm
from sklearn.utils import resample

# Uyarıları bastır
warnings.filterwarnings("ignore", category=UserWarning, module='rasterio')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- AYARLAR ---
FILTERED_ACACIA_FILE = "2018_acacia.gpkg"
SATELLITE_IMAGE_FILE = "deneme3.tiff"  # Sizin kullandığınız dosya
GRID_SIZE_M = 50
OUTPUT_NPZ_FILE = f"image_chips_labels_{GRID_SIZE_M}m_balanced.npz"
METRIC_CRS = "EPSG:3763"

def balance_dataset(X, y):
    """Veri setini dengeler - oversampling ile"""
    if len(X) == 0:
        return X, y
        
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    y_0 = y[y == 0]
    y_1 = y[y == 1]
    
    print(f"⏳ Dengeleme öncesi: Yok={len(X_0)}, Var={len(X_1)}")
    
    if len(X_1) == 0:
        print("⚠️  Hiç Akasya Var örneği yok!")
        return X, y
    
    target_samples = min(len(X_0) // 3, len(X_1) * 5)
    if target_samples < len(X_1):
        target_samples = len(X_1)  # En az mevcut kadar
    
    X_1_balanced, y_1_balanced = resample(
        X_1, y_1,
        replace=True,
        n_samples=target_samples,
        random_state=42
    )
    
    X_balanced = np.concatenate([X_0, X_1_balanced])
    y_balanced = np.concatenate([y_0, y_1_balanced])
    
    print(f"✅ Dengeleme sonrası: Yok={len(X_0)}, Var={len(X_1_balanced)}")
    print(f"📊 Yeni oran: {len(X_0)}:{len(X_1_balanced)} ≈ {len(X_0)/len(X_1_balanced):.1f}:1")
    
    return X_balanced, y_balanced

print("🚀 VERİ HAZIRLAMA BAŞLIYOR...")

# --- 1. Uydu Görüntüsünü ve Akasya Poligonlarını Yükle ---
print(f"📡 Uydu görüntüsü yükleniyor: {SATELLITE_IMAGE_FILE}")

try:
    with rasterio.open(SATELLITE_IMAGE_FILE) as src:
        image_transform = src.transform
        image_crs = src.crs
        image_bounds = src.bounds
        image_profile = src.profile.copy()
        
        # PİKSEL BOYUTUNU DOĞRU HESAPLA - KRİTİK DÜZELTME!
        # Transform: (a, b, c, d, e, f) 
        # a: x yönünde piksel boyutu, e: y yönünde piksel boyutu (genellikle negatif)
        pixel_size_x = abs(image_transform[0])
        pixel_size_y = abs(image_transform[4])
        
        print(f"📏 Gerçek piksel boyutları: {pixel_size_x:.6f} x {pixel_size_y:.6f} derece")
        print(f"🌍 Görüntü CRS: {image_crs}")
        print(f"📐 Görüntü boyutları: {src.width} x {src.height} piksel")
        print(f"🗺️  Görüntü sınırları: {image_bounds}")

except Exception as e:
    print(f"❌ Görüntü yüklenirken hata: {e}")
    exit()

print(f"\n🌳 Filtrelenmiş Akasya poligonları yükleniyor: {FILTERED_ACACIA_FILE}")

try:
    acacia_polygons_gdf_orig = gpd.read_file(FILTERED_ACACIA_FILE)
    
    if acacia_polygons_gdf_orig.empty:
        print("⚠️  UYARI: Yüklenen Akasya dosyası boş.")
        acacia_polygons_gdf = gpd.GeoDataFrame(geometry=[], crs=image_crs)
    else:
        print(f"✅ {len(acacia_polygons_gdf_orig)} Akasya poligonu yüklendi.")
        print(f"📌 Akasya CRS: {acacia_polygons_gdf_orig.crs}")
        
        # CRS dönüşümü
        if acacia_polygons_gdf_orig.crs != image_crs:
            print("🔄 Akasya poligonları görüntü CRS'ine dönüştürülüyor...")
            acacia_polygons_gdf = acacia_polygons_gdf_orig.to_crs(image_crs)
        else:
            acacia_polygons_gdf = acacia_polygons_gdf_orig

        # Spatial index oluştur
        if acacia_polygons_gdf.sindex is None:
            acacia_polygons_gdf.sindex.create_index()
        print("📊 Spatial index oluşturuldu.")

except Exception as e:
    print(f"❌ Akasya dosyası yüklenirken hata: {e}")
    exit()

# --- 2. Grid Oluşturma --- GÜNCEL TÜRKÇE KARAKTERSİZ
print(f"\n🔲 {GRID_SIZE_M}x{GRID_SIZE_M}m grid olusturuluyor...")

# ÖNCE grid oluşturma için görüntüyü metriğe dönüştür
try:
    temp_gdf = gpd.GeoDataFrame([1], geometry=[box(*image_bounds)], crs=image_crs)
    temp_gdf_proj = temp_gdf.to_crs(METRIC_CRS)
    minx, miny, maxx, maxy = temp_gdf_proj.total_bounds
    
    print(f"📏 Metrik CRS'de sınırlar: {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}")

    start_x = np.floor(minx / GRID_SIZE_M) * GRID_SIZE_M
    start_y = np.floor(miny / GRID_SIZE_M) * GRID_SIZE_M
    end_x = np.ceil(maxx / GRID_SIZE_M) * GRID_SIZE_M
    end_y = np.ceil(maxy / GRID_SIZE_M) * GRID_SIZE_M

    x_coords = np.arange(start_x, end_x, GRID_SIZE_M)
    y_coords = np.arange(start_y, end_y, GRID_SIZE_M)

    grid_cells_proj = []
    original_area_geom = temp_gdf_proj.geometry.iloc[0]

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            poly = box(x_coords[i], y_coords[j], x_coords[i] + GRID_SIZE_M, y_coords[j] + GRID_SIZE_M)
            if original_area_geom.intersects(poly):
                clipped_poly = original_area_geom.intersection(poly)
                if not clipped_poly.is_empty and clipped_poly.geom_type == 'Polygon':
                    grid_cells_proj.append(clipped_poly)

    grid_gdf_proj = gpd.GeoDataFrame(geometry=grid_cells_proj, crs=METRIC_CRS)
    grid_gdf = grid_gdf_proj.to_crs(image_crs)  # Görüntü CRS'ine dönüştür
    grid_gdf['grid_id'] = range(len(grid_gdf))
    print(f"✅ {len(grid_gdf)} adet grid hucresi olusturuldu.")

except Exception as e:
    print(f"❌ Grid olusturulurken hata: {e}")
    exit()

# --- 3. Kare Kare Etiketleme ve Kirpma --- GÜNCEL TÜRKÇE KARAKTERSİZ
print("\n🔍 Grid hücreleri isleniyor...")
image_chips = []
labels = []

# KRİTİK DÜZELTME: Piksel boyutunu doğru hesapla
# Görüntü derece cinsindeyse, metreye çevirmemiz lazım
# Yaklaşık: 1 derece ≈ 111,000 metre

try:
    # Dereceyi metreye çevirmek için yaklaşık faktör
    DEGREE_TO_METERS = 111000
    
    # Piksel boyutunu metre cinsinden hesapla
    pixel_size_x_meters = pixel_size_x * DEGREE_TO_METERS
    pixel_size_y_meters = pixel_size_y * DEGREE_TO_METERS
    
    print(f"📐 Metre cinsinden piksel boyutlari: {pixel_size_x_meters:.2f} x {pixel_size_y_meters:.2f}m")
    
    # Grid boyutuna göre piksel sayısını hesapla
    target_width_px = max(1, ceil(GRID_SIZE_M / pixel_size_x_meters))
    target_height_px = max(1, ceil(GRID_SIZE_M / pixel_size_y_meters))
    
    num_bands = image_profile['count']
    target_chip_shape = (target_height_px, target_width_px, num_bands)
    print(f"🎯 Hedef Chip Boyutu: {target_chip_shape}")

    if target_width_px > 100 or target_height_px > 100:
        print("⚠️  UYARI: Chip boyutlari cok buyuk! Grid boyutunu kucultun.")
        # Acil onlem: Sabit boyut kullan
        target_width_px = 10
        target_height_px = 10
        print(f"🔧 Zorunlu yeni boyut: {target_height_px}x{target_width_px}")

except Exception as e:
    print(f"❌ Boyut hesaplama hatasi: {e}")
    # Acil onlem: Sabit boyut
    target_width_px = 10
    target_height_px = 10
    target_chip_shape = (target_height_px, target_width_px, 3)
    print(f"🔧 Sabit boyut kullaniliyor: {target_chip_shape}")

# Görüntüyü aç ve işle
try:
    with rasterio.open(SATELLITE_IMAGE_FILE) as src:
        for index, grid_cell in tqdm(grid_gdf.iterrows(), total=min(1000, len(grid_gdf)), desc="Hücreler"):  # Sadece ilk 1000 için
            has_acacia_label = 0
            cell_geom = grid_cell.geometry

            # 1. Etiketleme
            if not acacia_polygons_gdf.empty:
                try:
                    possible_matches_indices = list(acacia_polygons_gdf.sindex.intersection(cell_geom.bounds))
                    if possible_matches_indices:
                        possible_matches = acacia_polygons_gdf.iloc[possible_matches_indices]
                        for _, acacia_poly in possible_matches.iterrows():
                            if cell_geom.intersects(acacia_poly.geometry):
                                has_acacia_label = 1
                                break
                except Exception:
                    pass

            # 2. Görüntü Kirpma
            try:
                out_image, out_transform = mask(src, [cell_geom], crop=True, all_touched=True, nodata=0)

                if out_image.size == 0 or np.all(out_image == 0):
                    continue

                # Boyut kontrolü ve padding
                h, w = out_image.shape[1], out_image.shape[2]
                
                # Eğer çip çok büyükse, atla
                if h > target_height_px * 3 or w > target_width_px * 3:
                    continue
                
                pad_h = target_height_px - h
                pad_w = target_width_px - w

                # Hedef boyuttan büyükse kırp
                if pad_h < 0:
                    out_image = out_image[:, :target_height_px, :]
                    pad_h = 0
                if pad_w < 0:
                    out_image = out_image[:, :, :target_width_px]
                    pad_w = 0

                # Padding uygula
                if pad_h > 0 or pad_w > 0:
                    padded_image = np.pad(out_image, ((0, 0), (0, pad_h), (0, pad_w)), 
                                        mode='constant', constant_values=0)
                else:
                    padded_image = out_image

                # Transpose ve kaydet
                chip_transposed = np.transpose(padded_image, (1, 2, 0))
                
                # Son boyut kontrolü
                if chip_transposed.shape[0] == target_height_px and chip_transposed.shape[1] == target_width_px:
                    image_chips.append(chip_transposed)
                    labels.append(has_acacia_label)

            except Exception as e:
                continue

except Exception as e:
    print(f"❌ Görüntü isleme hatasi: {e}")

print(f"\n✅ {len(image_chips)} adet gorsuntu chip'i olusturuldu.")

if not image_chips:
    print("❌ HATA: Hicbir gorsuntu chip'i olusturulamadi!")
    print("🔍 Olası nedenler:")
    print("   - Grid boyutu cok buyuk")
    print("   - Gorsuntu CRS'i ve grid CRS'i uyumsuz")
    print("   - Gorsuntu dosyasi bozuk")
    exit()

X = np.array(image_chips, dtype=np.float32)
y = np.array(labels, dtype=np.uint8)

print("📊 Olusturulan Veri Boyutlari:")
print(f"  Gorsuntu Chipleri (X): {X.shape}")
print(f"  Etiketler (y): {y.shape}")

label_counts = dict(zip(*np.unique(y, return_counts=True)))
print(f"  Etiket Dagitimi: {label_counts}")

# --- VERİ DENGESİZLİĞİNİ DÜZELT ---
print("\n🔄 VERI DENGELENIYOR...")
X_balanced, y_balanced = balance_dataset(X, y)

# Veriyi kaydet
print(f"\n💾 Dengelenmis veri '{OUTPUT_NPZ_FILE}' dosyasina kaydediliyor...")
np.savez_compressed(OUTPUT_NPZ_FILE, X=X_balanced, y=y_balanced)
print("✅ Dosya basariyla kaydedildi.")

print("\n🎉 ISLEM TAMAMLANDI!")
