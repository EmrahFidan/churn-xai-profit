"""İnsan tarafından okunan TÜM metinler burada toplanır (tek kaynak).

İleride İngilizceye çevirmek için yalnızca bu dosyayı çevirip adımı yeniden
çalıştırmak yeterlidir; tüm figür başlıkları, eksen/legend etiketleri, tablo
kolon başlıkları ve özet mesajları İngilizce çıkar. Metinleri kodun içine
dağıtmayın — buraya ekleyin.

Kod kimlikleri ve dosya yolları ASCII/İngilizce kalır; veri setlerinin orijinal
feature kolon adlarına dokunulmaz (yalnızca gösterimde kullanılan etiketler
Türkçedir).
"""

# --- churn sınıfı etiketleri (tüm figürlerde sabit) ---
CHURN_ETIKET = {0: "Churn yok (0)", 1: "Churn var (1)"}

# --- figür dosya adları (ASCII, sabit) ---
FIG_DOSYA = {
    "churn_balance": "churn_balance.png",
    "numeric_distributions": "numeric_distributions.png",
    "categorical_churn": "categorical_churn.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "missingness": "missingness.png",
}

# --- figür başlıkları ({set} = veri seti adı) ---
FIG_BASLIK = {
    "churn_balance": "Churn dağılımı — {set}",
    "numeric_distributions": "Sayısal değişken dağılımları (churn'e göre) — {set}",
    "categorical_churn": "Kategorik değişkenlere göre churn oranı — {set}",
    "correlation_heatmap": "Sayısal değişken korelasyonları — {set}",
    "missingness": "Eksik veri (temizlik öncesi) — {set}",
}

# --- eksen / ortak etiketler ---
EKSEN = {
    "musteri_sayisi": "Müşteri sayısı",
    "churn_durumu": "Churn durumu",
    "deger": "Değer",
    "yogunluk": "Yoğunluk",
    "churn_orani": "Churn oranı",
    "eksik_sayisi": "Eksik hücre sayısı",
    "kategori": "Kategori",
}

# --- tablo kolon başlıkları (CSV + konsol) ---
KOLON = {
    # genel bakış tablosu
    "veri_seti": "Veri Seti",
    "sektor": "Sektör",
    "satir": "Satır",
    "sutun": "Sütun",
    "churn_yuzde": "Churn %",
    "eksik_once": "Eksik (öncesi)",
    "eksik_sonra": "Eksik (sonrası)",
    "not": "Not",
    # temizlik log
    "kolon": "Kolon",
    "islem": "İşlem",
    "detay": "Detay",
    # leakage
    "ozellik": "Özellik",
    "tekil_auc": "Tekil AUC",
    "bayrak": "Bayrak",
    "aksiyon": "Önerilen Aksiyon",
    "gerekce": "Gerekçe",
}

# --- bayrak / aksiyon değerleri ---
BAYRAK_SUPHELI = "ŞÜPHELİ"
BAYRAK_NORMAL = "-"
AKSIYON_INCELE = "incele"
AKSIYON_TUT = "tut"

# --- leakage gerekçe metinleri ---
GEREKCE_SUPHELI = "Tek başına AUC ≥ 0.90; hedefi neredeyse birebir ayırıyor — sızıntı şüphesi."
GEREKCE_NORMAL = "-"
# alan bilgisi notları (set, feature) -> ek açıklama
GEREKCE_ALAN = {
    ("cell2cell", "RetentionCalls"): "Elde tutma araması churn kararından sonra yapılmış olabilir.",
    ("cell2cell", "RetentionOffersAccepted"): "Elde tutma teklifi churn süreciyle eşzamanlı; ileriye dönük bilgi riski.",
    ("cell2cell", "MadeCallToRetentionTeam"): "Elde tutma ekibiyle temas churn sinyaliyle çakışabilir.",
    ("bank", "IsActiveMember"): "Aktiflik durumu churn ile tanımsal olarak örtüşebilir.",
    ("ecommerce", "DaySinceLastOrder"): "Son siparişten geçen gün, churn tanımına yakın olabilir.",
    ("iranian", "Status"): "'Status' churn durumuyla doğrudan ilişkili olabilir.",
}

# --- konsol / özet mesaj şablonları ---
MSG = {
    "on_kontrol_ok": "Ön kontrol: {n} veri seti + 1 etiketsiz holdout bulundu, referans profillerle uyumlu.",
    "on_kontrol_eksik": "EKSİK DOSYA: {dosya} bulunamadı — durduruldu.",
    "profil_satiri": "{set:11s} {satir:>6d}x{sutun:<3d}  churn={churn:>5.1f}%  (ref {ref})  hedef='{hedef}'",
    "id_dusuruldu": "{set}: kimlik kolonu düşürüldü -> {kolonlar}",
    "temizlik_satiri": "{set:11s} {kolon:28s} {islem}",
    "leakage_supheli": "{set}: ŞÜPHELİ -> {ozellik} (AUC={auc:.3f}) [{aksiyon}]",
    "kayit": "Kaydedildi: {yol}",
    "bolum": "===== {ad} =====",
    "bitti": "ADIM 1 tamamlandı. Düşürme kararı kullanıcıya bırakıldı (leakage). Modellemeye geçilmedi.",
}
