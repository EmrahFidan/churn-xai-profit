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

# ===================== ADIM 2: modelleme + kalibrasyon =====================
MODEL_AD = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
YONTEM_AD = {
    "ham": "Ham (kalibre değil)",
    "Platt": "Platt (sigmoid)",
    "Isotonic": "Isotonic",
}

FIG2_DOSYA = {
    "calibration": "calibration_curves.png",
    "pr": "pr_curve.png",
    "roc": "roc_curve.png",
    "model_comparison": "model_comparison.png",
}
FIG2_BASLIK = {
    "calibration": "Kalibrasyon eğrileri ({model}) — {set}",
    "pr": "Precision-Recall eğrileri — {set}",
    "roc": "ROC eğrileri — {set}",
    "model_comparison": "Model karşılaştırması (PR-AUC / Duyarlılık / F1) — {set}",
}
EKSEN2 = {
    "recall": "Duyarlılık (recall)",
    "precision": "Kesinlik (precision)",
    "fpr": "Yanlış pozitif oranı",
    "tpr": "Doğru pozitif oranı (recall)",
    "tahmin_olasilik": "Tahmin edilen olasılık (kova ortalaması)",
    "gercek_oran": "Gerçek pozitif oranı",
    "metrik_deger": "Değer",
}
# tablo kolon başlıkları (Adım 2)
KOLON2 = {
    "veri_seti": "Veri Seti",
    "model": "Model",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Duyarlılık",
    "precision": "Kesinlik",
    "f1": "F1",
    "yontem": "Yöntem",
    "brier": "Brier",
    "ece": "ECE",
    "kolon": "Kolon",
    "rol": "Rol",
    "not": "Not",
}
ROL = {
    "sayisal": "sayısal",
    "nominal": "nominal (one-hot)",
    "ozel_hp": "özel: string -> sayısal, eksik/Unknown -> train medyanı",
    "ozel_hp_bayrak": "özel: HandsetPrice 'Unknown'/parse edilemez bayrağı (0/1)",
    "ozel_sa": "özel: en sık 15 kategori + 'Other' -> one-hot",
}
MSG2 = {
    "hpo": "{set} / {model}: HPO bitti (en iyi PR-AUC={skor:.4f})",
    "deg": "{set} / {model}: 5-kat değerlendirme + kalibrasyon bitti",
    "kazanan": "{set}: en iyi model {model} (PR-AUC={skor})",
    "kalib_kazanan": "{set} / {model}: kalibrasyon kazananı {yontem} (ECE={ece})",
    "iranian_status": "iranian: 'Status' çıkarılınca PR-AUC {oncesi} -> {sonrasi} (fark {fark:+.4f}) [{model}]",
    "bitti": "ADIM 2 tamamlandı. Model ailesi ve kalibrasyon seçimi kullanıcıya bırakıldı. Dengeleme (RQ1) yapılmadı.",
}

# ===================== ADIM 3: RQ1 dengeleme =====================
KOSUL_AD = {
    "baseline": "Baseline (doğal)",
    "class_weight": "Class-weight",
    "smote": "SMOTENC/SMOTE",
    "adasyn": "ADASYN",
    "threshold": "Eşik kaydırma (max-F1)",
}
FIG3_DOSYA = {
    "prauc": "rq1_prauc_by_method.png",
    "recall_precision": "rq1_recall_precision.png",
    "calibration": "rq1_calibration_by_method.png",
    "pr_operating": "rq1_pr_curve_operating.png",
}
FIG3_BASLIK = {
    "prauc": "RQ1 — Yönteme göre PR-AUC — {set}",
    "recall_precision": "RQ1 — Duyarlılık/Kesinlik dengesi (işletim noktası) — {set}",
    "calibration": "RQ1 — Yönteme göre kalibrasyon (ECE) — {set}",
    "pr_operating": "RQ1 — PR eğrisi + işletim noktaları — {set}",
}
KOLON3 = {
    "veri_seti": "Veri Seti",
    "yontem": "Yöntem",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Duyarlılık",
    "precision": "Kesinlik",
    "f1": "F1",
    "ece": "ECE",
    "brier": "Brier",
    "esik": "Eşik",
}
MSG3 = {
    "set_basla": "{set}: RQ1 dengeleme karşılaştırması (LightGBM sabit, {n} koşul)",
    "kosul": "  {set} / {yontem}: PR-AUC={pr} recall={rec} precision={pre} ECE={ece} eşik={esik}",
    "adasyn_not": "NOT: ADASYN encode edilmiş matriste çalışır; one-hot kategorik kolonlar için kesirli "
                  "değerler üretir (sentetik örnekler 0/1 dışına çıkar) — Methods/limitations'a yazılacak.",
    "bitti": "ADIM 3 (RQ1) tamamlandı. Yöntem seçimi/yorum kullanıcıya bırakıldı. SHAP/RQ2 yapılmadı.",
}

# ===================== ADIM 4: SHAP / RQ2 =====================
FIG4_DOSYA = {
    "beeswarm": "shap_beeswarm.png",
    "importance": "shap_importance_bar.png",
    "dependence": "shap_dependence_topK.png",
    "waterfall_high": "shap_waterfall_high.png",
    "waterfall_low": "shap_waterfall_low.png",
    "consistency": "rq2_driver_consistency_heatmap.png",
}
FIG4_BASLIK = {
    "beeswarm": "SHAP özet (beeswarm, ilk 15) — {set}",
    "importance": "Global önem (ortalama |SHAP|, kavram-toplamı) — {set}",
    "dependence": "SHAP bağımlılık (en güçlü değişkenler) — {set}",
    "waterfall_high": "Tekil açıklama — yüksek riskli müşteri — {set}",
    "waterfall_low": "Tekil açıklama — düşük riskli müşteri — {set}",
    "consistency": "RQ2 — Setler-arası kavramsal sürücü tutarlılığı",
}
EKSEN4 = {
    "shap_deger": "SHAP değeri (churn olasılığına etki)",
    "ortalama_etki": "Ortalama |SHAP| (churn'e katkı)",
    "kavram": "Kavram",
}
KOLON4 = {
    "feature": "Özellik",
    "mean_abs_shap": "Ortalama |SHAP|",
    "sira": "Sıra",
    "kavram": "Kavram",
    "veri_seti": "Veri Seti",
    "durum": "Durum",
    "olasilik": "Churn olasılığı",
    "ham_deger": "Ham değer",
    "shap_katki": "SHAP katkısı",
    "top3_say": "Top-3 sektör sayısı",
}
DURUM = {"high": "yüksek riskli", "low": "düşük riskli"}
MSG4 = {
    "set": "{set}: SHAP hesaplandı (örneklem n={n}); top sürücü: {top}",
    "ornek_not": "{set}: SHAP açıklaması için stratified örneklem n={n} (tam veri {N}).",
    "eslenemeyen": "{set}: kavram haritasında olmayan (->'diger') feature'lar: {liste}",
    "iranian_status": "iranian SHAP: 'Status' global önem sırası={sira}/{toplam}, pay=%{pay:.1f} — {yorum}",
    "bitti": "ADIM 4 (RQ2) tamamlandı. Yorum/karar kullanıcıya bırakıldı. Kâr/ROI (RQ3) yapılmadı.",
}
