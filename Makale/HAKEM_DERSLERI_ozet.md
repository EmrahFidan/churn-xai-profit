# Hakem Dersleri — 1. Makale (AgriEngineering) Review Analizi

> Kaynak: 4 hakem x 3 tur = 8 review dosyasi (v1/, v2/). Makale KABUL edildi.
> Amac: 2. makaleyi (Churn) yazarken bu kaliplari onceden karsilamak = hakem-proof.

## EN KRITIK 3 KALIP (her turda tekrar cikti)

### 1. SAYISAL TUTARLILIK (en cok tekrarlanan sikayet)
Hakemler Abstract/Intro/Tablo/Figur arasindaki HER sayiyi karsilastiriyor.
- R2: Abstract 10028 ama Figure 2 10618 // Abstract 25 epoch ama Table 3 100 epoch
- R4: EfficientNet 98.66 vs YOLO 98.56 -> YOLO ustun DEME
DERS: Submit oncesi tek sayi denetimi. Abstract PR-AUC/EMP/p, tablolarla ondalik ondalik ayni.
0.664 vs 0.665 ustun DEGIL (gurultu).

### 2. LEAKAGE / SPLIT PROTOKOLU (R2 takintisi - bizim guclu yanimiz)
- R2: Split ham veride yapildi diye ACIKCA yaz; dengeleme/augment SADECE train, test ASLA.
- Response letterda degil MANUSCRIPTte yaz.
DERS: Methodsta ac: 5-fold split ham veride; SMOTENC/esik/kalibrasyon yalniz fold-ici train;
test folda hicbir islem yok. Bizde ZATEN dogru - vurgula. Leakage taramasi guclu koz.

### 3. TEKRARLANABILIRLIK + KOD/VERI LINKI (R2, R3 israrla)
- R2: GitHub repo linki + calistirma talimati + ortam/bagimlilik.
- R3: Veri linki + PROPER METADATA (kaynak, kac ornek, nasil toplanmis).
DERS: GitHub linki (raw veri yuklendi) + requirements + README. Her 5 set icin metadata
(Kaggle/UCI link + lisans + churn tanimi). 5-seed + %95 CI bizde var - vurgula.

## DIGER KALIPLAR
- Metodoloji netligi: model konfigu TAM, metriklerin matematiksel tanimi (PR-AUC/EMP/ECE formul),
  neden bu model gerekcesi (LightGBM vs XGB p=0.23).
- RQ: Introductionda ayri alt-baslik (R3).
- Ablation ZORUNLU (R4): bizde var (kar-eshigi -298/-789 pct) - one cikar.
- Gorsel kanit (R1): SHAP + saçilim + kalibrasyon egrisi hazir.
- Yuksek sonuca aciklama (R4): iranian 0.96 neden kolay, cell2cell 0.47 gercek sinir - acikla.
- Discussion KRITIK olsun (R4): neden dengeleme basarisiz, neden transfer coktu, neden
  predictability != profitability - betimleme degil NEDEN.
- Kisaltmalar ilk gectikte acik (R1): SHAP/EMP/EMPC/CLV/PR-AUC/ECE/ROI/SMOTENC.
- Referans (R3/R4): DOI tam, duplike yok, guncel 2023-2026. Uydurma atif YOK.
- Hardware tutarli (R3): GPU yok, standart donanim - erisilebilirlik artisi.

## SUBMIT-ONCESI CHECKLIST
1. Tek sayi denetimi (Abstract <-> tablo ondalik ayni)
2. Iddia = sayi destekli
3. Methodsta leakage/split ACIK
4. GitHub linki + requirements + README
5. 5 veri seti metadatasi (link + lisans)
6. RQ Introductionda ayri alt-baslik
7. Metrik matematiksel tanim (formul)
8. Ablation one cikmis
9. Discussion KRITIK (nedenler)
10. Kisaltmalar ilk gectikte acik
11. Referans DOI tam, duplike yok
12. Gorsel kanit (SHAP + saçilim + kalibrasyon)

---

## CHURN'E OZEL EK DERSLER (ham v1/v2 review dosyalari okundu)

### EN KRITIK: ASIRI IDDIA = 3 TUR AZAP (R4 dersi)
Eski makalede V1 (hybrid balancing) 'katkimiz/en iyi' diye sunuldu; ablation
V1'in baseline'i (V2) GECMEDIGINI, V3'un kazandigini gosterdi. R4 celiskiyi
yakaladi -> makale 'yontem ustun'den 'stratejilerin ampirik kiyasi'na cevrildi.
CHURN DERSI: RQ1'de ayni risk var (dengeleme PR-AUC'yi iyilestirmiyor).
ASLA 'en iyi dengeleme yontemi' cumlesi kurma. Trade-off cercevesini bozma.
Kural: ablation kanitlamadan hicbir bilesene 'kazandi' deme.

### YENI SERT DERSLER (ust ozette yoktu)
1. 'ROBUSTNESS' test etmeden KULLANMA. R4 tum robustness iddialarini sildirdi
   (3. tur). Bizde 5-seed + CI = 'stability across seeds' de; genis iddia yok.
2. AYNI VERI SETINDE ONCEKI ISLERLE KIYAS (R4: '[12] %99.11 almis, sen?').
   Telco/Bank cok calisildi -> Related Work'te onceden savun:
   PR-AUC (accuracy degil) + leakage-kontrollu split + farkli protokol.
3. SAYISAL TUTARSIZLIK TUR TUR GERI GELIYOR (R2 ayni hatayi 2. turda buldu).
   Submit oncesi tek-sayi denetimi SART.
4. GOSTERIM TUTARLILIGI (R2): ya hep yuzde ya hep 0-1. KARAR: 0-1 olcek.
5. METADATA 'link' DEGIL proper metadata (R3 israr): 5 set icin kaynak linki
   + lisans + ornek sayisi + churn tanimi tablosu.
6. DOI eksikleri yakalandi (R3): her ref DOI'li, duplike yok, 2023-2026.
7. TUM modeller icin kiyas egrisi (R2: tekini gosterip digerini atlama).

### CHURN ZATEN GUCLU (bastan kapattigimiz tuzaklar)
Ablation var (K0-K4) / dengeleme-kazaniyor iddiasi yok / leakage kontrolu
guclu / RQ net / 5-seed CI / kod+ham veri GitHub'da / Discussion neden-odakli.

### SUBMIT ONCESI EK TIK
[ ] robustness -> stability dili
[ ] onceki-is kiyas paragrafi (Related Work)
[ ] 5 set metadata tablosu
[ ] DOI + duplike + guncellik
[ ] tek-sayi denetimi
[ ] 0-1 gosterim her yerde
[ ] hicbir yerde 'en iyi yontem' iddiasi yok
