# YAZIM ANAYASASI - 2. Makale (Churn/XAI/Profit)
> Her bolum yazilirken bu kurallara uyulur. Degisiklik = ikimizin onayi.

## A. SUREC
1. Her bolum ONCE Turkce -> Emrah onaylar -> SONRA Ingilizce ceviri.
2. Sira: Methods -> Results -> Discussion -> Intro -> Conclusion+Abstract.
3. Related Works + kaynakca + resmi MDPI formati = hocalar (biz dokunmayiz).
4. Her bolum ayri teslim edilir; Emrah Word'e alir.

## B. ALTIN KURAL - AI KOKUSU YOK (sifir tolerans)
- Moreover/Furthermore/Additionally spam'i yok; gecisler az ve dogal.
- Dolgu yasak: It is important to note / It is worth mentioning /
  It should be noted / In this context / As mentioned above.
- Klise yasak: delve, leverage, seamless, comprehensive, pivotal, realm,
  landscape, paradigm, harness, unlock.
- Degisken cumle ritmi; asiri simetrik liste yok, paragraf agirlikli.
- Sayi konusur, sifat degil. Abartili ozguven yok.

## C. PARAGRAF/CUMLE KURALI (net standart)
- Paragraf = 3-6 cumle. Tek cumlelik paragraf YOK, 8+ cumle blok YOK.
- Her paragraf TEK fikir; fikir bitti -> paragraf biter.
- Cumle ort. 15-25 kelime; art arda iki 30+ kelimelik cumle yok.

## D. HAKEM-PROOF
- 'robustness' DEME -> 'stability across seeds' (test etmedik).
- 'en iyi / superior yontem' iddiasi YOK (ablation kanitlamadikca).
- 0.664 vs 0.665 = ustun DEGIL (gurultu); yalniz anlamli farklar vurgulanir.
- Onceki-is kiyasi: ayni veri setinde literatur skoru varsa neden farkli
  oldugumuz (PR-AUC, leakage-kontrollu split) aciklanir.
- Kisaltma ilk gectiginde acilir: SHAP, EMP, EMPC, CLV, PR-AUC, ECE, ROI, SMOTENC.
- Metrik ilk gectiginde matematiksel tanim/formul verilir.

## E. DURUSTLUK
- Uydurma sayi/atif YOK. Sayilar outputs/tables CSV'lerinden cekilir.
- Sayi elimde degilse metinde [TABLO_X'ten sayi] placeholder birakilir.
- Negatif bulgu (transfer coktu) durustce yazilir.
- Emin olunmayan yerde Emrah'a sorulur, tahmin edilmez.

## F. KILITLI KARARLAR
- Gosterim: 0-1 olcek (0.664), yuzde DEGIL. Tum makale/tablo/figur ayni.
- Ingilizce: US spelling (MDPI standardi).
- Uzunluk: tam/detayli yazilir, Emrah kisaltir.
- Ozne: 'we' kullanilir (we propose / we find), asiri pasif yok.
- Tablo/figur referansi placeholder: (Table X), (Figure Y) - numara sonra.
- RQ'lar Intro'da ayri alt-baslik.
