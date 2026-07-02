"""All human-readable text (English variant, single source).

Mirrors strings_tr.py key-for-key with English values. The active language is
selected in config.yaml (language: "en" | "tr") and resolved by src/strings.py.
Code identifiers, file paths and original dataset feature column names are never
translated — only display labels are.
"""

# --- churn class labels (fixed across all figures) ---
CHURN_ETIKET = {0: "No churn (0)", 1: "Churn (1)"}

# --- figure file names (ASCII, fixed) ---
FIG_DOSYA = {
    "churn_balance": "churn_balance.png",
    "numeric_distributions": "numeric_distributions.png",
    "categorical_churn": "categorical_churn.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "missingness": "missingness.png",
}

# --- figure titles ({set} = dataset name) ---
FIG_BASLIK = {
    "churn_balance": "Churn distribution — {set}",
    "numeric_distributions": "Numeric feature distributions (by churn) — {set}",
    "categorical_churn": "Churn rate by categorical feature — {set}",
    "correlation_heatmap": "Numeric feature correlations — {set}",
    "missingness": "Missing data (pre-cleaning) — {set}",
}

# --- axis / shared labels ---
EKSEN = {
    "musteri_sayisi": "Number of customers",
    "churn_durumu": "Churn status",
    "deger": "Value",
    "churn_orani": "Churn rate",
    "eksik_sayisi": "Missing cell count",
    "kategori": "Category",
}

# --- table column headers (CSV + console) ---
KOLON = {
    "veri_seti": "Dataset",
    "sektor": "Sector",
    "satir": "Rows",
    "sutun": "Columns",
    "churn_yuzde": "Churn %",
    "eksik_once": "Missing (before)",
    "eksik_sonra": "Missing (after)",
    "not": "Note",
    "kolon": "Column",
    "islem": "Operation",
    "detay": "Detail",
    "ozellik": "Feature",
    "tekil_auc": "Univariate AUC",
    "bayrak": "Flag",
    "aksiyon": "Recommended action",
    "gerekce": "Rationale",
}

# --- flag / action values ---
BAYRAK_SUPHELI = "SUSPICIOUS"
BAYRAK_NORMAL = "-"
AKSIYON_INCELE = "review"
AKSIYON_TUT = "keep"

# --- leakage rationale texts ---
GEREKCE_SUPHELI = "Univariate AUC ≥ 0.90; separates the target almost perfectly — leakage suspected."
GEREKCE_NORMAL = "-"
GEREKCE_ALAN = {
    ("cell2cell", "RetentionCalls"): "Retention call may have occurred after the churn decision.",
    ("cell2cell", "RetentionOffersAccepted"): "Retention offer concurrent with the churn process; look-ahead risk.",
    ("cell2cell", "MadeCallToRetentionTeam"): "Contact with retention team may coincide with the churn signal.",
    ("bank", "IsActiveMember"): "Activity status may overlap definitionally with churn.",
    ("ecommerce", "DaySinceLastOrder"): "Days since last order may be close to the churn definition.",
    ("iranian", "Status"): "'Status' may be directly related to the churn state.",
}

# --- console / summary message templates ---
MSG = {
    "on_kontrol_ok": "Pre-check: {n} datasets + 1 unlabeled holdout found, consistent with reference profiles.",
    "on_kontrol_eksik": "MISSING FILE: {dosya} not found — stopped.",
    "profil_satiri": "{set:11s} {satir:>6d}x{sutun:<3d}  churn={churn:>5.1f}%  (ref {ref})  target='{hedef}'",
    "id_dusuruldu": "{set}: dropped identifier column(s) -> {kolonlar}",
    "temizlik_satiri": "{set:11s} {kolon:28s} {islem}",
    "leakage_supheli": "{set}: SUSPICIOUS -> {ozellik} (AUC={auc:.3f}) [{aksiyon}]",
    "kayit": "Saved: {yol}",
    "bolum": "===== {ad} =====",
    "bitti": "STEP 1 complete. Column-drop decision left to the user (leakage). No modeling performed.",
}

# ===================== STEP 2: modeling + calibration =====================
MODEL_AD = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
YONTEM_AD = {
    "ham": "Raw (uncalibrated)",
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
    "calibration": "Calibration curves ({model}) — {set}",
    "pr": "Precision-Recall curves — {set}",
    "roc": "ROC curves — {set}",
    "model_comparison": "Model comparison (PR-AUC / Recall / F1) — {set}",
}
EKSEN2 = {
    "recall": "Recall",
    "precision": "Precision",
    "fpr": "False positive rate",
    "tpr": "True positive rate (recall)",
    "tahmin_olasilik": "Predicted probability (bin mean)",
    "gercek_oran": "True positive fraction",
    "metrik_deger": "Value",
}
KOLON2 = {
    "veri_seti": "Dataset",
    "model": "Model",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "yontem": "Method",
    "brier": "Brier",
    "ece": "ECE",
    "kolon": "Column",
    "rol": "Role",
    "not": "Note",
}
ROL = {
    "sayisal": "numeric",
    "nominal": "nominal (one-hot)",
    "ozel_hp": "special: string -> numeric, missing/Unknown -> train median",
    "ozel_hp_bayrak": "special: HandsetPrice 'Unknown'/unparseable flag (0/1)",
    "ozel_sa": "special: top-15 categories + 'Other' -> one-hot",
}
MSG2 = {
    "hpo": "{set} / {model}: HPO done (best PR-AUC={skor:.4f})",
    "iranian_status": "iranian: removing 'Status' -> PR-AUC {oncesi} -> {sonrasi} (diff {fark:+.4f}) [{model}]",
    "bitti": "STEP 2 complete. Model family and calibration choice left to the user. No resampling (RQ1) performed.",
}

# ===================== STEP 3: RQ1 class balancing =====================
KOSUL_AD = {
    "baseline": "Baseline (natural)",
    "class_weight": "Class-weight",
    "smote": "SMOTENC/SMOTE",
    "adasyn": "ADASYN",
    "threshold": "Threshold shift (max-F1)",
}
FIG3_DOSYA = {
    "prauc": "rq1_prauc_by_method.png",
    "recall_precision": "rq1_recall_precision.png",
    "calibration": "rq1_calibration_by_method.png",
    "pr_operating": "rq1_pr_curve_operating.png",
}
FIG3_BASLIK = {
    "prauc": "RQ1 — PR-AUC by method — {set}",
    "recall_precision": "RQ1 — Recall/Precision trade-off (operating point) — {set}",
    "calibration": "RQ1 — Calibration by method (ECE) — {set}",
    "pr_operating": "RQ1 — PR curve + operating points — {set}",
}
KOLON3 = {
    "veri_seti": "Dataset",
    "yontem": "Method",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "ece": "ECE",
    "brier": "Brier",
    "esik": "Threshold",
}
MSG3 = {
    "set_basla": "{set}: RQ1 class-balancing comparison (LightGBM fixed, {n} conditions)",
    "kosul": "  {set} / {yontem}: PR-AUC={pr} recall={rec} precision={pre} ECE={ece} threshold={esik}",
    "adasyn_not": "NOTE: ADASYN operates on the encoded matrix; it produces fractional values for "
                  "one-hot categorical columns (synthetic samples fall outside 0/1) — to be noted in Methods/limitations.",
    "bitti": "STEP 3 (RQ1) complete. Method choice/interpretation left to the user. No SHAP/RQ2 performed.",
}

# ===================== STEP 4: SHAP / RQ2 =====================
FIG4_DOSYA = {
    "beeswarm": "shap_beeswarm.png",
    "importance": "shap_importance_bar.png",
    "dependence": "shap_dependence_topK.png",
    "waterfall_high": "shap_waterfall_high.png",
    "waterfall_low": "shap_waterfall_low.png",
    "consistency": "rq2_driver_consistency_heatmap.png",
}
FIG4_BASLIK = {
    "beeswarm": "SHAP summary (beeswarm, top 15) — {set}",
    "importance": "Global importance (mean |SHAP|, concept-aggregated) — {set}",
    "dependence": "SHAP dependence (strongest features) — {set}",
    "waterfall_high": "Local explanation — high-risk customer — {set}",
    "waterfall_low": "Local explanation — low-risk customer — {set}",
    "consistency": "RQ2 — Cross-dataset conceptual driver consistency",
}
EKSEN4 = {
    "shap_deger": "SHAP value (effect on churn probability)",
    "ortalama_etki": "Mean |SHAP| (contribution to churn)",
    "kavram": "Concept",
}
KOLON4 = {
    "feature": "Feature",
    "mean_abs_shap": "Mean |SHAP|",
    "sira": "Rank",
    "kavram": "Concept",
    "veri_seti": "Dataset",
    "durum": "Case",
    "olasilik": "Churn probability",
    "ham_deger": "Raw value",
    "shap_katki": "SHAP contribution",
    "top3_say": "Top-3 sector count",
}
DURUM = {"high": "high-risk", "low": "low-risk"}
MSG4 = {
    "set": "{set}: SHAP computed (sample n={n}); top drivers: {top}",
    "ornek_not": "{set}: stratified sample n={n} for SHAP explanation (full data {N}).",
    "eslenemeyen": "{set}: features absent from the concept map (->'other'): {liste}",
    "iranian_status": "iranian SHAP: 'Status' global importance rank={sira}/{toplam}, share={pay:.1f}% — {yorum}",
    "bitti": "STEP 4 (RQ2) complete. Interpretation left to the user. No profit/ROI (RQ3) performed.",
}

# ===================== STEP 5: profit / ROI / EMP (RQ3) =====================
STRATEJI_AD = {
    "A": "Accuracy threshold (t=0.5)",
    "B": "Profit-maximizing threshold (t*)",
    "hepsi": "Target everyone",
    "hic": "No intervention",
}
FIG5_DOSYA = {
    "profit_threshold": "rq3_profit_vs_threshold.png",
    "roi_sensitivity": "rq3_roi_sensitivity.png",
    "strategy": "rq3_strategy_comparison.png",
    "emp": "rq3_emp_by_dataset.png",
}
FIG5_BASLIK = {
    "profit_threshold": "RQ3 — Profit vs threshold (c={c}%, γ={g}) — {set}",
    "roi_sensitivity": "RQ3 — ROI sensitivity (cost × γ), at profit threshold — {set}",
    "strategy": "RQ3 — Strategy comparison (c={c}%, γ={g}) — {set}",
    "emp": "RQ3 — Cross-dataset EMP (expected maximum profit per customer)",
}
EKSEN5 = {
    "esik": "Decision threshold (t)",
    "kar": "Total profit (CLV units)",
    "kar_kisi": "Profit per customer (CLV units)",
    "maliyet_oran": "Intervention cost (fraction of mean CLV)",
    "gamma": "γ (retention success)",
    "roi": "ROI (profit / intervention cost)",
}
KOLON5 = {
    "veri_seti": "Dataset",
    "c": "Cost (c)",
    "c_oran": "Cost fraction",
    "gamma": "γ",
    "esik_a": "Threshold A (0.5)",
    "esik_b": "Threshold B (t*)",
    "kar_a": "Profit A",
    "kar_b": "Profit B",
    "roi_a": "ROI A",
    "roi_b": "ROI B",
    "roi_artis": "ROI gain (B−A)",
    "kar_artis_yuzde": "Profit gain %",
    "emp": "EMP (per customer)",
    "clv_temeli": "CLV basis",
    "ort_clv": "Mean CLV",
    "medyan_clv": "Median CLV",
    "sifir_clv": "Zero-CLV count",
}
MSG5 = {
    "clv": "{set}: CLV basis = {temel}; mean={ort:.1f} median={medyan:.1f} (zero CLV: {sifir})",
    "emp_varsayim": "EMP assumption: γ ~ Beta({a},{b}) (E[γ]={e:.2f}), reference cost = {c}% of mean CLV.",
    "set_ozet": "{set}: profit-threshold ROI-gain range (c,γ sweep): {dusuk} … {yuksek}; EMP={emp:.4f}",
    "emp_sira": "EMP ranking (high->low): {sira}",
    "bitti": "STEP 5 (RQ3) complete. Interpretation left to the user. No transfer (Step 6) performed.",
}

# ===================== STEP 6: conditional transfer probe =====================
TRANSFER_KARAR = {"dahil": "PASS", "kismi": "PARTIAL", "zayif": "WEAK"}
FIG6_DOSYA = {
    "prauc": "transfer_prauc_comparison.png",
    "retention": "transfer_retention_ratio.png",
}
FIG6_BASLIK = {
    "prauc": "Transfer probe — PR-AUC (transfer vs in-domain reference vs trivial)",
    "retention": "Transfer retention ratio (transfer / in-domain reference)",
}
EKSEN6 = {
    "prauc": "PR-AUC",
    "oran": "Retention ratio (transfer / reference)",
}
KOLON6 = {
    "senaryo": "Scenario",
    "yon": "Direction",
    "ortak_kavram": "Shared concept count",
    "transfer": "Transfer PR-AUC",
    "ref": "In-domain ref PR-AUC",
    "tam_ref": "Full-feature ref PR-AUC",
    "trivial": "Trivial PR-AUC",
    "oran": "Retention ratio",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "karar": "Decision",
    "kavram": "Concept",
    "kaynak_feature": "Source feature",
    "hedef_feature": "Target feature",
    "esleme_tipi": "Mapping type",
    "c2c_kolon": "Cell2Cell column",
    "iran_kolon": "Iranian column",
    "gerekce": "Selection rationale",
}
ESLEME_TIPI = {"onem": "importance-based", "semantik": "semantic"}
FIG6_DOSYA["semantic"] = "transfer_semantic_vs_importance.png"
FIG6_BASLIK["semantic"] = "Transfer — semantic vs importance-based mapping (retention ratio)"
MSG6 = {
    "senaryo": "{ad} ({yon}): shared concepts={k} | transfer PR-AUC={tr:.3f} | ref={ref:.3f} "
               "| trivial={tv:.3f} | ratio={oran:.2f} -> {karar}",
    "semantik": "{ad} [semantic]: concepts={k} | transfer PR-AUC={tr:.3f} | ref={ref:.3f} "
                "| trivial={tv:.3f} | ratio={oran:.2f} -> {karar}",
    "atlanan": "  concepts skipped in semantic mapping: {liste}",
    "bitti": "STEP 6 (transfer) complete. Interpretation left to the user. No robustness/write-up (Step 7) performed.",
}

# ===================== STEP 7: robustness (ablation + CI + significance) =====================
KOSUL_ABLATION = {
    "K0": "K0 Full system (raw + profit threshold)",
    "K1": "K1 −profit threshold (accuracy 0.5)",
    "K2": "K2 −calibration (class-weight)",
    "K3": "K3 −strong model (LogReg)",
    "K4": "K4 +imbalance (SMOTE)",
}
FIG7_DOSYA = {
    "ablation": "ablation_profit_waterfall.png",
    "ci": "robustness_ci_forest.png",
    "significance": "significance_heatmap.png",
}
FIG7_BASLIK = {
    "ablation": "Profit-chain ablation — profit lost when a component is removed (5 seeds mean ± 95% CI)",
    "ci": "Repeated runs — PR-AUC point + 95% CI (5 seeds)",
    "significance": "Significance — model-pair Wilcoxon p-value",
}
EKSEN7 = {
    "kar": "Total profit (CLV units)",
    "prauc": "PR-AUC",
}
KOLON7 = {
    "veri_seti": "Dataset",
    "kosul": "Condition",
    "kar_ort": "Profit (mean)",
    "kar_ci_low": "Profit CI low",
    "kar_ci_high": "Profit CI high",
    "roi": "ROI",
    "ece": "ECE",
    "esik": "Threshold t*",
    "dkar": "Δprofit (vs K0)",
    "dkar_yuzde": "Δprofit %",
    "metrik": "Metric",
    "ort": "Mean",
    "std": "Std",
    "ci_low": "CI low (95%)",
    "ci_high": "CI high (95%)",
    "kiyas": "Comparison",
    "test": "Test",
    "istatistik": "Statistic",
    "p": "p-value",
    "sonuc": "Result",
}
ANLAMLI = {True: "significant (p<0.05)", False: "noise (p≥0.05)"}
MSG7 = {
    "ablation": "{set} / {kosul}: profit={kar:.0f} ROI={roi:.2f} ECE={ece:.3f} t*={esik:.2f} | Δvs K0={dkar:+.0f} ({yuzde:+.1f}%)",
    "ci": "{set} {metrik}: {ort:.4f} ± {std:.4f}  [95% CI {lo:.4f}, {hi:.4f}]",
    "yontem": "Significance method: Wilcoxon signed-rank over paired scores (dataset×seed OOF PR-AUC, n={n}).",
    "bitti": "STEP 7 (robustness) complete. Experimental phase done; interpretation left to the user, write-up next.",
}

# ===================== STEP 8: profit-driven baseline (ProfLogit) =====================
YAKLASIM_AD = {"ours": "Ours (raw LightGBM + profit threshold)", "proflogit": "ProfLogit (EMPC-maximizing)"}
FIG8_DOSYA = {"empc": "proflogit_vs_ours_empc.png", "profit": "proflogit_vs_ours_profit.png"}
FIG8_BASLIK = {
    "empc": "ProfLogit vs ours — EMPC (5 datasets, fold mean ± std)",
    "profit": "ProfLogit vs ours — profit (5 datasets, fold mean ± std)",
}
KOLON8 = {
    "veri_seti": "Dataset", "yaklasim": "Approach", "empc": "EMPC", "kar": "Profit",
    "roi": "ROI", "pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision",
    "f1": "F1", "p_empc": "p (EMPC, Wilcoxon)", "p_kar": "p (profit, Wilcoxon)",
}
MSG8 = {
    "set": "{set} / {yaklasim}: EMPC={empc:.4f} profit={kar:.0f} ROI={roi:.2f} PR-AUC={pr:.3f} ({sure:.0f}s)",
    "ornek": "{set}: stratified sample n={n} for ProfLogit (full data {N}), seed={seed}.",
    "atif": "ProfLogit method: Stripling, vanden Broucke, Antonio, Baesens, Snoeck (2018), "
            "'Profit maximizing logistic model for customer churn prediction using genetic algorithms' "
            "(EMPC objective + genetic coefficient search).",
    "wilcoxon": "Paired Wilcoxon (dataset×fold, n={n}): EMPC p={pe:.4f}, profit p={pk:.4f}.",
    "bitti": "STEP 8 complete. Profit-driven baseline compared; interpretation left to the user.",
}

# ===================== STEP 9: signature finding (predictability ≠ profitability) =====================
FIG9_DOSYA = {"scatter": "prauc_vs_emp_scatter.png", "dagilim": "churner_value_distributions.png"}
FIG9_BASLIK = {
    "scatter": "Signature finding — predictability (PR-AUC) vs profitability (EMP)",
    "dagilim": "CLV distribution of churners only (by dataset)",
}
EKSEN9 = {"prauc": "PR-AUC (predictability)", "emp": "EMP (profitability, per customer)",
          "clv": "Churner CLV", "sayi": "Number of customers"}
KOLON9 = {
    "veri_seti": "Dataset", "pr_auc": "PR-AUC", "emp": "EMP",
    "churner_medyan": "Churner CLV median", "churner_gini": "Churner CLV Gini",
    "churner_cv": "Churner CLV CV", "churn_orani": "Churn rate",
}
MSG9 = {
    "korelasyon": "PR-AUC↔EMP: Spearman ρ={rho:.3f} (p={pr:.3f}), Pearson r={r:.3f} (p={pp:.3f}). "
                  "n=5 small → trend/illustration, no strong claim.",
    "sira": "PR-AUC ranking: {prs}  |  EMP ranking: {emps}  -> {yorum}",
    "bitti": "STEP 9 complete. Signature finding (predictability ≠ profitability) documented.",
}
