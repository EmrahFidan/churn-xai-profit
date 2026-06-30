"""Setler-arası sürücü tutarlılığı için KAVRAM HARİTASI (yorum katmanı).

Her ham feature'ı sektör-bağımsız bir kavrama bağlar. Bu YALNIZCA SHAP sürücülerini
kavram düzeyinde karşılaştırmak içindir; veriyi BİRLEŞTİRMEZ. Eşlenemeyen feature
'diger' olur ve loglanır. Kavram adları (insan-okur) strings_tr'de değil burada
kanonik anahtar; gösterim adı KAVRAM_AD'da.
"""

# kanonik kavram -> Türkçe gösterim
KAVRAM_AD = {
    "tenure": "İlişki süresi (tenure)",
    "sozlesme": "Sözleşme / taahhüt",
    "kullanim": "Kullanım hacmi",
    "dusus": "Kullanım düşüşü",
    "destek": "Şikâyet / destek / hizmet kalitesi",
    "etkilesim": "Etkileşim / recency",
    "parasal": "Parasal değer / harcama",
    "kredi": "Kredi / risk",
    "odeme": "Ödeme / fatura tipi",
    "hizmet": "Hizmet / abonelik kapsamı",
    "cihaz": "Cihaz / ekipman",
    "demografi": "Demografi",
    "diger": "Diğer",
}

# ham feature adı -> kanonik kavram (5 set; adlar setler arası genelde benzersiz)
HARITA = {
    # --- telco ---
    "tenure": "tenure", "Contract": "sozlesme",
    "MonthlyCharges": "parasal", "TotalCharges": "parasal",
    "PaymentMethod": "odeme", "PaperlessBilling": "odeme",
    "gender": "demografi", "SeniorCitizen": "demografi", "Partner": "demografi", "Dependents": "demografi",
    "PhoneService": "hizmet", "MultipleLines": "hizmet", "InternetService": "hizmet",
    "OnlineSecurity": "hizmet", "OnlineBackup": "hizmet", "DeviceProtection": "hizmet",
    "StreamingTV": "hizmet", "StreamingMovies": "hizmet", "TechSupport": "destek",
    # --- bank ---
    "Tenure": "tenure", "Balance": "parasal", "EstimatedSalary": "parasal",
    "NumOfProducts": "hizmet", "HasCrCard": "odeme", "IsActiveMember": "kullanim",
    "Geography": "demografi", "Gender": "demografi", "Age": "demografi", "CreditScore": "kredi",
    # --- ecommerce ---
    "NumberOfDeviceRegistered": "kullanim", "DaySinceLastOrder": "etkilesim",
    "SatisfactionScore": "destek", "Complain": "destek",
    "CashbackAmount": "parasal", "MaritalStatus": "demografi",
    "PreferedOrderCat": "diger", "WarehouseToHome": "etkilesim", "NumberOfAddress": "etkilesim",
    # --- iranian ---
    "Subscription  Length": "tenure", "Charge  Amount": "parasal", "Customer Value": "parasal",
    "Seconds of Use": "kullanim", "Frequency of use": "kullanim", "Frequency of SMS": "kullanim",
    "Distinct Called Numbers": "kullanim", "Status": "kullanim",
    "Call  Failure": "destek", "Complains": "destek",
    "Tariff Plan": "odeme", "Age Group": "demografi",
    # --- cell2cell ---
    "MonthsInService": "tenure", "NewCellphoneUser": "tenure", "NotNewCellphoneUser": "tenure",
    "MonthlyRevenue": "parasal", "TotalRecurringCharge": "parasal",
    "MonthlyMinutes": "kullanim", "ReceivedCalls": "kullanim", "OutboundCalls": "kullanim",
    "InboundCalls": "kullanim", "PeakCallsInOut": "kullanim", "OffPeakCallsInOut": "kullanim",
    "OverageMinutes": "kullanim", "RoamingCalls": "kullanim", "CallForwardingCalls": "kullanim",
    "CallWaitingCalls": "kullanim", "ThreewayCalls": "kullanim", "DirectorAssistedCalls": "kullanim",
    "ReferralsMadeBySubscriber": "kullanim",
    "PercChangeMinutes": "dusus", "PercChangeRevenues": "dusus",
    "CustomerCareCalls": "destek", "RetentionCalls": "destek", "RetentionOffersAccepted": "destek",
    "MadeCallToRetentionTeam": "destek", "DroppedCalls": "destek", "BlockedCalls": "destek",
    "DroppedBlockedCalls": "destek", "UnansweredCalls": "destek",
    "UniqueSubs": "hizmet", "ActiveSubs": "hizmet",
    "HasCreditCard": "odeme",
    "IncomeGroup": "demografi", "AgeHH1": "demografi", "AgeHH2": "demografi",
    "ChildrenInHH": "demografi", "Homeownership": "demografi", "Occupation": "demografi",
    "PrizmCode": "demografi", "ServiceArea": "demografi", "MaritalStatus_c2c": "demografi",
    "TruckOwner": "demografi", "RVOwner": "demografi", "OwnsComputer": "demografi",
    "OwnsMotorcycle": "demografi",
    "HandsetPrice": "cihaz", "handsetprice_unknown": "cihaz", "CurrentEquipmentDays": "cihaz",
    "Handsets": "cihaz", "HandsetModels": "cihaz", "HandsetRefurbished": "cihaz",
    "HandsetWebCapable": "cihaz", "CreditRating": "kredi", "AdjustmentsToCreditRating": "kredi",
    "BuysViaMailOrder": "diger", "RespondsToMailOffers": "diger", "OptOutMailings": "diger",
    "NonUSTravel": "diger",
}
# not: cell2cell ve ecommerce'de 'MaritalStatus' çakışır; ikisi de demografi -> sorun yok.
HARITA.setdefault("MaritalStatus", "demografi")

KAVRAM_SIRA = ["tenure", "sozlesme", "kullanim", "dusus", "etkilesim", "destek",
               "parasal", "kredi", "odeme", "hizmet", "cihaz", "demografi", "diger"]


def kavram(feature: str) -> str:
    """Ham feature adını kanonik kavrama eşler (eşlenemeyen -> 'diger')."""
    return HARITA.get(feature, "diger")
