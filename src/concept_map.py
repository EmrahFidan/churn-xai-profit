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
    "parasal": "Parasal değer / harcama",
    "odeme": "Ödeme / fatura tipi",
    "hizmet": "Hizmet / abonelik kapsamı",
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
    "Geography": "demografi", "Gender": "demografi", "Age": "demografi", "CreditScore": "diger",
    # --- ecommerce ---
    "NumberOfDeviceRegistered": "kullanim", "DaySinceLastOrder": "dusus",
    "SatisfactionScore": "destek", "Complain": "destek",
    "CashbackAmount": "parasal", "MaritalStatus": "demografi",
    "PreferedOrderCat": "diger", "WarehouseToHome": "diger", "NumberOfAddress": "diger",
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
    "HandsetPrice": "diger", "handsetprice_unknown": "diger", "CurrentEquipmentDays": "diger",
    "Handsets": "diger", "HandsetModels": "diger", "HandsetRefurbished": "diger",
    "HandsetWebCapable": "diger", "CreditRating": "diger", "AdjustmentsToCreditRating": "diger",
    "BuysViaMailOrder": "diger", "RespondsToMailOffers": "diger", "OptOutMailings": "diger",
    "NonUSTravel": "diger",
}
# not: cell2cell ve ecommerce'de 'MaritalStatus' çakışır; ikisi de demografi -> sorun yok.
HARITA.setdefault("MaritalStatus", "demografi")

KAVRAM_SIRA = ["tenure", "sozlesme", "kullanim", "dusus", "destek", "parasal", "odeme", "hizmet", "demografi", "diger"]


def kavram(feature: str) -> str:
    """Ham feature adını kanonik kavrama eşler (eşlenemeyen -> 'diger')."""
    return HARITA.get(feature, "diger")
