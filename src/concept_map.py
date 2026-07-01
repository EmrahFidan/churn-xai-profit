"""CONCEPT MAP for cross-dataset driver consistency (annotation layer).

Maps each raw feature to a sector-independent concept. This is ONLY for comparing SHAP
drivers at the concept level; it does NOT MERGE the data. Unmappable features become
'diger' and are logged. Concept names (human-readable) are not in strings_tr but here;
the canonical key is here, the display name is in KAVRAM_AD.
"""

# canonical concept -> English display
KAVRAM_AD = {
    "tenure": "Relationship duration (tenure)",
    "sozlesme": "Contract / commitment",
    "kullanim": "Usage volume",
    "dusus": "Usage decline",
    "destek": "Complaint / support / service quality",
    "etkilesim": "Engagement / recency",
    "parasal": "Monetary value / spending",
    "kredi": "Credit / risk",
    "odeme": "Payment / billing type",
    "hizmet": "Service / subscription scope",
    "cihaz": "Device / equipment",
    "demografi": "Demographics",
    "diger": "Other",
}

# raw feature name -> canonical concept (5 datasets; names are generally unique across sets)
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
# note: 'MaritalStatus' collides between cell2cell and ecommerce; both are demografi -> no problem.
HARITA.setdefault("MaritalStatus", "demografi")

KAVRAM_SIRA = ["tenure", "sozlesme", "kullanim", "dusus", "etkilesim", "destek",
               "parasal", "kredi", "odeme", "hizmet", "cihaz", "demografi", "diger"]


def kavram(feature: str) -> str:
    """Maps a raw feature name to its canonical concept (unmappable -> 'diger')."""
    return HARITA.get(feature, "diger")
