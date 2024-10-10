TARGET2IDX_WTO_BIO = {
    "O": 0,
    "NAME_STUDENT": 1,
    "EMAIL": 2,
    "USERNAME": 3,
    "ID_NUM": 4,
    "PHONE_NUM": 5,
    "URL_PERSONAL": 6,
    "STREET_ADDRESS": 7,
}
TARGET2IDX_WITH_BIO = {
    "O": 0,
    "B-NAME_STUDENT": 1,
    "B-EMAIL": 2,
    "B-USERNAME": 3,
    "B-ID_NUM": 4,
    "B-PHONE_NUM": 5,
    "B-URL_PERSONAL": 6,
    "B-STREET_ADDRESS": 7,
    "I-NAME_STUDENT": 8,
    "I-ID_NUM": 9,
    "I-PHONE_NUM": 10,
    "I-URL_PERSONAL": 11,
    "I-STREET_ADDRESS": 12,
}
IDX2TARGET_WTO_BIO = {v: k for k, v in TARGET2IDX_WTO_BIO.items()}
IDX2TARGET_WITH_BIO = {v: k for k, v in TARGET2IDX_WITH_BIO.items()}
