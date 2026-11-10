# Data

The CSV files are **not** stored in this repository.

`notebooks/wifi_indoor_localization.ipynb` downloads them automatically from:
https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip

## Manual download

1. Go to https://archive.ics.uci.edu/dataset/310/ujiindoorloc and click "Download".
2. Unzip the file.
3. Put `trainingData.csv` and `validationData.csv` in this `data/` folder (when running locally) or in the Colab
   working folder under `data/`.

## Files

| File | Rows | Use in this project |
|---|---|---|
| `trainingData.csv` | 19,937 | training, tuning and model selection (group cross-validation) |
| `validationData.csv` | 1,111 | final test set only, used once per final model |

Each row is one WiFi scan: 520 RSSI columns (`WAP001`–`WAP520`, −104 to 0 dBm, 100 = not detected), plus LONGITUDE,
LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION (1 = inside the room, 2 = outside in front of the door),
USERID, PHONEID and TIMESTAMP.

## Licence and citation

UJIIndoorLoc is licensed under CC BY 4.0.

Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Avariento, J. P., Arnau, T. J., Benedito-Bordonau, M., & Huerta, J. (2014).
UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems.
*2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 261–270.
