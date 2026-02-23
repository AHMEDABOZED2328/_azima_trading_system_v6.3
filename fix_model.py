import zipfile
import json
import os

model_path = '-- 0 -- merged/eurusd_model_vAzImA_26.01.keras'
bak_path = '-- 0 -- merged/eurusd_model_vAzImA_26.01.keras.bak'

if os.path.exists(bak_path):
    os.remove(bak_path)

os.rename(model_path, bak_path)

with zipfile.ZipFile(bak_path, 'r') as zin:
    with zipfile.ZipFile(model_path, 'w') as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == 'config.json':
                text = content.decode('utf-8')
                text = text.replace('"time_major": false,', '')
                text = text.replace('"time_major": true,', '')
                text = text.replace('"time_major": false', '')
                text = text.replace('"time_major": true', '')
                content = text.encode('utf-8')
            zout.writestr(item, content)

print("Keras config fixed successfully.")
