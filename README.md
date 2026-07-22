# Brain CT Annotation Platform
Applicazione locale per l’annotazione di immagini DICOM mediante bounding box.

## Requisiti
Prima dell’installazione assicurarsi di avere:
- Git
- Python 3.11

## Installazione su Windows
Aprire **PowerShell** o il **Prompt dei comandi** ed eseguire:
```bash
git clone https://github.com/samuelepe11/BrainCT_Annotation.git
cd BrainCT_Annotation

py -3.11 -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```
Se il setup è già stato effettuato una volta, è sufficiente eseguire:
```bash
cd CARTELLA_DEL_PROGETTO
source .venv/Scripts/activate
python app.py
```
L’applicazione verrà aperta automaticamente nel browser.
Per interromperla, tornare al terminale e premere:
```text
Ctrl+C
```

## Installazione su macOS o Linux
Aprire il terminale ed eseguire:
```bash
git clone https://github.com/samuelepe11/BrainCT_Annotation.git
cd BrainCT_Annotation

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```
Se il setup è già stato effettuato una volta, è sufficiente eseguire:
```bash
cd CARTELLA_DEL_PROGETTO
source .venv/bin/activate
python app.py
```
L’applicazione verrà aperta automaticamente nel browser.
Per interromperla, tornare al terminale e premere:
```text
Ctrl+C
```

## Utilizzo
1. Preparare un file ZIP contenente le slice DICOM di un singolo paziente.
2. Avviare l’applicazione con:
   ```bash
   python app.py
   ```
3. Caricare il file ZIP e premere **Inizia**.
4. Navigare tra le slice usando lo slider o i pulsanti di navigazione.
5. Disegnare una o più bounding box sulle aree di interesse e elezionare l’etichetta appropriata per ogni bounding box. Non è necessario annotare ogni slice.
6. Premere **Scarica report CSV** per esportare le annotazioni ed, eventualmente, premere **Annota un altro paziente** per iniziare una nuova sessione.

## Formato del file CSV
Il report contiene una riga per ogni bounding box annotata.
Le colonne esportate sono:

| Colonna | Descrizione                             |
|---|-----------------------------------------|
| `slice_idx` | Indice della slice                      |
| `image_name` | Nome del file DICOM                     |
| `height` | Altezza originale dell’immagine         |
| `width` | Larghezza originale dell’immagine       |
| `label` | Etichetta assegnata dall'annotatore     |
| `x_min` | Coordinata sinistra della bounding box  |
| `y_min` | Coordinata superiore della bounding box |
| `x_max` | Coordinata destra della bounding box    |
| `y_max` | Coordinata inferiore della bounding box |

## Privacy
L’applicazione viene eseguita localmente sul computer dell’utente.
- I file DICOM non vengono caricati su server esterni.
- Non viene generato alcun collegamento pubblico.
- Il report CSV viene creato localmente.


## Risoluzione dei problemi

### Il comando `python` non viene riconosciuto
Su Windows provare:
```bash
py -3.11 app.py
```
Su macOS o Linux provare:
```bash
python3 app.py
```

### L’ambiente virtuale non si attiva su PowerShell
Eseguire temporaneamente:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Poi ripetere:
```powershell
source .venv/Scripts/activate
```

### La porta 7860 è già occupata
Chiudere eventuali altre istanze dell’applicazione oppure modificare la porta in `app.py`:
```python
demo.launch(share=False, server_name="127.0.0.1", server_port=7861, inbrowser=True)
```