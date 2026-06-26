# Spam-/Phishing-Erkennung – Mercedes-Benz AG

Dieses Repository enthält meine Fallstudie für das Modul **Artificial Intelligence** (M.Sc., Hochschule Reutlingen). Ich baue darin mit maschinellem Lernen einen Klassifikator, der eingehende E-Mails automatisch in **Spam** und **legitim** einteilt.

## Worum geht es

E-Mail ist der wichtigste Kommunikationskanal in Unternehmen – und gleichzeitig ein großes Einfallstor für Spam und Phishing. Das kostet Arbeitszeit und ist ein echtes Sicherheitsrisiko. Ich wollte deshalb ein Modell bauen, das verdächtige Mails automatisch erkennt, damit Mitarbeitende und IT entlastet werden. Meine Forschungsfrage war, ob sich allein aus messbaren Merkmalen einer Mail (Wort- und Zeichenhäufigkeiten) zuverlässig vorhersagen lässt, ob es sich um Spam handelt.

## Daten

Ich habe den öffentlichen **Spambase**-Datensatz verwendet (UCI Machine Learning Repository, Hopkins et al. 1999) – 4.601 reale E-Mails mit 57 numerischen Merkmalen und der Zielvariable `spam`. Da interne Firmen-Mails sensibel und nicht öffentlich sind, dient dieser reale Datensatz als realitätsnahe Annäherung – die Fallstudie bildet bewusst keine echten internen Daten der Mercedes-Benz AG ab.

## Mein Vorgehen

Ich habe mich am CRISP-DM-Modell orientiert. Nach der explorativen Analyse (welche Wörter/Zeichen auf Spam hindeuten) habe ich die Daten vorbereitet (stratifizierter 80/20-Split, Standardisierung – ein Encoding war nicht nötig, weil alle Merkmale schon numerisch sind) und drei Modelle verglichen:

- **Logistische Regression** – gut interpretierbares Basismodell
- **Random Forest** – leistungsfähiges Ensemble
- **Neuronales Netz (MLP)** – komplexeres Vergleichsmodell

Weil die Klassen nur leicht unausgeglichen sind (~39 % Spam), habe ich mit `class_weight="balanced"` gearbeitet und zusätzlich die Entscheidungsschwelle untersucht.

## Ergebnisse

Am Ende habe ich mich für den **Random Forest** entschieden. Er erreicht die beste ROC-AUC (0,98) und den besten F1-Score und fängt **91,5 % der Spam-Mails bei nur 3,2 % Fehlalarmen** – für einen Spam-Filter ist diese Balance entscheidend, weil eine fälschlich blockierte legitime Mail genauso ärgerlich ist wie durchgelassener Spam. Die stärksten Spam-Signale sind Sonderzeichen (`!`, `$`), viele Großbuchstaben und Wörter wie *remove*, *free* und *your*.

## Aufbau

- `Spam_Detection.ipynb` – mein Notebook mit dem kompletten Code und allen Ergebnissen
- `spambase.csv` – der verwendete Datensatz
- `Hausarbeit_Spam_Detection.pdf` – die schriftliche Ausarbeitung
- `app.py` – REST-API (FastAPI) als Deployment-Prototyp
- `spam_model.joblib` – das serialisierte, trainierte Modell
- `requirements.txt` – die benötigten Python-Pakete

## Ausführen

Notebook:

```bash
pip install -r requirements.txt
jupyter notebook Spam_Detection.ipynb
```

Der Datensatz muss im selben Ordner wie das Notebook liegen.

Deployment-API starten (lädt `spam_model.joblib`):

```bash
uvicorn app:app --reload
```

Danach ist die interaktive API-Dokumentation unter `http://127.0.0.1:8000/docs` erreichbar.

---
*Erstellt im Rahmen meiner Prüfungsleistung im Modul Artificial Intelligence.*
