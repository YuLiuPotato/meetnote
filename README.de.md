# meetnote

[English](README.md) | [简体中文](README.zh-CN.md) | **Deutsch**

Besprechungsaufnahme → **Transkript mit Sprechern** + **ausführliche Zusammenfassung** + **Kernpunkte auf zwei Seiten**, mit einem Befehl.

- **Transkription** läuft lokal mit faster-whisper (large-v3) und nutzt automatisch eine NVIDIA-GPU. Eine Stunde Audio dauert wenige Minuten; ffmpeg wird nicht gebraucht.
- **Sprechertrennung** läuft lokal auf der CPU (sherpa-onnx + 3D-Speaker CAM++), gleichzeitig mit der Transkription.
- **Protokolle** schreibt ein Sprachmodell: standardmäßig DeepSeek, alternativ jeder OpenAI-kompatible Dienst oder Claude. Jede Aussage trägt einen Zeitstempel `[mm:ss]`, und ein separater Durchgang prüft den Entwurf gegen das Transkript, bevor er überarbeitet wird.
- Funktioniert mit deutschen, englischen, chinesischen und gemischtsprachigen Besprechungen: Jeder Teil bleibt in der Sprache, in der er gesprochen wurde. Die Oberfläche (Meldungen, Webseite, Dateinamen) gibt es auf Deutsch, Englisch und Chinesisch.

## Installation

Voraussetzungen: Windows und Python 3.10+. Mit einer NVIDIA-GPU geht es deutlich schneller.

```powershell
git clone https://github.com/YuLiuPotato/meetnote.git
cd meetnote
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Das Skript legt eine virtuelle Umgebung in `meetnote\.venv` an und einen Starter `meetnote.cmd` in `%USERPROFILE%\.local\bin` (dieser Ordner muss im PATH liegen). Das Whisper-Modell (~3 GB) und die Sprechermodelle (~35 MB) werden beim ersten Einsatz heruntergeladen.

### Konfiguration: `.env`

API-Schlüssel, Port und Zugangscode stehen in `meetnote\.env` (bei der Installation aus `.env.example` erzeugt), nicht in Systemvariablen. In einem Texteditor öffnen und eintragen:

```ini
DEEPSEEK_API_KEY=sk-ihr-schluessel
# Oberfläche auf Deutsch (sonst wie die Anzeigesprache des Systems)
MEETNOTE_LANG=de
```

Die Oberflächensprache (Meldungen, `--help`, Berichte und Dateinamen) folgt der Anzeigesprache des Systems; mit `MEETNOTE_LANG=de`, `en` oder `zh` in `.env` lässt sie sich festlegen. Sie ist unabhängig von der Sprache des Protokolls.

Suchreihenfolge: die Datei aus `MEETNOTE_ENV_FILE` → `.env` im aktuellen Ordner → `meetnote\.env`. Bereits gesetzte Umgebungsvariablen haben Vorrang. `.env` steht in `.gitignore`; nie weitergeben.

**Für Protokolle braucht es ein Sprachmodell-Backend** (`--backend auto` nimmt das erste konfigurierte in dieser Reihenfolge):

| Backend | In `.env` | Standardmodell |
|---|---|---|
| `deepseek` (empfohlen) | `DEEPSEEK_API_KEY` (von platform.deepseek.com) | `deepseek-chat`; `--model deepseek-reasoner` geht auch |
| `openai` | `MEETNOTE_BASE_URL` + `MEETNOTE_API_KEY` + `MEETNOTE_MODEL`, für jeden OpenAI-kompatiblen Dienst (OpenAI, Mistral, Kimi, Qwen, …) | frei wählbar |
| `claude-cli` | nichts; braucht das eigenständige Claude Code mit `claude` im PATH, angemeldet | Ihre Claude-Code-Einstellungen |
| `claude-api` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |

## Verwendung

```powershell
# Kompletter Durchlauf
meetnote run "D:\Aufnahmen\Teamsitzung.m4a"

# Namen/Fachbegriffe und etwas Hintergrund verbessern Erkennung und Protokoll
meetnote run Teamsitzung.m4a --terms "Kubernetes,OKR,Anna Müller" --context "Quartalsplanung des Produktteams"

# Wenn die Anzahl der Teilnehmenden bekannt ist, angeben: die Sprechertrennung wird deutlich zuverlässiger
meetnote run Teamsitzung.m4a --speakers 3

# Protokoll in einer anderen Sprache als die Aufnahme (Standard: wie die Aufnahme)
meetnote run Teamsitzung.m4a --summary-lang de

# Nur transkribieren / Protokoll aus vorhandenem Transkript neu erstellen (z. B. nach Änderung der Prompts)
meetnote transcribe Teamsitzung.m4a
meetnote summarize Teamsitzung_Protokoll

# Eine Besprechung in mehreren Dateien (z. B. die WAV-Teile eines Diktiergeräts): zu einer MP3 zusammenfügen und transkribieren.
# Dateien in der angegebenen Reihenfolge; Ordnerinhalt in natürlicher Namensreihenfolge (REC2 vor REC10)
meetnote run REC001.WAV REC002.WAV REC003.WAV
meetnote run D:\Aufnahmen\2026-11-20\

# Nur zu einer MP3 zusammenfügen, ohne Transkription
meetnote merge D:\Aufnahmen\2026-11-20\ -o Besprechung.mp3

# Sprecherkennungen durch Namen ersetzen, dann Protokoll neu erstellen
meetnote rename Teamsitzung_Protokoll S1=Anna S2=Jonas
meetnote summarize Teamsitzung_Protokoll
```

Die Ergebnisse landen im aktuellen Ordner unter `<Dateiname>_Protokoll\` (bei `MEETNOTE_LANG=de`; auf Englisch `<Dateiname>_notes\`, auf Chinesisch `<Dateiname>_纪要\`). Ein Ordner, der schon Dateien enthält, behält seine Benennung. Die Sprache des Protokollinhalts richtet sich nach `--summary-lang`.

```
Teamsitzung_Protokoll\
  Transkript.txt       Zeitstempel und Sprecher, z. B. "[01:23] S2: …"; neue Zeile bei jedem Sprecherwechsel
  Transkript.srt       Untertitel, zum Abgleich mit dem Audio in einem Player
  Notizen.md           Abschnittsnotizen: Meinungen/Fakten/Entscheidungen, je mit Zeitstempel und Bezug
  Zusammenfassung.md   die ganze Besprechung der Reihe nach, mit Belegstellen
  Kernpunkte.md        Ergebnisse zuerst, passt auf zwei A4-Seiten, mit Belegstellen
  Faktencheck.md       was der Faktencheck gefunden und korrigiert hat
  meta.json            Dauer, Sprache, Sprecher, Quelldatei
```

### Sprachen

- Die Sprache der Aufnahme wird **automatisch und abschnittsweise erkannt**: In einer gemischt deutsch-englischen Besprechung bleibt Deutsch Deutsch und Englisch Englisch, statt übersetzt zu werden. Jeder Redebeitrag wird an seinem eigenen Audio nachgeprüft und neu transkribiert, wenn Whisper ihn in der falschen Sprache geschrieben hat (bei Deutsch/Englisch anhand typischer Funktionswörter). Sprechen die Teilnehmenden erkennbar verschiedene Sprachen, wird die Aufnahme **Redebeitrag für Redebeitrag** transkribiert, jeder in seiner eigenen Sprache: Sonst hält Whisper pro 30-Sekunden-Fenster an einer Sprache fest und lässt die andere weg oder übersetzt sie. Mit `--language de` (oder `en`, `zh`, …) lässt sich eine Sprache erzwingen.
- Typische Whisper-Halluzinationen in Sprechpausen („Untertitel im Auftrag des ZDF“, „Vielen Dank fürs Zuschauen“) werden herausgefiltert.
- Das Protokoll folgt der Sprache der Aufnahme, außer `--summary-lang` ist gesetzt.

### Sprecher

Zeilen werden in der Reihenfolge des ersten Auftretens mit `S1`, `S2`, … gekennzeichnet. Das Protokoll ordnet nach Kennung zu („S1 argumentiert …“) und rät nie echte Namen; stellt sich jemand vor oder wird mit Namen angesprochen, steht dort z. B. „S2 (stellt sich als Anna vor)“.

- **Anzahl angeben** mit `--speakers N` (auf der Webseite „Teilnehmende“). Ohne Angabe wird sie geschätzt.
- **Echte Namen**: Im Transkript nachsehen, was jede Kennung gesagt hat, dann `meetnote rename` (auf der Webseite: „Sprecher · Namen eintragen“). Wurde eine Person auf zwei Kennungen aufgeteilt, beiden denselben Namen geben.
- Funktionsweise: pyannote segmentation-3.0 (über sherpa-onnx) findet Sprache und Sprecherwechsel; alle ~1,5 s Sprache gibt es einen CAM++-Stimmabdruck; die Fenster werden nach Stimmähnlichkeit gruppiert (Average-Linkage-Clustering). Kleine Gruppen werden nur dann einem Sprecher zugeschlagen, wenn sie ähnlich klingen; sonst bleiben sie eigene Sprecher, sodass auch Personen mit wenig Redeanteil (z. B. Fragende in einer Fragerunde) erhalten bleiben. Sprache, die die Sprechererkennung gehört, Whisper aber ausgelassen hat, wird separat transkribiert. Kein Hugging-Face-Token nötig; die Modelle liegen in `%USERPROFILE%\.cache\meetnote\models`.
- `--no-diarize` schaltet die Sprechertrennung ab.
- Grenzen: Bei gleichzeitigem Sprechen bekommt eine Zeile nur einen Sprecher; Einwürfe aus ein, zwei Wörtern werden oft falsch zugeordnet; Raum- und Telefonaufnahmen schneiden schlechter ab.

### Protokoll und Genauigkeit

Der Standardablauf `grounded` zielt auf einen Fehler besonders: Meinungen, die dem falschen Produkt, der falschen Version oder der falschen Person zugeschrieben werden.

1. **Extrahieren**: Das Transkript wird in ~10-Minuten-Abschnitte geteilt; jede Meinung wird mit *Bezug* und Zeitstempel notiert („Bezug unklar“, wenn er sich nicht bestimmen lässt).
2. **Mit Belegen schreiben**: Zusammenfassung und Kernpunkte entstehen aus diesen Notizen, jede Aussage mit `[mm:ss]`.
3. **Faktencheck**: Ein separater Aufruf prüft jede Aussage gegen das Transkript auf unbelegte Aussagen, falschen Bezug, falschen Sprecher, falsche Richtung/Zahl und Übergeneralisierung.
4. **Überarbeiten** nur dort, wo der Faktencheck etwas gefunden hat.

`--pipeline simple` ist der alte Ein-Durchgang-Modus: schneller und günstiger, aber bei langen Besprechungen verwechselt er mehr.

Quellentreue des Protokolls bewerten (Aussage für Aussage gegen das Transkript; bewertet standardmäßig von `deepseek-reasoner`):

```powershell
meetnote check Teamsitzung_Protokoll
```

Das Ergebnis steht in `Quellentreue.md`. Auch Sprachmodelle als Bewerter irren (TofuEval fand binäre „belegt?“-Urteile selbst mit GPT-4 unzuverlässig), daher eignet es sich zum Vergleich von Versionen, nicht als absolute Note. Wie der Ablauf bewertet und verbessert wurde: [docs/accuracy-evaluation.md](docs/accuracy-evaluation.md) (Englisch).

Existiert schon ein Transkript, überspringt `run` die Transkription; mit `--force` wird neu transkribiert.

Bei mehreren Dateien wird die zusammengefügte Aufnahme im Ausgabeordner als `<Name>.mp3` gespeichert und als eine Besprechung transkribiert. `meta.json` vermerkt, wo jede Datei beginnt (`parts`), und das Protokoll erfährt von den Übergängen, die oft Pausen sind. Eingaben dürfen beliebige Formate und Abtastraten haben; alles wird auf Abtastrate und Kanäle der ersten Datei umgerechnet. `meetnote merge --bitrate 96` setzt die MP3-Bitrate (Standard 128 kbps, bei Mono halbiert).

## Optionen

| Option | Wirkung |
|---|---|
| `-o ORDNER` | Ausgabeordner |
| `--terms` | Namen und Fachbegriffe, durch Kommas getrennt, für bessere Erkennung |
| `--context` | Hintergrund zur Besprechung, um Begriffe zu korrigieren und den Kontext zu verstehen |
| `--only synopsis/highlights` | nur eines der beiden Dokumente erstellen |
| `--backend` | Backend erzwingen: `deepseek` / `openai` / `claude-cli` / `claude-api` |
| `--model` | Modellname |
| `--whisper-model medium` | schnellere, etwas ungenauere Transkription |
| `--device cpu` | ohne GPU (deutlich langsamer) |
| `--language de` | Sprache der Aufnahme festlegen (Standard: erkennen) |
| `--speakers 3` | Anzahl der Teilnehmenden (Standard: schätzen) |
| `--no-diarize` | Sprecher nicht trennen |
| `--summary-lang de` | Sprache des Protokolls (Standard: wie die Aufnahme) |
| `--pipeline simple` | ein Durchgang: schneller, ungenauer |
| mehrere Dateien oder ein Ordner | zuerst zu einer MP3 zusammenfügen, dann transkribieren |

`MEETNOTE_CLAUDE` in `.env` kann auf die ausführbare Datei `claude` zeigen.

## Webdienst (für Kolleginnen und Kollegen)

Auf einem Rechner mit NVIDIA-GPU betreiben; Kolleginnen und Kollegen laden Aufnahmen im Browser hoch und lesen oder laden die Protokolle herunter, ohne etwas zu installieren. Die Seite lässt sich oben rechts zwischen Deutsch, Englisch und Chinesisch umschalten und folgt standardmäßig der Browsersprache.

- Aufträge werden in eine Warteschlange gestellt und einzeln transkribiert; eine Stunde Audio dauert 3–5 Minuten.
- Jeder Auftrag hat einen nicht erratbaren Link (`/#job=…`), der geteilt werden kann.
- Die Originalaufnahme wird nach der Transkription gelöscht; nur Text bleibt, standardmäßig 30 Tage (`--keep-days`).
- Schlägt das Zusammenfassen fehl, bleibt das Transkript erhalten, und die Seite bietet „Protokoll neu erstellen“ an.
- Unfertige Aufträge laufen nach einem Neustart weiter.
- Mehrere Dateien lassen sich als ein Auftrag hochladen (z. B. die WAV-Teile eines Diktiergeräts); sie werden in Dateinamen-Reihenfolge zu einer Aufnahme zusammengefügt. Mit `--keep-audio` kann die zusammengefügte MP3 auf der Auftragsseite heruntergeladen werden.

### 1. Einrichten (einmalig)

In `meetnote\.env` (der Schlüssel bleibt auf dem Server; Kolleginnen und Kollegen brauchen ihn nicht):

```ini
DEEPSEEK_API_KEY=sk-ihr-schluessel
# optional: Zugangscode, der beim Öffnen der Seite abgefragt wird
MEETNOTE_ACCESS_CODE=selbst-waehlen
# optional: Port, Standard 18765
# MEETNOTE_PORT=18765
```

Nach Änderungen an `.env` den Dienst neu starten.

### 2. Starten

`start-server.cmd` doppelklicken oder im Terminal ausführen. Der Dienst lauscht auf `http://localhost:18765` (Port aus `.env`) und speichert Daten in `%USERPROFILE%\meetnote-data`. Die erste Logzeile zeigt, welche `.env` gelesen wurde.

Beim Anmelden automatisch starten (nach der Anmeldung öffnet sich ein Dienstfenster; Schließen beendet den Dienst):

```powershell
schtasks /Create /TN meetnote /SC ONLOGON /TR "<meetnote-Ordner>\start-server.cmd"
```

Dient ein Laptop als Server, „Im Netzbetrieb in den Energiesparmodus wechseln nach“ auf „Nie“ stellen.

### 3. Zugang für Kolleginnen und Kollegen: Tailscale (empfohlen)

Tailscale auf dem Server installieren und anmelden, dann den Dienst nur innerhalb Ihres Tailnets (nicht im Internet) über den HTTPS-Proxy von Tailscale freigeben:

```powershell
tailscale serve --bg 18765
```

(Denselben Port wie in `.env` verwenden.)

Danach `https://<Rechner>.<Ihr-Tailnet>.ts.net` öffnen (`tailscale serve status` zeigt die vollständige Adresse). Zwei Wege, andere hineinzulassen:

- **Ins Tailnet einladen**: Admin-Konsole → Users → Invite users.
- **Nur diesen Rechner teilen**: Admin-Konsole → Machines → dieser Rechner → Share. Die Eingeladenen nehmen mit ihrem eigenen Tailscale-Konto an und erreichen nur diesen Rechner.

Kolleginnen und Kollegen brauchen ein installiertes, angemeldetes Tailscale. Freigabe beenden mit `tailscale serve --https=443 off`.

### Alternative: Büronetz

Im gemeinsamen lokalen Netz geht es ohne Tailscale: `MEETNOTE_HOST=0.0.0.0` in `.env` setzen, den Dienst starten, und Kolleginnen und Kollegen öffnen `http://<LAN-IP des Servers>:18765`. Zusätzlich muss der Port in der Windows-Firewall freigegeben werden (eingehend, privates Netzwerk). Das ist unverschlüsseltes http; daher auch einen Zugangscode setzen.

## Ausgabe anpassen

Aufbau und Regeln beider Dokumente stehen in Prompt-Dateien, die sich direkt bearbeiten lassen (die Installation ist editierbar, Änderungen wirken sofort):

- `meetnote\prompts\synopsis.md`: ausführliche Zusammenfassung
- `meetnote\prompts\highlights.md`: Kernpunkte auf zwei Seiten

Die Prompts sind auf Chinesisch geschrieben; die Sprache des Protokolls wird separat über `--summary-lang` festgelegt, die der Oberfläche über `MEETNOTE_LANG`.

## Bekannte Grenzen

- Sprecherkennungen sind automatisch, und manche kurze Zeile bekommt den falschen Sprecher; wichtige „wer hat was gesagt“-Stellen im Transkript nachprüfen.
- 8 GB Grafikspeicher reichen für eine Transkription gleichzeitig: Läuft die Kommandozeile, während der Webdienst transkribiert, werden beide sehr langsam.
- Transkripte gehen an den konfigurierten Sprachmodell-Anbieter; bei vertraulichen Besprechungen bedenken.
- Namen, Organisationen und gemischtsprachige Begriffe werden oft falsch gehört. `--terms` hilft, und die Kernpunkte enden mit einer Liste zu prüfender Namen.
- Temporäre Dateien von Apps wie WeChat werden automatisch gelöscht; Aufnahmen vor der Verarbeitung woanders speichern.

## Lizenz und Danksagung

Code unter der [MIT](LICENSE)-Lizenz. Aufgebaut auf [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) und [3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker). Die Modelle werden beim ersten Einsatz heruntergeladen, nicht mit diesem Repository verteilt und unterliegen ihren eigenen Lizenzen.
