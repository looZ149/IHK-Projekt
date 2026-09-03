import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(f"""
    # Software Architektur

    ## Frontend für den Benutzer
    - JavaScript mit dash zum einbinden in das Jupyter Notebook, Einbinden in den Editor selber am Ende
    - Einfach gehalten, übersichtlich.
    - Ganze .nd2 Datei hochladen.
        - Möglichkeit mehrere .nd2 files markieren zu lassen.
            - Möglichkeit nur eine oder mehrere Positionen markieren zu lassen
        - Vermutlich Sinnvoll zuerst den letzten Frame der Position zu checken, ob dort ein klares Bild besteht, und dann von hinten nach vorne die ROI markieren und Stabilisieren.

    - Buttons/Dropdowns/Checkboxen - Für die jeweiligen Möglichkeiten.
    - Export funktion für die ROI, bzw. weitergabe an andere Skripts?
        - Vermutlich nicht nötig da es direkt in die Jupyter Notebooks eingebunden wird.

    ## Service
    - Bild einlesen, Format validieren. (Preprocessing)
        - Passt die Auflösung?
        - Normalizing, Resizing
        - Eventuelle Kontrastanpassung?
    - Inferenz - (Prediction?)
        - Ruft das Model(YOLOv26?) auf und übergibt das Bild an dieses.
        - Nimmt die Modelausgabe entgegen und gibt diese weiter.
    - Postprocessing
        - Ergebnisse können wir hier erst Filtern?
            - Sucht die am wahrscheinlich beste ROI Markierung.
        - Falls wir Resizen, müssen wir hier wieder alles zurück auf Originalgröße bringen
            - Könnte problematisch werden?
                - Höhere Fehlerquote bei niedrigeren Pixeln?
        - Fasst Ergebnisse sauber zusammen.
            - Gibt diese eventuell weiter zum Speichern
            - Gibt diese weiter zum Tracking, etc..

    - Speicherung (?)
        - Falls nötig
        - Speichert das Ergebniss ab.

    ## Training - Unwichtig für das Jupyter Notebook(?)
    - Kleiner Code zum Ausführen von Trainingszyklen
    - Führt am Ende automatisch eine Validierung durch
    - Gibt am Ende eine "Erfolgsquote" des Trainings zurück.
    """)
    return


if __name__ == "__main__":
    app.run()
