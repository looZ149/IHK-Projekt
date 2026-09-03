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
    # Datenmodell

    ## AnalyzedObject Klasse
    - Name/ID/POS ID
    - Koordinaten der ROI
    - Wie viele Frames ?
    - Ausgabe als tiff

    ## Ordnerstruktur

    - src/
    - src/analyze/
    - src/output/pos_#/roi_#
    - src/dataset/images/training
    - src/dataset/images/validation
    - src/dataset/images/test
    - src/dataset/labels/training
    - src/dataset/labels/validation

    ## Bildvorverarbeitung - Preprocessing

    - Eingabeformat Prüfen und vereinheitlichen
        - Dateien kommen als nd2 an.
            - Direkt in tiffs umwandeln um möglichst viel Daten zu erhalten
        - YOLO erwartet 8-Bit, nd2 liefert 12 oder 16 Bit, also runter rechnen

    - Resizing
        - YOLO erwartet 640x640 Pixel
        - Resizing mit Letterboxing
            - Schützt Proportionen
            - Keine Bildinhalte gehen verloren.
        - ND2 Bilder sind vermutlich sehr hochauflösend. Schauen wie genau wir Resizen können.
            - ggf. Tiling anwenden

    - Channels anpassen
        - YOLO erwartet 3-Channel-RGB
            - Wir copy pasten den Greychannel einfach auf alle 3
            - Standartansatz, reicht wohl meistens aus

    - Normalisierung um das Training zu vereinfachen - Macht YOLO automatisch
        - Eventuelle Kontrastnormalisierung, jenachdem wie die Belichtung variiert

    ## Augmentierung

    - Vertikale & Horizontale Flips, Rotierung
    - Helligkeitsänderungen, ggf. um Belichtungseinstellungen zwischen verschiedenen Experimenten zu simulieren?

    ## Validierungsregeln
    - Error-Handling
        - Was passiert wenn falsche Bilddatei, Bild leer, kaputtes Bild etc
    """)
    return


if __name__ == "__main__":
    app.run()
