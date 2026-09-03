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
    # Models
    - YOLOv26 ist das neuste und schnellste Model. Es arbeitet ohne NMS
        - Non-Maximum-Surpression (NMS) ist ein Nachbearbeitungsschritt.
    - YOLOv11 ist ein gutes Model, mit Checkpoints für jede Aufgabe, mit NMS.
    - V26 ist wohl sehr gut auch ohne NMS und auch schnell. V11 hingegen gibt es lange, ist scheinbar ein bisschen genauer aber benötigt auch mehr Rechenleistung und ist eventuell langsamer.
    - NMS könnte bei dichten Populationen für Problemme sorgen. Sollte aber wenig Auswirkung auf die Markierung der ROI haben.
    - V26 Nano hat den Vorteil das ich mit weniger Trainingsdaten zu einem schnelleren guten Ergebnisse komme. Kann langfristig aber zu ungenau werden. Ich könnte mit Nano anfangen und debuggen und dann eventuell auf das große Modell wechseln. Dafür hingegen brauche ich aber viele Trainingsdaten

    - Es gibt noch andere Modelle wie:
        - RT-DETR, transformer based, NMS-free
        - Faster R-CNN, größer, 2-step-detection, eher veraltete Methode
        - EfficientDet, von Google, eher seltener in benutzung
        - MMDetection Framework, gibt Zugang zu vielen Architekturen unter einer API
        - Cellpose / StarDist - extra für Zellbilder gebaut worden.
        - SAM, von META, promptbased segmentation. Vielleicht nützlich für die Annotationen?

    # Fazit
    - Da es sich erstmals nur um eine ROI-Markierung handelt scheint YOLOv26 erstmals die beste Option zu sein.
    - Cellpose / StarDist sollte man definitiv in betracht ziehen sobald Segmentierung ein Thema wird.
    """)
    return


if __name__ == "__main__":
    app.run()
