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
    # Trainingskonfiguration

    ## Batch-size
    Wie viele Trainingsamples werden verarbeitet bevor das Modell die weights und biases anpasst

    ### Kleine Batch-sizes - 8, 16, 32
    - Verbraucht weniger Memory, kann dem Modell helfen sich generell an neue Daten zu "gewöhnen"
    - Längere Trainingszeit

    ### Größere Batch-sizes - 64, 128, 256
    - Schnelle Trainingszeit pro Epoche weil moderne GPUs diese Paralel verarbeiten können
    - Benötigt viel VRAM, kann generalization accuracy verringern falls man die learning rate nicht anpasst.

    ## Epochs
    Eine Epoche ist ein kompletter Trainingsdurchlauf. Jedes Sample aus dem Trainingsset wurde also einmal durch das Modell gegeben.

    - Zu wenig Epochen können underfitting verursachen
        - Das Modell lernt das zugrunde liegende Muster nicht.
    - Zu viele Epochen können overfitting verursachen
        - Das Modell merkt sich zu sehr die Trainingsdaten was probleme bei neuen Daten verursachen kann

    ## Patience/ Early Stopping
    - Wie viele Trainingsrunden (Epochen) oder Validierungen führen wir durch ohne das sich der Validation score erhöht.
    - Wichtig um overfitting vorzubeugen
        - Trainingsskript überwacht eine Metrik, z.B. Validation loss oder accuracy
        - Patience Counter: Wurde eine Epoche abgeschlossen ohne Verbesserung der Metrik, erhöt sich der Counter um 1, ansonsten reset to zero
        - Wird der Counter erreicht, wird das Training gestoppt
        - Viele Frameworks lassen uns die weights zurücksetzen zu der Epoche mit den besten Stats.

    ## Image Size
    Die Bildgröße auf die die Daten skaliert werden für das Training.
    - Letterboxing
        - Das Bild wird verkleinert und es wird versucht das ursprüngliche Seitenverhältnis beizubehalten. So bleiben auch kleinere Objekte unverzerrt
        - Langsamer, höherer Speicherbedarf
    - Cropping
        - Das Bild wird einfach zugeschnitten. Kann natürlich Informationsverlust bedeuten, besonderst bei kleinen Strukturen.

    ## Learning Rate
    Kontrolliert wie stark das Model seine Weights im laufe des Trainings anpasst. Gibt die Schrittgröße vor die der Optimizer geht um die besten Modelparameter zu finden.
    - Ist die Learning rate zu niedrig, lernt das Model nicht genug in jedem Schritt.
    - Ist die Learning rate zu hoch können wir schnell overfitting bekommen.
    - Step size
        - Gibt an wie aggresiv oder vorsichtig der Optimizer vorgeht
    - Loss function
        - Misst wie falsch die Aussage des Models gegenüber der Wahrheit ist.
        - Liefert den Gradient (Die Richtung) in dem die Gewichte angepasst werden sollen

    ## Optimizer
    Der Algorithmus der die Weights Updates berechnet. YOLO nutzt hier standardmäßig MuSGD

    - MuSGD
        - Schnell und stabil durch sein Hybrid Design zwischen SGD und Muon-Style matrix updates

    - Adam - Adaptive Moment Estimation
        - Passt die Learning rate individuell für jeden Parameter an.

    - AdamW
        - Funktioniert wie Adam, koppelt aber die Weight Decay (Gewichtsabnahme) direkt von den Gradient-Updates ab.
        - Standart für große Modelle, preferiert über Adam

    ## Freeze / Layer-Freezing
    Wie viele der früheren Schichten werden eingefroren und nicht weiter angepasst

    ## Augmentierungsparameter
    Konkrete Werte für Horizontal/Vertical Flips, Rotationswinkel, Helligkeitsjitter, etc..

    # Validierungskonzepte

    ## Split-Strategie
    - Es wird Session/Experiment basiert gesplitted, anstatt Splits basierend auf einzelnen Bildern anzulegen.
    - Durch diesen Split verhindern wir Data Leakage da das Modell diese Aufnahmesituation vorher noch nicht hatte. Bei Bild basierten Splits aus einer Session währe die Aufnahmesituation zumindest sehr ähnlich.
    - Training 60%, Validierung 20%, Test 20% der insgesammt verfügbaren Sessions.
    - Wichtig zu beachten ist die gesamt Zahl der Frames pro Split. Sodass wir am Ende nicht viel mehr Frames im z.B Test-Split haben als im Trainings-Split

    ## Holdout Methode
    - Ein fixer Split. Dieser bleibt identisch über den gesamten Trainingsverlauf.

    ## K-Fold Cross Validation
    - Der Datensatz wird in K gleich große Teile geteilt. Es wird K mal trainiert.
    - Hier werden die Sets rotiert. Sodass am Ende jedes Set mal als als Testset benutzt wurde

    ## Group K-Fold
    - Bilder werden in Gruppen geordnet.
    - Sessions sind hier direkt schon unsere Gruppen

    ## Leave-One-Out Cross Validation (LOOCV)
    - Extremform von K-Fold, K ist hier die Anzahl der Sessions.
    - Jede Session wird einmal als alleiniges Validationset genutzt, der Rest ist Training.
    - Sehr gründlich, sehr rechenintensiv.
    - So viele Trainingsläufe wie Sessions

    ## Nested Cross-Validation
    - Cross Validation in Cross Validation
    - Eine äußere Schleife für Modellbewertung, eine innere Schleife für Hyperparameter-tuning, damit die Hyperparameter nicht versehentlich auf den Testdaten übertrainiert werden.
    - Sehr aufwändig.

    ## Time Series Split/ Walk-forward Validation
    - Bei zeitlich geordneten Daten, werden nur Daten vor einem gewissen Zeitpunkt für Training genutzt. Und Daten nach diesem Zeitpunkt werden für Validierung und Test benutzt.

    ## Bootstrap Validation
    - Wiederholtes zufälliges Ziehen mit zurücklegen aus dem Datensatz

    ## Auswahl

    Holdout Methode wird vermutlich ausreichen. Ist aufjedenfall am unkompliziertesten.
    Group K-Fold könnten wir benutzen falls die Holdout Methode nicht ausreichend ist und unzufriedene Testergebnisse liefert.
    """)
    return


if __name__ == "__main__":
    app.run()
