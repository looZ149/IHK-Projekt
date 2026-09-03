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
    # Entwicklungswerkzeuge

    ## IDE
    Als IDE wird VSCode benutzt. Alternativ könnten wir Jetbrains Pycharm benutzen.
    - VSCode kommt mit Jupyter Integration.
    - Lightweight
    - Anpassbar

    ## Versionskontrolle
    Hierfür benutzen wir GIT. Damit kennen wir uns aus und ist auch die gängigste Wahl. Optional wird GitHub als Backup benutzt.

    ## Annotationtools
    Für die Annotationen der Bilder benutzen wir das acia Framework. Wir werden Datensätze in ein Jupyter Notebook laden und diese dort mithilfe von acia annotieren und speichern.


    # Bibliotheken

    ## ACIA
    Ein Framework welches bereits die Möglichkeiten bietet zu annotieren und die vom Experiment gelieferten nd2 Dateien zu verarbeiten.

    ## Ultralytics-Package
    Bietet uns alles was wir für Training, Validierung, fine-tuning etc.. brauchen. Es liefert uns ebenso die bereits vorhandenen YOLO Modelle auf denen wir aufbauen werden.
    """)
    return


if __name__ == "__main__":
    app.run()
