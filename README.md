# Introduzione

Repository contenente il codice per l'addestramento di un classificatore binario per la Human Activity Recognition di balli caraibici. Clicca [qui](./resources/reports/) per informazioni dettagliate.

In questo documento viene descritto come utilizzare il codice nel repository.

# Eseguire notebooks e scripts

Consiglio di creare un virtual environment con Python 3.10 o superiore e installarvi le dipendenze contenute in `requirements.txt`.

# Descrizione dei notebooks

- `estimate-poses.ipynb`: data una cartella di video in formato mp4, effettua la stima delle pose e produce dei file json in output.
- `evaluate-medium-dataset-swstep2.ipynb`: 
    1. data una cartella di json contenenti le pose stimate, crea il dataset csv contenente le sliding window (in realtà vi è un csv per ogni json, quindi il csv non è unico);
    2. addestra più modelli sul training set e valuta le prestazioni sul test set.
- `cross-validation-medium-swstep2.ipynb`: esegue una stratified group cross-validation di una Random Forest e calcola le curve di apprendimento all'aumentare della dimensione del dataset.

# Descrizione degli scripts di visualizzazione

> Per interrompere l'esecuzione degli script è necessario premere "q".

- `visualizer.py` consente di visualizzare le pose estratte da YOLO. Possono essere visualizzati un massimo di 4 file **json** in contemporanea. Ogni file può essere visualizzato in due modalità: *poses* oppure *boxes*. Nella gif che segue, quella di sinistra è *boxes*, quella di destra è *poses*.
![](./resources/media/filtered_vs_unfiltered.gif)
Per eseguire lo script: `python visualizer.py <config file path>` dove `<config file path>` è il percorso di un file json in cui sono specificati da 1 a 4 video da visualizzare in contemporanea. Ciascuna specifica di video deve contenere le seguenti proprietà:
    - `path`: percorso del json da visualizzare;
    - `display_poses` e `display_boxes`: booleani che specificano la modalità di visualizzazione. **Sono mutuamente esclusivi**, quindi se uno è true l'altro deve essere false;
    - `scale`: booleano, deve essere true se e solo se la modalità di visualizzazione è *boxes* e le pose del video non sono state scalate nell'intervallo [0,1];
    - `include_face`: booleano, deve essere true se e solo se le pose contenute nel video non hanno i punti chiave relativi al viso.

- `visualizer_sliding_window.py` consente di visualizzare tutte le sliding window contenute in un file **csv**, come nel video che segue:
![](./resources/media/sliding_windows.gif)
Per eseguire lo script: `python visualizer_sliding_window.py <config file path>` dove `<config file path>` è il percorso di un file json contenente le seguenti proprietà:
    - `path`: percorso del csv da visualizzare;
    - `delay_between_frames`: valore intero, è consigliabile lasciarlo a 25 per avere una visualizzazione a circa 40FPS nei video "poco carichi" di persone. Si può ridurre tale valore se si nota che lo script va a scatti, o lo si può aumentare se va troppo veloce.