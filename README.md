## The Structure of Source Code
#### Partition of Marking and Detecting from some kind of DataBase
mark.py <--------- IDB.py , lib.py
#### Partition of Attacking to some kind of DataBase
attack.py <------- AUI.py , lib.py
#### The core code file , which is the imported tool library
lib.py
#### Graphical interface Template code , which is imported for creating special UI
UI.py ----------- AUI.py ------------- AEUI.py

## Imported open-sourcce library
bitstring==3.1.9
numpy==1.20.1
pandas==1.2.4
plotly==5.6.0
psutil==5.8.0
PyQt5==5.15.6
SQLAlchemy==1.4.7
tqdm==4.59.0

## command line
python lib.py -name data -mode import  -d train.csv -t T
python lib.py -name data -mode mark -t T
python lib.py -name data -mode detect -t T
python lib.py -name data -mode effect  -d data.csv -t T
