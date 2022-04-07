#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import sqlite3 as sqldb
import os
from tqdm import tqdm
import sys
import hmac
import hashlib
import bitstring
import time
import datetime
import random
import string
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from multiprocessing import Pool
import argparse
import pickle

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from IDB import *
import os
import time
import numpy as np
import pandas as pd
import psutil
import plotly.express as px
import plotly

from sqlalchemy import create_engine
from lib import *

from UI import *

from cmd import *

global sec
sec = 0

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class WorkThread(QThread): # 工作线程，目的是与图形界面并行
    trigger = pyqtSignal()
    def __init__(self , cmd : str , parent = None):
        super(WorkThread, self).__init__(parent)
        self.cmd = cmd
    def run(self):
        #开始进行循环
        res = os.system( self.cmd )
        self.trigger.emit()

class RunThread( QThread ):
    trigger = pyqtSignal()
    def __init__(self , func , param):
        super(RunThread, self).__init__()
        self.func = func
        self.param = param
        
    def run( self ):
        self.param[0] = DataBase(self.param[0])
        
        self.param[0].connect_to_database()
        self.param = tuple( self.param )
        self.func( *self.param ) 
        self.trigger.emit()
        
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

        self.spinBox.setRange( 0,25000 )
        self.spinBox_2.setRange( 0,300 )
        self.spinBox_3.setRange( 0,64 )
        self.spinBox_4.setRange( 0,16 )

        self.actionOpen_File.triggered.connect(self.open_file)
        self.actionOpen_Database.triggered.connect(self.open_database)
        self.pushButton.clicked.connect( self.startrun )
        self.pushButton_2.clicked.connect( self.checkeffect )
        self.checkBox.stateChanged.connect( self.check )
        self.checkBox_2.stateChanged.connect( self.check )
        self.checkBox_3.stateChanged.connect( self.check )
        self.timer=QTimer()
        self.cnt = 0
        self.timer.timeout.connect(self.flush_time)
        self.progressBar.reset()
        self.actionOpen.triggered.connect( self.open_cmd )
        self.list = []
    def geteffectresult(self): # 获取有效性检测结果
        self.timer.stop()

        twm = load_pickle( f'{ self.table }.ex' )
        exres = twm.exres['o']
        self.insert_string( 'Original DataSet' , 'DisPlay' )
        self.insert_string( f'Mean Absolute Error: {exres[0]}' , 'Got' )
        self.insert_string( f'Mean Squared Error: {exres[1]}' , 'Got' )
        self.insert_string( f'Root Mean Squared Error: {exres[2]}' , 'Got' )

        exres = twm.exres['c']
        self.progressBar.setValue(100)
        self.insert_string( 'Marked DataSet' , 'DisPlay' )
        self.insert_string( f'Mean Absolute Error: {exres[0]}' , 'Got' )
        self.insert_string( f'Mean Squared Error: {exres[1]}' , 'Got' )
        self.insert_string( f'Root Mean Squared Error: {exres[2]}' , 'Got' )

    def checkeffect(self): # 检测数据有效性
        try:
            print( self.data )
        except:
            reply = QMessageBox.warning(self,"警告对话框","请选择原始数据文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
        self.table = self.lineEdit.text()
        if len( self.table ) == 0:
            reply = QMessageBox.warning(self,"警告对话框","请输入数据库中的表",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
        cmd = f'python lib.py -name { self.name } -mode effect  -d {self.data} -t {self.table}'
        self.cnt=0
        self.timer.start(100)
        self.run_worker( cmd , self.geteffectresult )
        
    def open_cmd( self ): # 打开命令行
        os.system( f'wt.exe -d { os.getcwd() }' )
    
    def flush_time(self): # 刷新时间
        self.cnt+=0.1
        self.lcdNumber.display( self.cnt )
        self.progressBar.setValue(self.cnt)
        self.list.append( psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 )
        
    def insert_string(self , new , op = 'opened'): # 打印日志信息
        s = self.textEdit.toPlainText()
        self.textEdit.setPlainText( s + '\n' + f'{op} {new}' )
        
        
    def check( self ): # 检测是否多选
        if self.checkBox.isChecked() + self.checkBox_2.isChecked() + self.checkBox_3.isChecked() > 1:
            reply = QMessageBox.warning(self,"警告对话框","不能勾选多个mode",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            return 
            
    def open_file(self): # 打开文件搜索框
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), 
        "All Files(*);;Text Files(*.txt)")
        print(fileName)
        print(fileType)
        self.insert_string(fileName)
        self.data = fileName;
        
        
    def open_database(self): # 打开数据库
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), 
        "All Files(*);;Text Files(*.txt)")
        print(fileName)
        print(fileType)
        self.insert_string(fileName)
        self.name = fileName;


    def draw_image(self):
        df = pd.DataFrame( { 'time' : np.array(range(len(self.list)))/10 } , index = range(len(self.list)) )
        df['memory usage'] = self.list
        fig = px.line( x= df['time'], y = df['memory usage'] )
        plotly.offline.plot(fig, filename='tmp.html' , auto_open = False)
        self.webEngineView.load(QUrl('file:///tmp.html'))
        
    def timestop(self): # 计时停止
            self.timer.stop()
            
            self.db = create_engine(f"sqlite:///{ self.name }",echo=False)
            self.db.connect()
            df = pd.read_sql_table( self.table , self.db )
            self.model = pandasModel(df)
            self.tableView.setModel( self.model )
            
            self.wm = load_pickle( self.table + '.res' )
            self.insert_string( 'detecting finished!' )
            
            self.insert_string( f'The Result of detecting is { self.wm.hit / self.wm.total * 100 }%' )
            
            self.spinBox.setValue( self.wm.gamma )
            self.spinBox_2.setValue( self.wm.nu )
            self.spinBox_3.setValue( self.wm.xi )
            self.spinBox_4.setValue( self.wm.lsb )
            self.progressBar.setValue(20)
            self.draw_image()
            
    def runstop(self): # 工作线程运行结束获取结果
            self.timer.stop()
            
            self.wm = load_pickle( self.table + '.wm' )
            self.db = create_engine(f"sqlite:///{ self.name }",echo=False)
            self.db.connect()
            df = pd.read_sql_table( self.table , self.db )
            self.model = pandasModel(df)
            self.tableView.setModel( self.model )
            
            self.insert_string( 'finished!' )
            self.progressBar.setValue(20)
            
            self.spinBox.setValue( self.wm.gamma )
            self.spinBox_2.setValue( self.wm.nu )
            self.spinBox_3.setValue( self.wm.xi )
            self.spinBox_4.setValue( self.wm.lsb )
            self.draw_image()
            
    def run_worker( self , cmd , func = None ): # 运行工作线程
        self.worker = WorkThread(cmd)
        self.worker.start()
        self.worker.trigger.connect( func )
        
    def run_runner( self , func , param , slot ): # 失败的运行工作线程
        self.worker = RunThread( func , param )
        self.worker.start()
        self.worker.trigger.connect( slot )
    def startrun( self ): # 开始运行，根据图形界面参数选择运行何种操作
        self.cnt=0
        self.list = []
        self.timer.start(100)
        self.progressBar.reset()
        self.progressBar.setRange(0, 20)
        try:
            print( self.name )
        except:
            reply = QMessageBox.warning(self,"警告对话框","请选择数据库文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.timer.stop()
            return 
        
        if self.checkBox_2.isChecked():
            mode = 'import'
        elif self.checkBox_3.isChecked():
            mode = 'mark'
        elif self.checkBox.isChecked():
            mode = 'detect'
        else:
            reply = QMessageBox.warning(self,"警告对话框","请选择mode",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            #print(reply)
            self.timer.stop()
            return 
        print(mode)
        if mode == 'import':
            try:
                print( self.data )
            except:
                reply = QMessageBox.warning(self,"警告对话框","请选择原始数据文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                self.timer.stop()
                return 
            self.table = self.lineEdit.text()
            if len( self.table ) == 0:
                reply = QMessageBox.warning(self,"警告对话框","请输入数据库中的表",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        
            self.insert_string( 'start importing!' )
            cmd = f'python lib.py -name { self.name } -mode {mode}  -d {self.data} -t {self.table}'

            self.run_worker( cmd , self.runstop )
        if mode == 'mark':
            self.table = self.lineEdit.text()
            if len( self.table ) == 0:
                reply = QMessageBox.warning(self,"警告对话框","请输入数据库中的表",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                self.timer.stop()
                return 
            cols , prime , num = load_pickle( self.table + '.cols' )
            dic = {}
            dic['g'] = self.spinBox.value()
            dic['n'] = self.spinBox_2.value()
            dic['x'] = self.spinBox_3.value()
            dic['l'] = self.spinBox_4.value()
            self.insert_string( 'start marking!' )
            g , n , x , l= dic['g'] , dic['n'] , dic['x'] , dic['l']
            cmd = f'python lib.py -name {self.name} -mode {mode} -t {self.table} -g { g } -n { n } -x { x } -l { l }'
            
            print(cmd)
            self.run_worker( cmd , self.runstop )

        if mode == 'detect':
            self.table = self.lineEdit.text()
            if len( self.table ) == 0:
                reply = QMessageBox.warning(self,"警告对话框","请输入数据库中的表",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                self.timer.stop()
                return 
            self.insert_string( 'start detecting!' )
            cmd = f'python lib.py -name {self.name} -mode {mode} -t {self.table}'
            print(cmd)
            self.run_worker( cmd , self.timestop )

        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
