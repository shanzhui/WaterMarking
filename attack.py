import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from cv2 import line
from IDB import *
import os
import time
import numpy as np
import pandas as pd
import psutil
import plotly.express as px
import plotly
import shutil

from sqlalchemy import create_engine
from lib import *

from AUI import *

from cmd import *

from AEUI import *


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
class Value:
        def __init__(self):
            pass
        def set_df( self , df ):
            self.df = df
            return self
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self): # 初始化函数
        super(Main, self).__init__()
        self.setupUi(self)
        self.progressBar.reset()
        self.progressBar_2.reset()
        self.progressBar_3.reset()
        self.charhex_dict = { str(i) : i for i in range( 0 , 10 ) }
        self.charhex_dict['a'] = 10
        self.charhex_dict['b'] = 11
        self.charhex_dict['c'] = 12
        self.charhex_dict['d'] = 13
        self.charhex_dict['e'] = 14
        self.charhex_dict['f'] = 15
        self.checkBox.stateChanged.connect( self.check )
        self.checkBox_2.stateChanged.connect( self.check )
        self.checkBox_3.stateChanged.connect( self.check )
        self.checkBox_4.stateChanged.connect( self.check )
        self.list = []
        self.actionOpenDataBase.triggered.connect(self.open_database)
        self.actionExperiment.triggered.connect( self.run_experiment )
        self.pushButton.clicked.connect( self.attack )
        self.ewindows = ExWin()

        self.isex = False
    def run( self , mode ): # 运行一次攻击与检测
        self.progressBar.reset()
        self.list = []
        self.progressBar.setRange( 0 , self.time-1 )
        for epoch in range( int(self.time) ):
            self._backup()
            if not self._attack(mode):
                return False
            self._check(mode)
            if not self.isex:
                self._image()
            
            self.progressBar.setValue( epoch )
        return True

    def _imageforexperiment( self , x , y ): # 为实验结果生成图片
        fig = px.line( x= x, y = y )
        name = f'experiment-{self.eparam}.html'
        plotly.offline.plot(fig, filename= name, auto_open = False)
        self.webEngineView.load(QUrl(f'file:///{name}'))
        print('Image name : ' , name)

    def run_experiment(self): # 运行实验
        if not self.open_experiment_window():
            return
        self.time = self.eparam[2]
        self.progressBar_3.setRange( int(self.eparam[0]) , int(self.eparam[1]) )
        self.isex = True
        tmps = []
        res = []
        for tmp in tqdm( range( int(self.eparam[0]) , int( self.eparam[1] ) + 1 ) ):
            if self.eparam[5] == 'ratio':
                self.ratio = tmp
                self.lsb = self.eparam[4]
            else:
                self.lsb = tmp
                self.ratio = self.eparam[4]
            tmps.append(tmp)
            self.insert_string( f'ratio :{self.ratio} lsb :{self.lsb}' , 'Running' )

            if not self.run( self.eparam[3] ):
                break
            self.progressBar_3.setValue( tmp )
            self.insert_string( '' , '\n' )
            res.append( np.mean( self.list ) )
        self._imageforexperiment( tmps , res )
        self.isex = False
        

    def open_experiment_window( self ): # 打开做实验的UI界面
        self.ewindows.exec()

        if self.ewindows.param is None:
            reply = QMessageBox.warning(self,"警告对话框","请正确输入实验设置",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        else:
            self.eparam = self.ewindows.param
            return True
        
    def open_database(self): # 打开数据库
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), 
        "All Files(*);;Text Files(*.txt)")
        print(fileName)
        print(fileType)
        self.insert_string(fileName)
        self.name = fileName;
    
    def check( self ): # 检测是否勾选多个mode
        if self.checkBox.isChecked() + self.checkBox_2.isChecked() + self.checkBox_3.isChecked() + self.checkBox_4.isChecked() > 1:
            reply = QMessageBox.warning(self,"警告对话框","不能勾选多个mode",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
            return 
    
    def input_data( self ): # 对UI界面输入的数据进行检测
        try:
            print(self.name)
        except:
            reply = QMessageBox.warning(self,"警告对话框","请打开数据库文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        self.table = self.lineEdit.text()
        if len( self.table ) == 0:
            reply = QMessageBox.warning(self,"警告对话框","请输入表名",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        if not self.isex: # 不是实验模式，则需要检测，否则，不需要检测，因为实验窗口需要设置下边的参数
            self.ratio = self.doubleSpinBox.value()
            if self.ratio == 0:
                reply = QMessageBox.warning(self,"警告对话框","请输入攻击比率",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                return False
            self.lsb = self.doubleSpinBox_2.value()
            if self.lsb == 0:
                reply = QMessageBox.warning(self,"警告对话框","请输入lsb",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
                return False
        
        return True
    def prepare( self ): # 攻击前的数据准备
        self.db = create_engine(f"sqlite:///{ self.backup }",echo=False)
        self.db.connect()
        self.df = pd.read_sql_table( self.table , self.db )
        self.model = pandasModel(self.df)
        self.tableView.setModel( self.model )
        self.key = get_key()

    def delete_attack( self ): # 删除攻击
        def transform_2_10_scale(from_str , from_scale  ):
            return sum( [ self.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
        def float2binary(f):
            return bitstring.BitArray(float=f, length=64).bin
        def binary2float( b ):
            return bitstring.BitArray(bin=b).float
        if not self.input_data():
            return False
        
        self.prepare()
        idx = list( self.df.index )
        np.random.shuffle(idx)
        sam = idx[ : int( self.ratio / 100 * len(idx) ) ]
        self.progressBar_2.reset()
        self.progressBar_2.setRange( 0 , 1 )
        self.df = self.df.iloc[sam]
        self.progressBar_2.setValue(1)
        self.df.to_sql( self.table , con = self.db , if_exists='replace' , index = False )
        return True
    def add_attack( self ): # 添加攻击
        def transform_2_10_scale(from_str , from_scale  ):
            return sum( [ self.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
        def float2binary(f):
            return bitstring.BitArray(float=f, length=64).bin
        def binary2float( b ):
            return bitstring.BitArray(bin=b).float
        if not self.input_data():
            return False
        
        self.prepare()
        idx = list( self.df.index )
        np.random.shuffle(idx)
        sam = int( self.ratio / 100 * len(idx) )
        cols , prime , num = load_pickle( self.table + '.cols' )
        dic = {}
        for col in cols:
            dic[ col ] = ( self.df[col].min() - 1 , self.df[col].max() + 1 )
        d = []
        self.progressBar_2.reset()
        self.progressBar_2.setRange( 0 , sam-1 )
        cnt = 0
        for i in tqdm( range(sam) ):
            tmp = []
            for col in cols:
                if 'int' in str( self.df[col].dtype ):
                    tmp.append( np.random.randint( *dic[col] ) )
                else:
                    tmp.append( np.random.uniform( *dic[col] ) )
            d.append( tmp )
            '''if cnt % 1000 == 0:
                self.progressBar_2.setValue(cnt)'''
            cnt += 1
        self.progressBar_2.setValue(sam-1)
        adddf = pd.DataFrame( d , columns = cols )
        self.df = pd.concat( [ self.df , adddf ] ).reset_index()
        del self.df['index']
        ls = list( self.df.index )
        np.random.shuffle( ls )
        self.df = self.df.iloc[ ls ].reset_index()
        del self.df['index']
        self.df.to_sql( self.table , con = self.db , if_exists='replace' , index = False )
        return True
    def update_attack(self): # 更新攻击
        def transform_2_10_scale(from_str , from_scale  ):
            return sum( [ self.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
        def float2binary(f):
            return bitstring.BitArray(float=f, length=64).bin
        def binary2float( b ):
            return bitstring.BitArray(bin=b).float
        if not self.input_data():
            return False
        
        self.prepare()
        
        idx = list( self.df.index )
        np.random.shuffle(idx)
        sam = idx[ : int( self.ratio / 100 * len(idx) ) ]
        cols , prime , num = load_pickle( self.table + '.cols' )
        
        self.progressBar_2.reset()
        self.progressBar_2.setRange( 0 , len(sam)-1 )
        cnt = 0
        for i in tqdm(sam):
            for col in self.df.columns:
                if col == prime:
                    continue
                if 'int' in str( self.df[col].dtype ):
                    self.df.iloc[i][col] = np.random.randint( self.df[col].min() - 1 , self.df[col].max() + 1 )
                else:
                    self.df.iloc[i][col] = np.random.normal() * self.df[col].max()
            '''if cnt % 1000 == 0:
                self.progressBar_2.setValue(cnt)'''
            cnt += 1
        self.progressBar_2.setValue(len(sam)-1)
        self.df.to_sql( self.table , con = self.db , if_exists='replace' , index = False )
        return True
    
    def random_attack_(self): # 并发执行的函数，与下边的random_attack配合使用
        def transform_2_10_scale(from_str , from_scale  ):
            return sum( [ self.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
        def float2binary(f):
            return bitstring.BitArray(float=f, length=64).bin
        def binary2float( b ):
            return bitstring.BitArray(bin=b).float
        def binary2int( b ):
            return bitstring.BitArray(bin=b).int

        cols , prime , num = load_pickle( self.table + '.cols' )

        cnt = 0
        cls = list(range(len(self.df.columns)))
        # print(ls)
        for i , line in tqdm( self.df.iterrows() ):
            for c in cls:
                # print(c)
                col = cols[c]
                if col == prime:
                    continue
                binary_form = float2binary( float( line[col] ) )
                ls = list(binary_form)
                def change( i , v ):
                    hash_value = hashlib.md5()
                    hash_value.update(str(v).encode('utf-8') + str(i).encode('utf-8') + self.key)
                    value = transform_2_10_scale(hash_value.hexdigest(),16)
                    if (value + np.random.randint( 0 , self.ratio )) % int( 100 / self.ratio + 0.5 ) == 0 :
                        if str(i) == '0':
                            return '1'
                        else:
                            return '0'
                    else:
                        return ls[i]
                ls = [ change(i , self.df.iloc[i][col]) for i in range(len(ls)) ]
                if 'int' in str( self.df[col].dtype ):
                    self.df.loc[i,col] = binary2int( ''.join(ls) )
                else:
                    self.df.loc[i,col] = binary2float( ''.join(ls) )
            cnt += 1
        return self.df
    def random_attack(self): # 执行并发的随机攻击
        if not self.input_data():
            return False
        self.prepare()
        tmp = Value()
        tmp.charhex_dict = self.charhex_dict
        tmp.df = self.df
        tmp.table = self.table
        tmp.key = self.key
        tmp.ratio = self.ratio

        self.progressBar_2.reset()
        self.progressBar_2.setRange( 0 , 1 )

        dfs = df_split( self.df , int( len(self.df) / 8 + 1 ) ) 
        pool = Pool(8)
        res = pool.map(Main.random_attack_, [ tmp.set_df(df) for df in dfs ] )
        pool.close()
        pool.join()
        self.df = pd.concat(res).reset_index()
        del self.df['index']

        self.progressBar_2.setValue( 1 )
        self.df.to_sql( self.table , con = self.db , if_exists='replace' , index = False )

    def _random_attack( self ): # 旧版本随机攻击方式，仅作参考，不会运行
        def transform_2_10_scale(from_str , from_scale  ):
            return sum( [ self.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
        def float2binary(f):
            return bitstring.BitArray(float=f, length=64).bin
        def binary2float( b ):
            return bitstring.BitArray(bin=b).float
        def binary2int( b ):
            return bitstring.BitArray(bin=b).int
        if not self.input_data():
            return False
        
        self.prepare()
        

        idx = list( self.df.index )
        np.random.shuffle(idx)
        sam = idx[ : int( self.ratio / 100 * len(idx) ) ]
        cols , prime , num = load_pickle( self.table + '.cols' )
        
        self.progressBar_2.reset()
        self.progressBar_2.setRange( 0 , len(sam)-1 )
        cnt = 0
        for i in tqdm( sam ):
            ls = list(range(len(self.df.columns)))
            np.random.shuffle(ls)
            ls = ls[0:int(len(self.df.columns)*self.ratio)]
            for col in ls:
                col = self.df.columns[col]
                if col == prime:
                    continue
                binary_form = float2binary( float( self.df.iloc[i][col] ) )
                ls = list(binary_form)
                def change( i , v ):
                    hash_value = hashlib.md5()
                    hash_value.update(str(v).encode('utf-8') + str(i).encode('utf-8') + self.key)
                    value = transform_2_10_scale(hash_value.hexdigest(),16)
                    if value + np.random.randint( 0 , self.ratio ) == 0 :
                        if str(i) == '0':
                            return '1'
                        else:
                            return '0'
                    else:
                        return ls[i]
                ls = [ change(i , self.df.iloc[i][col]) for i in range(len(ls)) ]
                if 'int' in str( self.df[col].dtype ):
                    self.df.iloc[i][col] = binary2int( ''.join(ls) )
                else:
                    self.df.iloc[i][col] = binary2float( ''.join(ls) )
            
            # self.df.iloc[i][col] = binary2float( ''.join(ls) )
            '''if cnt % 1000 == 0:
                self.progressBar_2.setValue(cnt)'''
            cnt += 1
        self.progressBar_2.setValue(len(sam)-1)
        self.df.to_sql( self.table , con = self.db , if_exists='replace' , index = False )
        return True
    def _attack( self , mode ): # 根据关键词跳转到具体执行的攻击函数
        if mode == 'update':
            return self.update_attack()
        elif mode == 'delete':
            return self.delete_attack()
        elif mode == 'random':
            return self.random_attack()
        elif mode == 'add':
            return self.add_attack()
    def _backup( self ): # 备份数据
        try:
            shutil.copyfile( self.name , 'backup' )
            self.backup = 'backup'
        except:
            reply = QMessageBox.warning(self,"警告对话框","备份出错",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
    
    def _check( self , mode ): # 检测命令有效性与结果展示
            cmd = f'python lib.py -name {self.backup} -mode detect -t {self.table}'
            print('cmd :' ,  cmd)
            os.system( cmd )
            self.db = create_engine(f"sqlite:///{ self.backup }",echo=False)
            self.db.connect()
            df = pd.read_sql_table( self.table , self.db )
            self.model = pandasModel(df)
            self.tableView.setModel( self.model )
            
            self.wm = load_pickle( self.table + '.res' )
            self.insert_string( 'detecting finished!' , 'run' )
            
            self.insert_string( f'The Result of detecting is { self.wm.hit / self.wm.total * 100 }%' , 'Get' )
            self.list.append( self.wm.hit / self.wm.total * 100 )
    def _image( self ): # 正常攻击生成与加载图片
        df = pd.DataFrame( { 'epoch' : np.array(range(len(self.list)))/10 } , index = range(len(self.list)) )
        df['correct'] = self.list
        fig = px.line( x= df['epoch'], y = df['correct'] )
        plotly.offline.plot(fig, filename='backup.html' , auto_open = False)
        self.webEngineView.load(QUrl('file:///backup.html'))
    def insert_string(self , new , op = 'opened'): # 在UI界面打印信息
        s = self.textEdit.toPlainText()
        self.textEdit.setPlainText( s + '\n' + f'{op} {new}' )
        
    def attack( self ): # 选择攻击模式
        if self.checkBox.isChecked():
            mode = 'update'
        elif self.checkBox_2.isChecked():
            mode = 'delete'
        elif self.checkBox_3.isChecked():
            mode = 'random'
        elif self.checkBox_4.isChecked():
            mode = 'add'
        else:
            reply = QMessageBox.warning(self,"警告对话框","请选择mode",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return
        self.time = self.doubleSpinBox_3.value()
        if self.time == 0:
            reply = QMessageBox.warning(self,"警告对话框","请输入实验次数",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        self.run(mode)
            

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())