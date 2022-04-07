from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from AttackExperiment import *


class ExWin( QDialog , Ui_Form ):
    def __init__(self) -> None:
        super(ExWin ,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect( self.get_param )
        
        self.spinBox.setPrefix( "Run " )
        self.spinBox.setSuffix( " times" )
        self.param = None

        self.doubleSpinBox.setRange( 1,100 )
        self.doubleSpinBox_2.setRange( 1,100 )
        self.doubleSpinBox_3.setRange( 1,100 )
        self.doubleSpinBox.setPrefix( "Satrt From " )
        self.doubleSpinBox_2.setPrefix( "To " )
        self.doubleSpinBox_3.setPrefix( "Fixing " )
        

        self.checkBox_6.stateChanged.connect( self.set_for_ratio )
        self.checkBox_5.stateChanged.connect( self.set_for_lsb )
    def set_for_ratio(self):
        if self.checkBox_6.isChecked():
            self.doubleSpinBox.setRange( 1,100 )
            self.doubleSpinBox.setPrefix( "Satrt From " )
            self.doubleSpinBox.setSuffix( " %" )

            self.doubleSpinBox_2.setRange( 1,100 )
            self.doubleSpinBox_2.setPrefix( "To " )
            self.doubleSpinBox_2.setSuffix( " %" )

            self.doubleSpinBox_3.setRange( 1,64 )
            self.doubleSpinBox_3.setPrefix( "Fixing " )
            self.doubleSpinBox_3.setSuffix( " bit" )
    def set_for_lsb( self ):
        if self.checkBox_5.isChecked():
            self.doubleSpinBox.setRange( 1,64 )
            self.doubleSpinBox.setPrefix( "Satrt From " )
            self.doubleSpinBox.setSuffix( " bit" )

            self.doubleSpinBox_2.setRange( 1,64 )
            self.doubleSpinBox_2.setPrefix( "To " )
            self.doubleSpinBox_2.setSuffix( " bit" )

            self.doubleSpinBox_3.setRange( 1,100 )
            self.doubleSpinBox_3.setPrefix( "Fixing " )
            self.doubleSpinBox_3.setSuffix( " %" )
    def check_ratio_lsb_checkbox( self ):
        if self.checkBox_5.isChecked() + self.checkBox_6.isChecked()  > 1:
            reply = QMessageBox.warning(self,"警告对话框","只能选择一种遍历",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            return False
        return True

    def check_ratio_lsb_is_checked( self ):
        if self.checkBox_5.isChecked():
            runmode = 'lsb'
        elif self.checkBox_6.isChecked():
            runmode = 'ratio'
        else:
            reply = QMessageBox.warning(self,"警告对话框","请选择遍历方式",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        return runmode
    def check_multi( self ):
        if self.checkBox.isChecked() + self.checkBox_2.isChecked() + self.checkBox_3.isChecked() + self.checkBox_4.isChecked() > 1:
            reply = QMessageBox.warning(self,"警告对话框","不能勾选多个mode",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
            return False
        return True
    def check_is_checked( self ):
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
            return False
        return mode
    def check_input( self ):
        if self.param[1] < self.param[0] :
            reply = QMessageBox.warning(self,"警告对话框","范围出错",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        
        if self.param[2] == 0:
            reply = QMessageBox.warning(self,"警告对话框","实验次数应该是正整数",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return False
        
        return True
        
    def get_param( self ):
        if not self.check_ratio_lsb_checkbox():
            return 

        if not self.check_multi() :
            return 
        mode = self.check_is_checked()
        if not mode:
            return 
        runmode = self.check_ratio_lsb_is_checked()
        if not runmode:
            return 
        start = self.doubleSpinBox.value()
        end = self.doubleSpinBox_2.value()
        fix = self.doubleSpinBox_3.value()
        time = self.spinBox.value()

        self.param = ( start , end , time , mode , fix , runmode )

        if not self.check_input():
            return 

        self.close()

