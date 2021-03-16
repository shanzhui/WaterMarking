#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pymysql as sqldb # 推荐使用MySQL，因为连接操作比较方便
import time
import datetime
import sys
from PyQt5.QtWidgets import QApplication, QWidget


# In[ ]:





# In[2]:


class DataBase: # 自己封装的数据库操纵类
    
    def __init__(self , db_name , connect = None):
        super(DataBase , self).__init__()
        self.name = db_name
        self.connect = connect
        
    def connect_to_database(self): # 连接至数据库
        self.connect = sqldb.connect(host="localhost", user="root", password="密码", database="数据库名",charset="utf8mb4")
        return self
    
    def commit(self): # 提交事务
        self.connect.commit()
    
    def select_all(self , table_name , columns = []): # 查询全部数据SQL语句
        if self.connect is None:
            print("Please Connect to DataBase First!!!")
            return 
        else:
            if columns == []:
                return self.connect.execute('select * from ' + table_name).fetchall()
            else:
                s = ' '.join(columns)

                return self.connect.execute('select ' + s + ' from ' + table_name).fetchall()
    def update(self , table_name , prime_col , prime_val , column , value): # 根据主键更新数据SQL语句
        if self.connect is None:
            print("Please Connect to DataBase First!!!")
            return 
        else:
            return self.connect.execute('update ' + table_name + ' set ' + column + ' = ' + str(value) + ' where ' + prime_col + ' = ' + str(prime_val)).fetchall()


# In[3]:


db = DataBase('Tmp') # 获取封装的数据库操纵对象


# In[4]:


db.connect_to_database() # 连接至数据库


# In[11]:


db_handle = db
columns = ['bill_ref_id', 'store_ref_id', 'customer_ref_id', 'doctor_ref_id', 'num_drugs_bill',
       'total_quantity_bill', 'mrp_bill', 'total_spend_bill',
       'return_value_bill', 'returned_quantity_bill', 'quantity_ethical',
       'quantity_generic', 'quantity_surgical', 'quantity_ayurvedic',
       'quantity_general', 'quantity_otc', 'quantity_chronic',
       'quantity_acute', 'quantity_h1']
gamma = 100 
nu = 17
xi = 32
db_table = 'Tmp'
total = 0
hit = 0


# In[6]:


key = 'ozfNolowrQ9rpWS5tXNAqLjMk1YRiqDccsxFVqwb6o9q3XuQTHfHF1G18SK8XuzamaQADdGNu0MMkw7yagyKBh25wwjokWumLV0H61hJ9cpi2TzZYEVowlR8UXbwaxof'


# In[ ]:





# In[ ]:





# In[ ]:


def mark( prime_key , col_value , bit_idx ):
        binary_form = float2bin( col_value ) # 将float转化为binary
        
        first_hash = # 获取first_hash，即再在此处进行哈希
        if first_hash % 2 == 0:
            bit = 0
        else:
            bit = 1
        
        ls = list(binary_form)
        ls[bit_idx] = str(bit)
        return bin2float( ''.join(ls) ) # 再将二进制转化为float
    
def detect( prime_key , col_value , bit_idx ):
        binary_form = float2bin( col_value ) # 将float转化为binary
        
        first_hash = # 获取first_hash，即再在此处进行哈希 ，detect与mark都有这一步，得到的哈希值应该是相同的
        if first_hash % 2 == 0:
            bit = 0
        else:
            bit = 1
            
        if binary_form[bit_idx] == str(bit):
            return 1
        else:
            return 0


# In[ ]:


def change_by_line( h , line ):
        
    h_value = # 哈希值转化为整数值
    if h_value % gamma == 0:
        attr_idx = h_value % nu
        if attr_idx == 0:
            attr_idx = 1
        bit_idx = h_value % xi
        print( h , h_value , attr_idx , bit_idx ,  )
        # 以下代码是添加水印
        mark( line[0] , line[attr_idx] , bit_idx ) # 添加水印
        db_handle.update( db_table , columns[0] , line[0] , columns[attr_idx] , new_value ) # 执行更新语句
        # 以下代码是检测水印
        res = detect( line[0] , line[attr_idx] , bit_idx ) # 检测水印
        hit += res # 记录命中
        total += 1 # 记录总数


# In[ ]:





# In[ ]:


lines = db_handle.select_all( db_table ) # 执行SQL语句，获取数据表全部数据
for line in lines:
    h = # 获取哈希值
    change_by_line(  h , line )
db_handle.commit() # 提交数据库更改


# In[14]:


# 下边是图形界面示例，仅仅作为推荐使用PYQT5的样例
if __name__ == '__main__':

    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(512, 512)
    w.move(300, 300)
    w.setWindowTitle('Test')
    w.show()

    sys.exit(app.exec_())


# In[ ]:




