#!/usr/bin/env python
# coding: utf-8

# In[72]:

import pandas as pd
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

from multiprocessing import Pool
import argparse
import pickle

# In[2]:
def to_pickle( df , path ):
    f = open( path , 'wb' )
    pickle.dump(df , f)
    f.close()
def load_pickle( path ):
    f = open( path , 'rb' )
    df = pickle.load( f )
    f.close()
    return df

class DataBase: # 管理与数据库的连接类
    
    def __init__(self , db_name , connect = None):
        super(DataBase , self).__init__()
        self.name = db_name
        self.connect = connect
        
    def connect_to_database(self):
        self.connect = sqldb.connect(self.name)
        return self
    
    def commit(self):
        self.connect.commit()
    
    def select_all(self , table_name , columns = []):
        if self.connect is None:
            print("Please Connect to DataBase First!!!")
            return 
        else:
            if columns == []:
                return self.connect.execute(f'select * from {table_name}').fetchall()
            else:
                s = ' '.join(columns)
                # print(f'select {s} from {table_name}')
                return self.connect.execute(f'select {s} from {table_name}').fetchall()
    def update(self , table_name , prime_col , prime_val , column , value):
        if self.connect is None:
            print("Please Connect to DataBase First!!!")
            return 
        else:
            # print(f'update {table_name} set {column} = {value} where {prime_col} = {prime_val}')
            return self.connect.execute(f'update {table_name} set {column} = {value} where {prime_col} = {prime_val}').fetchall()


# In[67]:


class WaterMark: # 管理数据库与水印相关属性信息的类
    
    def __init__(
        self , 
        columns = [] ,
        key = None , 
        db_handle = None , 
        db_table = 'Temperature' , 
        gamma = 100 , 
        nu = 101 ,
        xi = 32 , 
        method = 'mark' , 
        prime_idx = 0 ,
        lsb = 8 , 
    ):
        super(WaterMark, self).__init__()
        self.gamma = gamma
        self.nu = nu
        self.xi = xi
        self.db_handle = db_handle
        self.key = key
        self.columns = columns
        self.db_table = db_table
        self.prime_idx = prime_idx
        self.method = method
        
        self.charhex_dict = { str(i) : i for i in range( 0 , 10 ) }
        self.charhex_dict['a'] = 10
        self.charhex_dict['b'] = 11
        self.charhex_dict['c'] = 12
        self.charhex_dict['d'] = 13
        self.charhex_dict['e'] = 14
        self.charhex_dict['f'] = 15
        
        self.total = 0
        self.hit = 0
        
        self.f = sys.stdout
        self.row_cnt = 0
        self.progress = None
        self.lsb = lsb
        self.marked = False
    
    def set_method(self , method):
        self.method = method
        return method
    
    def set_gamma( self , new_value ):
        self.gamma = new_value
        return self.gamme
    
    def set_nu( self , new_value ):
        self.nu = new_value
        return self.nu
    
    def set_xi( self , new_value ):
        self.xi = new_value
        return self.xi
    
    def set_lsb( self , new_value ):
        self.lsb = new_value
        return self.lsb


# 


# In[17]:


def df_split( df , length = 1000 ):
    ls = []
    for i in range( int(len(df) / length) + 1 ):
        lf = i*length
        rt = (i+1)*length
        ls.append( df.iloc[lf:rt] )
    return ls


# In[23]:




wm = None
# In[56]:
def transform_2_10_scale(wm , from_str , from_scale  ):
        return sum( [ wm.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
def float2binary(wm , f):
        return bitstring.BitArray(float=f, length=64).bin
def binary2float(wm ,  b ):
        return bitstring.BitArray(bin=b).float
def mark_line_( param ):
        prime_key , col_value , bit_idx , wm = param
        binary_form = float2binary( wm , col_value )
        
        hash_value = hashlib.md5()
        hash_value.update(wm.key + str(prime_key).encode('utf-8'))
        
        first_hash = transform_2_10_scale(wm , hash_value.hexdigest(),16)
        if first_hash % 2 == 0:
            bit = '0'
        else:
            bit = '1'
        
        ls = list(binary_form)# [ str(int(e)) for e in list(binary_form) ]
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key)
        second_hash = transform_2_10_scale(wm , hash_value.hexdigest(),16)
        lidx = second_hash % wm.lsb
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key + str(prime_key).encode('utf-8'))
        third_hash = transform_2_10_scale(wm , hash_value.hexdigest(),16)
        
        method = third_hash % 4
        if method == 0:
            if ls[bit_idx] == '1' and bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 1:
            if ls[bit_idx] == '1' or bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 2:
            if ls[bit_idx] != bit:
                bit = 0
            else:
                bit = 1
        else:
            if ls[bit_idx] == bit:
                bit = 0
            else:
                bit = 1
        # print( ls[-lidx] , ',' , bit )
        ls[-lidx] = str(bit)
        # print( first_hash % 2 , second_hash % wm.lsb , third_hash % 4 )
        # print( method , lidx , bit )
        
        return binary2float( wm , ''.join(ls) )

def multimark( param , pb = None  ): # 并发添加水印
    wm , df = param
    def transform_2_10_scale(from_str , from_scale  ):
        return sum( [ wm.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
    def float2binary(f):
        return bitstring.BitArray(float=f, length=64).bin
    def binary2float( b ):
        return bitstring.BitArray(bin=b).float
    def mark_line( prime_key , col_value , bit_idx ): # 对一行添加水印
        binary_form = float2binary( col_value )
        
        hash_value = hashlib.md5()
        hash_value.update(wm.key + str(prime_key).encode('utf-8'))
        
        first_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        if first_hash % 2 == 0:
            bit = '0'
        else:
            bit = '1'
        
        ls = list(binary_form)# [ str(int(e)) for e in list(binary_form) ]
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key)
        second_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        lidx = second_hash % wm.lsb
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key + str(prime_key).encode('utf-8'))
        third_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        
        method = third_hash % 4
        if method == 0:
            if ls[bit_idx] == '1' and bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 1:
            if ls[bit_idx] == '1' or bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 2:
            if ls[bit_idx] != bit:
                bit = 0
            else:
                bit = 1
        else:
            if ls[bit_idx] == bit:
                bit = 0
            else:
                bit = 1
        # print( ls[-lidx] , ',' , bit )
        ls[-lidx] = str(bit)
        # print( first_hash % 2 , second_hash % wm.lsb , third_hash % 4 )
        # print( method , lidx , bit )
        
        return binary2float( ''.join(ls) )
    realindex = 0
    for line in tqdm( df.iterrows() ):
        idx = line[0]
        line = line[1]
        h = hmac.new(wm.key, str(line[wm.prime_idx]).encode('utf-8'), digestmod='MD5').hexdigest()
        h_value = transform_2_10_scale( h , 16 )
        if h_value % wm.gamma == 0:
            attr_idx = h_value % wm.nu
            if attr_idx == 0:
                attr_idx = 1
            bit_idx = h_value % wm.xi
            new_value = mark_line( line[wm.prime_idx] , line[df.columns[attr_idx]] , bit_idx )
            old = line[line.index[attr_idx]]
            # line[line.index[attr_idx]] = new_value
            df.loc[idx,df.columns[attr_idx]] = new_value
            # print( old , new_value , df.iloc[ realindex ][df.columns[attr_idx]] )
            # wm.db_handle.update( wm.db_table , wm.columns[wm.prime_idx] , line[wm.prime_idx] , wm.columns[attr_idx] , new_value )
            # print(f'{datetime.datetime.now()} : {line.index[attr_idx]} : {old} -> {new_value}\n' )
            wm.total += 1
        realindex += 1
    return df , wm.total


def mark( param , pb = None  ): # 非并发添加水印
    wm , df = param

    params = []
    for line in tqdm( df.iterrows() ):
        idx = line[0]
        line = line[1]
        h = hmac.new(wm.key, str(line[wm.prime_idx]).encode('utf-8'), digestmod='MD5').hexdigest()
        h_value = transform_2_10_scale( h , 16 )
        if h_value % wm.gamma == 0:
            attr_idx = h_value % wm.nu
            if attr_idx == 0:
                attr_idx = 1
            bit_idx = h_value % wm.xi
            params.append( [ [ idx , attr_idx , line[line.index[attr_idx]] ] , [ line[wm.prime_idx] , line[df.columns[attr_idx]] , bit_idx , wm ] ] )
    pool = Pool(20)
    res = pool.map(mark_line_, [ p[1] for p in params ] )
    pool.close()
    pool.join()

    for i in tqdm( range( len(params) ) ):
        idx,attr_idx,old = params[i][0]
        df.loc[idx,df.columns[attr_idx]] = res[i]
        print( old , res[i] )
    wm.total = len(res)
    return df


# In[57]:


def get_key(length = 128):
    return "".join( [ random.sample(string.ascii_letters+string.digits,1)[0] for i in range(length) ] ).encode('utf-8')


# In[69]:



def detect( param , pb = None ): # 并发检测水印
    wm , df = param
    def transform_2_10_scale(from_str , from_scale  ):
        return sum( [ wm.charhex_dict[ch] * (from_scale ** i) for ch , i in zip(from_str[::-1] , range(len(from_str))) ] )
    def float2binary(f):
        return bitstring.BitArray(float=f, length=64).bin
    def binary2float( b ):
        return bitstring.BitArray(bin=b).float
    def detect_line( prime_key , col_value , bit_idx ): # 检测一行水印
        binary_form = float2binary( col_value )
        
        hash_value = hashlib.md5()
        hash_value.update(wm.key + str(prime_key).encode('utf-8'))
        
        first_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        if first_hash % 2 == 0:
            bit = '0'
        else:
            bit = '1'
        
        ls = list(binary_form)# [ str(int(e)) for e in list(binary_form) ]
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key)
        second_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        lidx = second_hash % wm.lsb
        
        hash_value = hashlib.md5()
        hash_value.update(str(prime_key).encode('utf-8') + wm.key + str(prime_key).encode('utf-8'))
        third_hash = transform_2_10_scale(hash_value.hexdigest(),16)
        
        method = third_hash % 4
        if method == 0:
            if ls[bit_idx] == '1' and bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 1:
            if ls[bit_idx] == '1' or bit == '1':
                bit = 1
            else:
                bit = 0
        elif method == 2:
            if ls[bit_idx] != bit:
                bit = 0
            else:
                bit = 1
        else:
            if ls[bit_idx] == bit:
                bit = 0
            else:
                bit = 1
        return ls[-lidx] == str(bit)
    if pb is not None:
        pb.setRange(0, len(df)-1)
        pb.reset()
    wm.total = 0
    wm.hit = 0
    for line in tqdm( df.iterrows() ):
        if pb is not None:
            pb.setValue(line[0])
        line = line[1]
        h = hmac.new(wm.key, str(line[wm.prime_idx]).encode('utf-8'), digestmod='MD5').hexdigest()
        h_value = transform_2_10_scale( h , 16 )
        if h_value % wm.gamma == 0:
            attr_idx = h_value % wm.nu
            if attr_idx == 0:
                attr_idx = 1
            bit_idx = h_value % wm.xi
            flag = detect_line( line[wm.prime_idx] , line[df.columns[attr_idx]] , bit_idx )
            if flag:
                wm.hit += 1
            wm.total += 1
    return df , wm.hit , wm.total

# In[73]:


args = argparse.ArgumentParser(description = 'The Program of marking and detecting watermark',epilog = 'Information end ')


# In[75]:


args.add_argument( '-name' , type = str , help = 'path of database' , default = '' )
args.add_argument( '-m' , '-mode' , type = str , help = 'choose marking or detecting',
                  choices=['mark', 'm' , 'detect' , 'd' , 'import' , 'i' , 'effect' , 'e'] )
args.add_argument( '-g' , '-gamma' , type = int , help = 'set gamma' , default = -1  )
args.add_argument( '-n' , '-nu' , type = int , help = 'set nu' , default = -1 )
args.add_argument( '-x' , '-xi' , type = int , help = 'set xi' , default = -1  )
args.add_argument( '-l' , '-lsb' , type = int , help = 'set lsb' , default = -1 )
args.add_argument( '-d' , '-data' , type = str , help = 'path of original data' , default = '' )
args.add_argument( '-t' , '-table' , type = str , help = 'name of table' , default = '' )
args.add_argument( '-p' , '-prime' , type = str , help = 'name of prime key' , default = 'time' )
# args.add_argument( '-c' , '-config' , type = str , help = 'path of configuration file' , default = '' )

# In[76]:

def dodetect(  db , table , pb = None  ): # 检测水印，调用并发检测水印的函数
    twm = load_pickle( table + '.wm' )
    twm.db_handle = db
    twm.f = sys.stdout
    wm = twm
    data = pd.read_sql_query( f'select * from {table}' , db.connect )
    # detect( [wm,data] , pb )
    wm.db_handle = None
    wm.f = None

    dfs = df_split( data , int( len(data) / 8 + 1 ) ) 
    pool = Pool(8)
    res = pool.map(detect, [ [wm,df] for df in dfs ] )
    pool.close()
    pool.join()
    data = pd.concat([ p[0] for p in res ]).reset_index()
    del data['index']
    wm.hit = sum( [ p[1] for p in res ] )
    wm.total = sum( [ p[2] for p in res ] )
    print( wm.hit /  wm.total )
    
    to_pickle( wm , table + '.res' )
    return wm

def prepare_data( data , feas , target ):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data[ feas ],
                                                    data[target],
                                                    test_size=0.2,
                                                    random_state=0)
    return X_train, X_test, y_train, y_test
def build_rfr():
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    regr = RandomForestRegressor(max_depth=3 , n_estimators=100 , random_state=2022 , n_jobs=-1)
    pipe = Pipeline([ ('reduce_dim', PCA()), ('scaler', StandardScaler()),
                 ('regressor', regr)])
    return pipe
def run_model( model , x , y , test ):
    model.fit( x , y )
    return model.predict( test )

def run_pipe( data , feas , target ):
    from sklearn import metrics
    X_train, X_test, y_train, y_test = prepare_data( data , feas , target )
    y_hat = run_model( build_rfr() , X_train , y_train , X_test )
    exres = [ metrics.mean_absolute_error(y_test, y_hat) , 
    metrics.mean_squared_error(y_test, y_hat) , np.sqrt(metrics.mean_squared_error(y_test, y_hat)) ]
    print('Mean Absolute Error:', exres[0] )
    print('Mean Squared Error:', exres[1] )
    print('Root Mean Squared Error:', exres[2] )
    return exres

def checkeffect( db , path , table , prime ):
    origin = pd.read_csv( path ).fillna(0)
    #print( origin.columns )
    if 'train.csv' in path or 'sd.csv' in path:
        origin.columns = [ c.replace('(' , '_').replace(')' , '') for c in origin.columns ]
    data = pd.read_sql_query( f'select * from {table}' , db.connect ).fillna(0)

    feas = [c for c in data.columns if c != prime ]
    feas , target = feas[:-1] , feas[-1:]

    res_dic = {}
    res_dic['o'] = run_pipe( origin , feas , target )
    res_dic['c'] = run_pipe( data , feas , target )
    wm = load_pickle( table + '.wm' )
    wm.exres = res_dic
    print(res_dic)
    to_pickle( wm , table + '.ex' )
    
def import_data( db , path = 'train.csv' ,  table_name = 'Temperature' , config = '.cols' ): # 导入数据
    data = pd.read_csv( path )
    data = data.fillna(0)
    data.columns = [ c.replace('(' , '_').replace(')' , '') for c in data.columns ]
    # print( data.columns )
    ls = list( zip( data.columns , data.dtypes ) ) 
    s = f'{ls[0][0]} {str(ls[0][1])} primary key ,' + ' '  + ' '.join( [ f'{tp[0]} { str(tp[1]) },' for tp in ls[1:-1] ] ) + ' ' + f'{ls[-1][0]} {str(ls[-1][1])}'

    cmd = f'CREATE TABLE  {table_name}({s});'
    # print(cmd)
    try:
        db.connect.execute( cmd )
    except:
        pass
    data.to_sql( table_name , con = db.connect , if_exists='replace' , index = False  )
    db.commit()
    to_pickle( [data.columns , dic['p'] , len(data)] , table_name + config )

def domark( db , table , columns , prime_idx = 'time' , dic = {} , configurepath = '.wm' , pb = None  ): # 添加水印，调用并发添加水印的函数
    wm = WaterMark(
        db_handle = db , columns = columns , prime_idx='time' , gamma = dic['g'],
        nu = dic['n'] , 
        xi = dic['x'] , 
        lsb = dic['l']
    )
    wm.key = get_key()
    data = pd.read_sql_query( f'select * from {table}' , db.connect )
    print(data.shape)
    wm.db_handle = None
    wm.f = None
    # data = mark([wm,data] , pb)
    start_time = time.time()
    dfs = df_split( data , int( len(data) / 8 + 1 ) ) 
    pool = Pool(8)
    print( time.time() - start_time , 's' )
    res = pool.map(multimark, [ [wm,df] for df in dfs ] )
    print( time.time() - start_time , 's' )
    pool.close()
    pool.join()
    print( time.time() - start_time , 's' )
    data = pd.concat([ p[0] for p in res ]).reset_index()
    del data['index']
    wm.total = sum( [ p[1] for p in res ] )
    # wm.marked = True
    data.to_sql( table , con = db.connect , if_exists='replace' , index = False  )
    
    to_pickle( wm , table + configurepath ) 


def main( dic ): # 主函数，参数解析，跳转执行相关的函数
    start_time = time.time()
    if len(dic['name']) == 0:
        print( 'please input database path if you want to connect to one database' )  
    db = DataBase(dic['name'])
    db.connect_to_database()
    if dic['m'] == 'import' or dic['m'] == 'i':
        if len(dic['d']) == 0:
            print( 'please input data path if you want to import into database from other data' )
            return 
        if len(dic['t']) == 0:
            print( 'please input table name if you want to import into database from other data' )
            return 
        import_data(  db ,  dic['d'] ,  dic['t'] )
    elif dic['m'] == 'mark' or dic['m'] == 'm':
        if len(dic['t']) == 0:
            print( 'please input table name if you want to import into database' )
            return 
        cols , prime , num = load_pickle( dic['t'] + '.cols' )
        if dic['g'] == 0 or dic['g'] == -1:
            dic['g'] = np.random.randint( len(cols) ,num/10 )
        if dic['n'] == 0 or dic['n'] == -1:
            dic['n'] = np.random.randint( len(cols)/2 , len(cols) )
        if dic['x'] == 0 or dic['x'] == -1:
            dic['x'] = np.random.randint( 16 , 64 )
        if dic['l'] == 0 or dic['l'] == -1:
            dic['l'] = np.random.randint( 8 , 16 )
        domark( db ,  dic['t'] ,  cols , prime , dic )
    elif dic['m'] == 'detect' or dic['m'] == 'm':
        if len(dic['t']) == 0:
            print( 'please input data path if you want to import into database' )
            return 
        dodetect( db , dic['t'] )
    elif dic['m'] == 'effect' or dic['m'] == 'e':
        if len(dic['d']) == 0:
            print( 'please input data path if you want to import into database from other data' )
            return 
        if len(dic['t']) == 0:
            print( 'please input table name if you want to import into database from other data' )
            return 
        checkeffect( db , dic['d'] , dic['t'] , dic['p'] )
        
    print(dic)
    print( f"elapsed time :  { time.time() - start_time }s" )
if __name__=="__main__":
    parse = args.parse_args()
    dic = parse.__dict__
    main(dic)
    
