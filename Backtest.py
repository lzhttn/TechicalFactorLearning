# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:45:22 2020

@author: L
"""
import numpy as np
import lightgbm as lgb
import pandas as pd
import os
import datetime
import time
from imp import reload
import Account



money_init = 1000000


PATH_STOCK_FEATURE = r'C:\pv\baostock data\stock feature20210128' #14
# PATH_STOCK_DATA = r'C:\pv\baostock data\dayK 20111221-20201208'
PATH_STOCK_DATA = r'C:\pv\baostock data\stock20210128' 
FN_HS300 = r'C:/pv/baostock data/index20210128/sh.000001.csv'
FN_FEATURE_HS300= r'C:/pv/baostock data/index20210128/sh.000001 feat.csv'



if not os.path.exists(PATH_STOCK_FEATURE):
    os.makedirs(PATH_STOCK_FEATURE)




def read_feat_file():
    df = pd.read_csv(STOCK_ALL)
    # df = pd.read_csv(r'cleaned.csv', dtype={'trade_date': int, 
        # 'ts_code': str, 'open': np.float16, 'high': np.float16,
        # 'low': np.float16, 'close': np.float16, 
        # 'pre_close': np.float16, 'turnover_rate': np.float16})
    return df

def get_company_info(market_item):
    df = pd.read_csv(FN_COMPANY, encoding='gbk')
    df = df[ df['market'].isin(market_item)]
    return df


def is_limit(ser, gap=0.01):
    # print(ser)
    if  (ser['code'][3:6]=='688') or ((ser['code'][3:6]=='300' and ser['date']>='2020-08-24')):
        limit_price = ser['preclose'] * 1.2
    else:
        limit_price = ser['preclose'] * 1.1
    if ser['isST'] == 1:
        limit_price = ser['preclose'] * 1.05
    if ser['close'] >= limit_price - gap:
        return 1
    else:
        return 0

def get_weekday(x):
    x = str(x)
    return datetime.datetime.fromtimestamp(time.mktime(time.strptime(x, "%Y%m%d"))).weekday()
            

def get_weekday_string(x):
    # x = str(x)
    return datetime.datetime.fromtimestamp(time.mktime(time.strptime(x, "%Y-%m-%d"))).weekday()

def get_fn(path):
    out = []
    for root, dirs,files in os.walk(path):
        for name in files:
            out.append(name)
    return out

def get_full_fn(path):
    out = []
    for root, dirs,files in os.walk(path):
        for name in files:
            out.append(os.path.join(root, name))
    return out


def gen_index_feature(in_fn= FN_HS300, out_fn= FN_FEATURE_HS300):#in_fn= FN_HS300, out_fn= FN_FEATURE_HS300)
    if not os.path.exists(out_fn):    
        print(in_fn)
        df = pd.read_csv(in_fn)
        df['szzs_close0@close-1']= df['close'] / df['preclose']
        df['szzs_close0@close-1_shift1']= df['szzs_close0@close-1'].shift(1)
        df['szzs_close0@close-1_shift2']= df['szzs_close0@close-1'].shift(2)
        df['szzs_close0@close-1_shift3']= df['szzs_close0@close-1'].shift(3)
        df['szzs_close0@close-1_shift4']= df['szzs_close0@close-1'].shift(4)
        df['szzs_close0@close-1_shift5']= df['szzs_close0@close-1'].shift(5)
        
        df['szzs_amount0@amount-1']= df['amount'] / df['amount'].shift(1)
        df['szzs_amount0@amount-1_shift1']= df['szzs_amount0@amount-1'].shift(1)
        df['szzs_amount0@amount-1_shift2']= df['szzs_amount0@amount-1'].shift(2)
        df['szzs_amount0@amount-1_shift3']= df['szzs_amount0@amount-1'].shift(3)
        df['szzs_amount0@amount-1_shift4']= df['szzs_amount0@amount-1'].shift(4)
        df['szzs_amount0@amount-1_shift5']= df['szzs_amount0@amount-1'].shift(5)
      
        df.to_csv(out_fn, index=None)
        
    else:
        df = pd.read_csv(out_fn)       
    return df





def concat_stock():
    lis = get_fn(PATH_STOCK_FEATURE)
    # lis = [r'%s/%s'%(PATH_STOCK_BAO, x) for x in lis ]
    out = map(read_stock_feature, lis)
    out2 = pd.concat(out)
    # out2 = out2.reset_index(drop=True)
    return out2






def get_stock_list_gen_feature(in_fn=PATH_STOCK_DATA):#, out_fn=PATH_STOCK_FEATURE
    fn_stock_dayK = get_fn(in_fn ) 
    fn_stock_feature = get_fn(PATH_STOCK_FEATURE )
    stock_lis = ["%s/%s"%(in_fn, x) for x in fn_stock_dayK if x not in fn_stock_feature] 
    print( len(stock_lis))
    return stock_lis


def run_gen_stock_feature(stock_lis):
    for i in stock_lis:
        # gen_stock_feature_one_stock(i)
        gen_stock_feature_one_stock_OLD_0121(i)
    # multiprocess(stock_lis, 4, gen_stock_feature_one_stock)
    return





def get_Xtype(ser):
    if (ser['close'] > ser['close_ma5']) and (ser['close_ma3'] > ser['close_ma5']) :
        return 1
    if (ser['close'] > ser['close_ma5']) and (ser['close_ma3'] <= ser['close_ma5']) :
        return 2
    if (ser['close'] <= ser['close_ma5']) and (ser['close_ma3']> ser['close_ma5']) :      
        return 3
    
    if (ser['close'] <= ser['close_ma5']) and (ser['close_ma3'] <= ser['close_ma5']):
        return 4



def get_Xtype_all2(ser):
    if ser['close'] > ser['close_ma5'] * 1.01 :
        return 1
    elif ser['close'] < ser['close_ma5']* 0.99:
        return  2
    else:
        return 3


       




def main_prepare_hs300():
    stock_lis= [ '%s/%s.csv'%( PATH_STOCK_DATA,x) for x in hs300_lis]
    run_gen_stock_feature(stock_lis)
    # run_gen_stock_feature(stock_lis)

def main_prepare():
    stock_lis= get_stock_list_gen_feature()
    run_gen_stock_feature(stock_lis)


def read_stock_feature(fn, xtype=1, col='Xtype_all2'):
    df = pd.read_csv(fn)

    feat_col = ['day','name','close_price','preclose', 'next_close','amount','open','isST'] + \
        [x for x in df.columns if ('@' in x) or ('Y' in x)]
    df = df[feat_col]
    # df = df[df[col] == xtype] #要修改#
    return  df

def main_get_feat():
    lis = get_full_fn(PATH_STOCK_FEATURE)
    # lis = [r'%s/%s'%(PATH_STOCK_BAO, x) for x in lis ]
    out0 = map(read_stock_feature, lis)
    out = pd.concat(out0, ignore_index=True)
    # out2 = out.reset_index(drop=True)
    return out


def gen_stock_feature_one_stock_OLD_0121(fn_stock, out_path=PATH_STOCK_FEATURE, 
                                 y_thres=0, exclude_day_for_new_lists=30):#特征带@
    # print(fn_stock)
    df = pd.read_csv(fn_stock)
    df = df[ df['tradestatus']==1]
    if len(df) > 0:
        df = df.rename(columns={'turn':'@turn'})
        df['next_close'] = df['close'].shift(-1)#未来信息，不能作为特征
        df = df.merge(FEAT_INDEX, on='date', how='left')
        # df['Y_ret>%s'%(y_thres) ] = ((df['next_close'] / df['close'] - 1) > y_thres)
        # 未来20个交易日最高涨幅超过12%，最大回撤不超过6%
        
        df['@zhangting'] = df.apply(is_limit, axis=1)
        df['@month'] = df['date'].apply(lambda x: int(str(x)[5:7]))
        df['@weekday'] = df['date'].apply(get_weekday_string)
        df['close_ma5'] = df['close'].rolling(5).mean()
        df['close_ma10'] = df['close'].rolling(10).mean()
        df['close_ma20'] = df['close'].rolling(20).mean()
        
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        
        # df['close0@close-1'] = df['close'] / df['close'].shift(1) -1
        # df['close0@close-1_shift1'] = df['close0@close-1'].shift(1)
        # df['close0@close-1_shift2'] = df['close0@close-1'].shift(2)
        # df['close0@close-1_shift3'] = df['close0@close-1'].shift(3)
        # df['close0@close-1_shift4'] = df['close0@close-1'].shift(4)
        # df['close0@close-1_shift5'] = df['close0@close-1'].shift(5)
        
        # df['volume0@volume-1'] = df['volume'] / df['volume'].shift(1) -1
        # df['volume0@volume-1_shift1'] = df['volume0@volume-1'].shift(1)
        # df['volume0@volume-1_shift2'] = df['volume0@volume-1'].shift(2)
        # df['volume0@volume-1_shift3'] = df['volume0@volume-1'].shift(3)
        # df['volume0@volume-1_shift4'] = df['volume0@volume-1'].shift(4)
        # df['volume0@volume-1_shift5'] = df['volume0@volume-1'].shift(5)
        
        df['close0@close_ma5'] = df['close']/df['close_ma5'] -1
        df['close0@close_ma10'] = df['close']/df['close_ma10'] -1
        df['close0@close_ma20'] = df['close']/df['close_ma20'] -1
        
        df['Xtype_close_gt_ma5'] = df['close0@close_ma5'] >0
        df['Xtype_close_gt_ma10'] = df['close0@close_ma10'] >0
        df['Xtype_close_gt_ma20'] = df['close0@close_ma20'] >0
        
        
        df['open@close_ma10'] = df['open']/df['close_ma10'] -1
        df['high@close_ma10'] = df['high']/df['close_ma10'] -1
        df['low@close_ma10'] = df['low']/df['close_ma10'] -1
        
        df['volume0@volume_ma5'] = df['volume']/df['volume_ma5'] -1
        df['volume0@volume_ma10'] = df['volume']/df['volume_ma10'] -1
        
        df['volume0@volume_ma5_shift1'] =df['volume0@volume_ma5'].shift(1)
        df['volume0@volume_ma5_shift2'] =df['volume0@volume_ma5'].shift(2)
        df['volume0@volume_ma5_shift3'] =df['volume0@volume_ma5'].shift(3)
        df['volume0@volume_ma5_shift4'] =df['volume0@volume_ma5'].shift(4)
        df['volume0@volume_ma5_shift5'] =df['volume0@volume_ma5'].shift(5)
        
        df['volume0@volume_ma10_shift1'] =df['volume0@volume_ma10'].shift(1)
        df['volume0@volume_ma10_shift2'] =df['volume0@volume_ma10'].shift(2)
        df['volume0@volume_ma10_shift3'] =df['volume0@volume_ma10'].shift(3)
        df['volume0@volume_ma10_shift4'] =df['volume0@volume_ma10'].shift(4)
        df['volume0@volume_ma10_shift5'] =df['volume0@volume_ma10'].shift(5)
        
        df['close0@close_ma5_shift1'] =df['close0@close_ma5'].shift(1)
        df['close0@close_ma5_shift2'] =df['close0@close_ma5'].shift(2)
        df['close0@close_ma5_shift3'] =df['close0@close_ma5'].shift(3)
        df['close0@close_ma5_shift4'] =df['close0@close_ma5'].shift(4)
        df['close0@close_ma5_shift5'] =df['close0@close_ma5'].shift(5)
                
        df['close0@close_ma10_shift1'] =df['close0@close_ma10'].shift(1)
        df['close0@close_ma10_shift2'] =df['close0@close_ma10'].shift(2)
        df['close0@close_ma10_shift3'] =df['close0@close_ma10'].shift(3)
        df['close0@close_ma10_shift4'] =df['close0@close_ma10'].shift(4)
        df['close0@close_ma10_shift5'] =df['close0@close_ma10'].shift(5)
        
        df['close0@close_ma20_shift1'] =df['close0@close_ma20'].shift(1)
        df['close0@close_ma20_shift2'] =df['close0@close_ma20'].shift(2)
        df['close0@close_ma20_shift3'] =df['close0@close_ma20'].shift(3)
        df['close0@close_ma20_shift4'] =df['close0@close_ma20'].shift(4)
        df['close0@close_ma20_shift5'] =df['close0@close_ma20'].shift(5)
        
        df['max_price_ma20'] = df['high'].iloc[::-1].rolling(20).max()#向下滚动。最大值
        df['min_price_ma20'] = df['low'].iloc[::-1].rolling(20).min()
        
        # df['Y_ret_20D'] = df['close'].shift(-20) / df['close'] -1
        df['Y_max_ret_20D'] = df['max_price_ma20']/ df['close'] -1
        df['Y_max_loss_20D'] = df['min_price_ma20']/ df['close'] -1
        # df['Y_max_loss_20D'] =df['min_price_ma20']/ df['close']-1
        df['Y_ret_5D'] = df['close'].shift(-5) / df['close'] -1 #20天收益率大于12%
        df['Y_ret_10D'] = df['close'].shift(-10) / df['close'] -1
        df['Y_ret_15D'] = df['close'].shift(-15) / df['close'] -1
        df['Y_ret_20D'] = df['close'].shift(-20) / df['close'] -1
        # df['Y_ret_20D_max'] = df['max_price_ma20']/ df['close'] -1
        # df['max_drawdown_20D'] = df['min_close_ma20'] / df['close'] > 0.94# 未来20天的最低价相对 当天收盘价的损失小于6%
        df['Y_ret_5D>0.03'] = df['Y_ret_5D'] > 0.03
        df['Y_ret_10D>0.06'] = df['Y_ret_10D'] > 0.06
        df['Y_ret_15D>0.09'] = df['Y_ret_15D'] > 0.09
        df['Y_ret_20D>0.12'] = df['Y_ret_20D'] > 0.12
        df['Y_MAX_ret_20D>0.12 MAX_dd<0.07'] = (df['Y_max_ret_20D'] > 0.12) & (df['Y_max_loss_20D']>-0.07 ) 
        
        
        # df['Y_max>0.12 max_dd<0.07']= (df['Y_max_ret_20D']>0.12) & (df['Y_max_loss_20D']>-0.07 )
        
        df = df.rename(columns={'code':'name', 'date':'day', 'close':'close_price'})        
        df = df.iloc[exclude_day_for_new_lists:, : ]
        
        out_fn = '%s/%s'%( out_path, fn_stock.split('/')[-1] )
        df.to_csv(out_fn, index = None)
        return 




def run_lgb( txt=''):
    
    num_leaves_lis = [31]
    learning_rate_lis = [  0.1]
    n_estimators_lis = [2000]
    colsample_bytree_lis = [1]
    subsample_lis = [ 0.9]
    subsample_freq_lis = [3]
    reg_alpha_lis = [0]
    reg_lambda_lis = [ 0.1]
    scale_pos_weight_lis = [10]
    
    hold_stock_num= 5
    prob_thres = 0.5
                                       


    for num_leaves in num_leaves_lis:
        for learning_rate in learning_rate_lis:
            for n_estimators in n_estimators_lis:
                for colsample_bytree in colsample_bytree_lis:
                    for  subsample in subsample_lis:
                        for  subsample_freq in subsample_freq_lis:
                            for reg_alpha in reg_alpha_lis:
                                for reg_lambda in reg_lambda_lis:
                                    for scale_pos_weight in scale_pos_weight_lis:
                                    
                                        fn = r'C:\pv\超短策略\res2\ leave=%s lr=%s n_est=%s subsample=%s subsample_freq= %s colsample_bytree=%s reg_alpha=%s reg_lamba=%s scale_pos_weight=%s %s'%\
                                            ( num_leaves, learning_rate, 
                                             n_estimators, subsample, subsample_freq, 
                                             colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight, txt )
        
                                        model = lgb.LGBMClassifier(random_state = 0, 
                                                                    learning_rate = learning_rate,
                                                                    num_leaves = num_leaves,
                                                                    n_estimators =n_estimators,

                                                                    colsample_bytree = colsample_bytree,
                                                                    subsample = subsample,
                                                                    subsample_freq = subsample_freq,
                                                                    reg_alpha= reg_alpha,
                                                                    reg_lambda = reg_lambda)
                                                                    # scale_pos_weight=scale_pos_weight)

        
                                        model.fit(trn, trn_label.astype('int').ravel())
                                    

                                    
                                        pred_prob = model.predict_proba(val, num_iteration=model.best_iteration_)[:,1]
                                        
                                        buy_idx = (feat_val['day'] >= val_date_min) & (feat_val['day'] <= val_date_max)
                                        buy_df0 = feat_val[buy_idx]
                                        buy_df0['prob'] = pred_prob
                                        buy_df = buy_df0.copy()
                                        buy_df['next_open'] = buy_df['next_close']
                                        buy_df['threshold'] =  pred_prob > prob_thres
                                        # buy_df['threshold'] = 1
                                        # buy_df['cyb'] = buy_df['name'].apply( lambda x: x[3:6]=='300' )
                                        # buy_df = buy_df[(buy_df['threshold'] == True) & \
                                        #                 (buy_df['@zhangting'] == False) &\
                                        #                 (buy_df['cyb'] ==True )]
                                    
                                        # buy_df = buy_df[(buy_df['threshold'] == True) & \
                                                        # (buy_df['@zhangting'] == False)]
                                        buy_df = buy_df[(buy_df['@zhangting'] == False)]
                                        
                                        eval_res = evaluate(fn, buy_df)
                   
    
 

def evaluate(fn, df):
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values(by=['prob'], ascending=False)
    window = [10, 20, 30, 50, 70, 100]
    Y_col_1  =['Y_ret_5D','Y_ret_10D','Y_ret_15D','Y_ret_20D']
    Y_col_2 = ['Y_ret_5D>0.03', 'Y_ret_10D>0.06', 'Y_ret_15D>0.09', 'Y_ret_20D>0.12', 'Y_MAX_ret_20D>0.12 MAX_dd<0.07']
    
    out = '\n%s\n'%fn
    for win in window:
        tmp = df.iloc[:win, ]  
        for col1 in Y_col_1:
            values1 = tmp[col1].values
            out += "prob最大的前%s个 %s 收益%.1f%%\n"%( win, col1, np.nanmean(values1 )*100  )
        for col2 in Y_col_2:
            values2 = tmp[ col2].dropna().values
            out += "prob最大的前%s个 %s 准确率%.1f%%\n"%( win, col2, sum(values2)/ len(values2) *100 )

    return out


def win_score_eval(preds, valid_df):    
    labels = valid_df #.get_label()
    preds = np.round(preds)
    tp = np.sum((preds==1)&(labels==1))
    pp = np.sum(preds==1)
    scores = tp/(pp+0.001) + 2.5*tp - pp
    return 'win', scores, True

def shuffle(df0):
    lis = df0.index.tolist()
    import random
    random.seed(0)
    random.shuffle(lis)
    return df0.iloc[lis, :].reset_index(drop=True)

if  1:
    
    FEAT_INDEX0 = gen_index_feature()
    feat_index_col = ['date'] + [x for x in FEAT_INDEX0.columns if '@' in x]
    FEAT_INDEX = FEAT_INDEX0[feat_index_col]
    # main_prepare_hs300()
    
    
    # main_prepare()
       
    
    # bb0= main_get_feat()

    Xtype_label = None
    
    bb1 = bb0
    bb_noST = bb1[ bb1['isST']!=1 ]
    bb_hs300 = bb_noST[ bb_noST['name'].isin(hs300_lis)]
    
    feat_trn = bb_hs300.dropna().reset_index(drop=True)
    # feat_trn  = shuffle(feat_trn )
    
    is_predict =True    #进行预测时，不排除空值，以实现到最新一天的预测
    if is_predict:
        feat_val = bb_hs300.reset_index(drop=True)        
    else:
        feat_val = bb_hs300.dropna().reset_index(drop=True)
    
    col_x1 = [ 'isST']
    col_x2 = [x for x in bb_hs300.columns if '@' in x \
                and 'szzs' not in x \
                # and 'month' not in x\
                # and 'weekday' not in x\
                and 'close0@close-1' not in x \
                and 'volume0@volume-1' not in x]
                # and 'ma20' not in x]    
    col_x_all  = col_x1 + col_x2
    col_y = ['Y_ret_20D>0.12']
    col_y_txt = col_y[0].replace('>','_gt_').replace('<','_st_')
    
    train_date_min = '2018-01-01' #多取一天20161230
    train_date_max = '2020-06-31'
    
    val_date_min =  '2020-07-01'
    val_date_max = '2021-12-31'
    
    idx = (feat_trn['day'] >= train_date_min) & (feat_trn['day'] <= train_date_max)
    trn = feat_trn[idx][col_x_all].values
    trn_label = feat_trn[idx][col_y].values
    
    idx = (feat_val['day'] >= val_date_min) & (feat_val['day'] <= val_date_max)
    val = feat_val[idx][col_x_all].values
    val_label = feat_val[idx][col_y].values
    val_day = feat_val[idx]['day'].unique()
    
    
    label_txt =''
    aa = run_lgb(txt=label_txt)
    print('done!')
 