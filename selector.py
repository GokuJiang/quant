import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split



start_date ='20161031'
end_date = '20170105'
factor_date = '20161031'#为了获取10月末的因子数据而设定的日期参数
index = '000300.SH'#沪深300指数
# stock_sample = get_index_stocks(index,start_date)#选取起始时间作为参数来构建沪深300指数成分股股票池
# trade_days = get_trade_days(start_date,end_date)#获取交易日时段

trade_days = get_all_trade_days()
factors = ['factor_ma', 'factor_bbi']

class CAL(object):
    def __init__(self):
        #获取所交易日
        days = get_all_trade_days()
        #日期格式化
        days = [day.strftime('%Y%m%d') for day in days]
        #创建DateFarme
        self.tradeDate = pd.DataFrame(days,columns=['date'])

    def getDate(self,date):
        #在tradeDate中查询这一天是否存在
        queryCondition = ""
        if isinstance(date, str):
            queryCondition = 'date>="' + date + '"'
        else:            
            queryCondition = 'date>="' + date.strftime('%Y%m%d') + '"'

        date = self.tradeDate.query(queryCondition)
        log.info(date)
        return date['date'].values[0]

    def getDateByAdvance(self,date,step):
        queryCondition = 'date>= "' + self.getDate(date) + '"'
        log.info(queryCondition)
        
        index = int(self.tradeDate.query(queryCondition).index[0])+step
        
        
        if index >=0 and index < self.tradeDate.shape[0]:
            return self.tradeDate['date'].values[index]
        
        return None

def train_model(account,samples, beginDate, endDate):
    trade_days = get_trade_days(beginDate, endDate).strftime('%Y-%m-%d')

    log.info("获取因子...")
    #获取因子数据
    factor_df = pd.DataFrame(columns=['stockID', 'factor_date'] + factors)
    for stock in samples:
        facQuery = query(factor.date,factor.ma,factor.bbi).filter(factor.symbol == stock,factor.date.in_(trade_days))
        fac = get_factors(facQuery) 
        fac.insert(0,'stockID',stock)
        factor_df = pd.concat([factor_df,fac], ignore_index=True)
        # factorDict[stock] = fac
    
    factor_df =factor_df.sort_index(by=['factor_date','stockID'])   
    
    log.info("获取每日收盘价...")
    #获取每日收盘价
    samplePrice = get_price(samples, beginDate, endDate, '1d', ['close'], True, None,is_panel=1)
    closePrice = samplePrice['close']
    closeIndex = closePrice.index
    
    #计算涨幅
    increase = {}
    for i in range(0,len(closePrice)):
        #获取索引
        index = closeIndex[i]
        lastIndex = closeIndex[i-1]
        #获取当天与上一天的收盘价
        currentPrice = closePrice.loc[index]
        lastPrice = closePrice.loc[lastIndex]
        #计算涨幅
        increaseRatio = (currentPrice - lastPrice) / lastPrice * 100
        increase[index] = increaseRatio

    increase_df=pd.DataFrame(increase)  
    
    log.info("生成数据集...")

    train = pd.DataFrame(columns=['increaseRatio']+factors)
    targets = []
    for day in trade_days:
        fac = factor_df.query('factor_date == "' + day + '"')
        price = increase_df[day].values
        
        fac.insert(1,'increaseRatio',price)
        
        #生成数据集
        traindf = fac
        traindf=traindf.dropna()
        traindf=traindf.sort_index(by='increaseRatio')
        train = pd.concat([train,traindf], ignore_index=True)
        
        # #target二值化，大于平均数的设置1，小于平均数的设置0
        target=list(traindf['increaseRatio'].apply(lambda x:1 if x>np.mean(list(traindf['increaseRatio'])) else 0))
        for y in target:
            targets.append(y)
        
    del train['increaseRatio']
    del train['factor_date']
    
    targetSet = np.array(targets)
    
    log.info("构建训练和集合测试集...")

    #构建训练和集合测试集
    countOfTrain = int(len(train['stockID']) /10 * 7)

    X_train = train.iloc[0:countOfTrain,:]
    X_test = train.iloc[countOfTrain:,:]
    Y_train = targetSet[:countOfTrain]
    Y_test = targetSet[countOfTrain:]
    log.info(X_train.iloc[:,:-1])
    log.info("开始拟合...")

    # 创建并且训练一个支持向量机分类模型,根据上一个交易日的因子预测涨跌,返回预测涨幅最大的前10支股票
    clf = SVC(probability=True)  
    #clf是支持向量机模型，开启概率估计
    clf.fit(X_train.iloc[:,:-1].values, Y_train)
    #fit就是训练模型，让模型适应数据
    log.info("开始预测...")

    #预测值，predict_proba会输出两个类型的概率，输出的类型是[[0类的概率，1类的概率],[0类的概率，1类的概率]，。。。]，所以选择X[1]是选择判别为1类（涨）的概率，下面要排序
    predicted_results = [x[1] for index, x in enumerate(clf.predict_proba(X_train.iloc[:,:-1].values))]
    
    log.info(predicted_results)

    X_test['predict']=predicted_results    
    
    # #按照判断涨的概率排序
    X_test=X_test.sort(columns='predict',ascending=False)
    # #选概率大于0.5的股
    # test1=test1[test1['predict']>=0.5]
    # buylist=test1['secID'][:20]#选概率最大的50只股票

#初始化账户       
def initialize(account): 
    account.sample = '000300.SH'
    account.max_stocks = 5 # 最大持有股数
    begin_date = '2013-12-01'
    end_date = '2015-12-31'
    
    samples = get_index_stocks(account.sample, end_date)
    
    train_model(account,samples, begin_date, end_date)

    pass
    

#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):  
    pass

  
