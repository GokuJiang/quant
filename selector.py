'''
params:
    data: 回测的data对象
    samples: 股票池
    currentData: 回测当前日期，类型为datatime
'''

def train_model(data,samples, currentDay):
    cal = CAL()
    lastDay = cal.getDateByAdvance(currentDay,-1).strftime('%Y-%m-%d')
    currentDay = currentDay.strftime("%Y-%m-%d")
    log.info("获取因子...")
    
    #获取因子数据
    factor_df = pd.DataFrame(columns=['stockID', 'factor_date'] + factors)
    for stock in samples:
        facQuery = query(factor.date,factor.ma,factor.bbi).filter(factor.symbol == stock,factor.date.in_([lastDay]))
        fac = get_factors(facQuery) 
        fac.insert(0,'stockID',stock)
        factor_df = pd.concat([factor_df,fac], ignore_index=True)

    factor_df =factor_df.sort_index(by=['factor_date','stockID'])   

    log.info("获取每日收盘价...")
    #获取每日收盘价
    samplePrice = data.history(samples, 'close', 2, '1d', skip_paused = False, fq = None, is_panel = 1)
    closePrice = samplePrice['close']
    closeIndex = closePrice.index
    
    #计算涨幅
    increase = {}
    for i in range(1,len(closePrice)):
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

    log.info("构建训练集...")
    price = increase_df[lastDay].values
    factor_df.insert(1,'increaseRatio',price)
    traindf = factor_df
    traindf=traindf.dropna()
    traindf=traindf.sort_index(by='increaseRatio')

    # target二值化，大于平均数的设置1，小于平均数的设置0
    target=list(traindf['increaseRatio'].apply(lambda x:1 if x>np.mean(list(traindf['increaseRatio'])) else 0))
    
    #生成数据集
    del traindf['increaseRatio']
    del traindf['factor_date']
    
    log.info("构建测试集...")
    #构建测试集
    X_test = factor_df
    X_test = X_test.dropna()

    log.info("开始拟合...")
    # 创建并且训练一个支持向量机分类模型,根据上一个交易日的因子预测涨跌,返回预测涨幅最大的前10支股票
    clf = SVC(probability=True)  
    #clf是支持向量机模型，开启概率估计
    clf.fit(traindf.iloc[:,1:].values, np.array(target))
    #fit就是训练模型，让模型适应数据
    log.info("开始预测...")

    #预测值，predict_proba会输出两个类型的概率，输出的类型是[[0类的概率，1类的概率],[0类的概率，1类的概率]，。。。]，所以选择X[1]是选择判别为1类（涨）的概率，下面要排序
    predicted_results = [x[1] for index, x in enumerate(clf.predict_proba(X_test.iloc[:,-2:].values))]
    
    X_test.insert(0,'predict',predicted_results)
    
    # #按照判断涨的概率排序
    X_test=X_test.sort_index(by='predict')
    # #选概率大于0.5的股
    test=X_test[X_test['predict']>=0.5]
    buylist=test['stockID'][:20]#选概率最大的20只股票
    log.info(buylist)

    return buylist
