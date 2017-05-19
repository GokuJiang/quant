# coding: utf-8
from datetime import datetime
import numpy as np
import pandas as pd
import CAL
from sklearn import SVC

start = '2013-01-01'                       # 回测起始时间
end = '2017-01-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')           # 证券池，支持股票和基金
capital_base = 100000                      # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
tradeCache = pd.DataFrame()

trainperiod='-3M'#训练周期3个月
factor=['VOL240','NetProfitGrowRate','FEARNG','PE']


def initialize(account):                   # 初始化虚拟账户状态
	pass

def handle_data(account):                  # 每个交易日的买入卖出指令
	myQuant = MyQuant(account, account.current_date)
	print myQuant
	
	buylist = myQuant.queryStocks()
	print buylist
	
	myQuant.timing(buylist)

	
	
class MyQuant(object):
	
	def __init__(self, account, date = datetime.now(), MAthreshold = 0, stocks = []):
		self.__account = account
		self.__calendar = CAL.Calendar('China.SSE')
		self.__date = date
		self.__MAthreshold = MAthreshold
		self.__stockPools = stocks
		self.__buyList = []
	
	def stockChoosing(self, param):
		pass
	
	def timing(self, stocks):
		self.__stockPools = stocks
		
		for stock in self.__stockPools:
			longMA = self.__MA(20, stock)
			shortMA = self.__MA(1, stock)
			if shortMA - longMA > self.__MAthreshold:
				order_pct_to(stock, 1. / len(self.__stockPools))
			else:
				order_to(stock, 0)
	
	def __MA(self, days, secID):
		ma = {}
		
		if isinstance(secID, basestring):
			secID = [secID]
		
		for stock in secID:
			trade = self.__getTrades.advanceDate(
				[stock], 
				self.__calendar.advanceDate(self.__date, '-' + str(days) + 'B'), 
				self.__calendar.advanceDate(self.__date, '-1B'))
			prices = trade.loc[:]['closePrice'].values
			ma[stock] = np.mean([float(price) for price in prices])
		
		return ma if len(secID) > 1 else ma[ma.keys()[0]]
		
	def __getTrades(self, secID, beginDate, endDate):
		global tradeCache
		
		sql = '(secID == "' + '" | secID == "'.join(secID) + '") & tradeDate >= "' + beginDate.strftime('%Y-%m-%d') + '" & tradeDate <= "' + endDate.strftime('%Y-%m-%d') + '"'
		
		if 'secID' not in tradeCache.columns or tradeCache.query(sql).empty:
			tradeCache = DataAPI.MktEqudGet(
				secID = self.__stockPools, 
				beginDate = beginDate, 
				endDate = self.__calendar.advanceDate(endDate, '+1M'),
				field = ['secID', 'tradeDate', 'closePrice'], 
				pandas = '1')
			
			print sql + ' missing'
			
		return tradeCache.query(sql)
	
	def queryStocks(self):
		
		preday=self.__calendar.advanceDate(self.__account.current_date,trainperiod)
		yesterday=self.__calendar.advanceDate(self.__account.current_date,'-1B')
		#获得训练第一天和上一个收盘日
		##############################################

		#各列分别是secid，3个月前的因子值，3个月内的强弱（1代表比大盘强，0代表比大盘弱）
		fac=DataAPI.MktStockFactorsOneDayGet(
			tradeDate=preday,
			secID=self.__account.universe,
			field=['secID']+factor,
			pandas="1")
		#创建价格df:price
		price1=DataAPI.MktEqudAdjGet(
			secID=self.__account.universe,
			tradeDate=preday,
			field=u"secID,closePrice",pandas="1")
		price2=DataAPI.MktEqudAdjGet(
			secID=self.__account.universe,
			tradeDate=yesterday,
			field=u"secID,closePrice",pandas="1")

		price2['closePrice2']=price2['closePrice']
		del price2['closePrice']
		price=pd.merge(price1,price2)

		tmp1=[]
		tmp=(price['closePrice2']-price['closePrice']) / price['closePrice'] * 100

		for i in tmp:
			tmp1.append(int(i))

		price['zhangdie']=tmp1
		del price['closePrice']
		del price['closePrice2']

		traindf=pd.merge(fac,price)
		traindf.set_index(traindf.secID)
		del traindf['secID']

		traindf=traindf.dropna()
		traindf=traindf.sort(columns='zhangdie')
		traindf.reset_index(drop=True,inplace=True)

		traindf=traindf.iloc[:len(traindf['zhangdie'])/10*3,:].append(traindf.iloc[len(traindf['zhangdie'])/10*7:,:])


		##################
		#target二值化，大于平均数的设置1，小于平均数的设置0
		target=list(traindf['zhangdie'].apply(lambda x:1 if x>np.mean(list(traindf['zhangdie'])) else 0))

		train=traindf.iloc[:,0:-1].values
		#构建train列表和target列表完毕

		###########################
		#建立test集
		test1 = DataAPI.MktStockFactorsOneDayGet(
			tradeDate=yesterday,
			secID=self.__account.universe,
			field=['secID']+factor,
			pandas="1")
		test1=test1.dropna()
		test=test1.iloc[:,1:].values

		#################### core #########################
		# 创建并且训练一个支持向量机分类模型,根据上一个交易日的因子预测涨跌,返回预测涨幅最大的前10支股票
		clf = SVC(probability=True)  
		#clf是支持向量机模型，开启概率估计
		clf.fit(train, target)
		#fit就是训练模型，让模型适应数据

		#预测值，predict_proba会输出两个类型的概率，输出的类型是[[0类的概率，1类的概率],[0类的概率，1类的概率]，。。。]，所以选择X[1]是选择判别为1类（涨）的概率，下面要排序
		predicted_results = [x[1] for index, x in enumerate(clf.predict_proba(test))]
		test1['predict']=predicted_results    

		#按照判断涨的概率排序
		test1=test1.sort(columns='predict',ascending=False)
		#选概率大于0.5的股
		test1=test1[test1['predict']>=0.5]
		buylist=test1['secID'][:20]#选概率最大的50只股票
		return buylist

