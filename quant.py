# coding: utf-8
import CAL
import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from lib.detect_peaks import detect_peaks

start = '2013-01-01'                       # 回测起始时间
end = '2017-01-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')           # 证券池，支持股票和基金
capital_base = 100000                      # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟

trainperiod='-3M'#训练周期3个月
factor=['ROE','ROA','EPS','EBITDA','PR','L/A','FixAssetRatio','CMV','PE','EV/EBITDA','PS','DividendYieldRatio','B/M']#选3个
cal = CAL.Calendar('China.SSE')#创建日历

def initialize(account):                   # 初始化虚拟账户状态
	pass

def handle_data(account):                  # 每个交易日的买入卖出指令
	
	myQuant = MyQuant(account, account.current_date)
	myQuant.timing()
	
class MyQuant(object):
	
	def __init__(self, account, date = datetime.date.today(), MAthreshold = 0, stocks = [],risk=0.02):
		self.__account = account
		self.__calendar = CAL.Calendar('China.SSE')
		self.__date = date
		self.__tradeCache = pd.DataFrame()
		self.__MAthreshold = MAthreshold
		self.__stockPool = stocks if stocks else self.__account.universe
		self.__boughtPool = []
		self.__IdxMA()
		self.__risk = risk

	def stockChoosing(self):
		preday=self.__calendar.advanceDate(self.__account.current_date,trainperiod).strftime('%Y%m%d')
		yesterday=self.__calendar.advanceDate(self.__account.current_date,'-1B').strftime('%Y%m%d')
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
		self.__stockPool = buylist.to_dict().values()
		
		return self.__stockPool
	
	def timing(self, stocks = []):
		self.__stockPool = stocks if stocks else self.stockChoosing()

		
		for stock in self.__stockPool:
			MA100 = self.__MA(stock, 100)
			MA20 = self.__MA(stock, 20)
			MA30 = self.__MA(stock, 30)
			MA = (MA20 + MA30) / 2

			if self.__IdxMA(1) > self.__IdxMA():
				if stock not in self.__boughtPool and self.__MA(stock, days = 1) > (MA * 1.03):
					# order_pct_to(stock, 1. / len(self.__stockPool))
					self.__positionControl()
					self.__boughtPool.append(stock)
				elif stock in self.__boughtPool and self.__MA(stock, days = 1) < (MA * 0.97):
					order_to(stock, 0)
					self.__boughtPool.remove(stock)
	
	#仓控
	def __positionControl(self):
		#遍历股票池中每只股票
		for stock in self.__stockPool:
			if stock not in self.__boughtPool:
				#获取每只股票的进仓量
				positionQuantity = self.__stockMaxQuantity(stock)
				#产生下单信号
				# order_to(stock, positionQuantity)
		
	#每只股票最大进仓量
	def __stockMaxQuantity(self,stock):
		#最大风险随时
		max_lose = self.__account.cash * self.__risk
		beginDate = self.__calendar.advanceDate(self.__date, '-10B')
		#获取今日最高价
		highestPrice = self.__getTrades(
			[stock],
			field = ['highestPrice'],
			beginDate = self.__date, endDate = self.__date)['closePrice'].values[0]
		#10十日内收盘价
		historyTrade10 = self.__getTrades([stock],days=10).sort_index(by='closePrice')
		# #10日内最低价格
		min10 = historyTrade10['closePrice'].values[0]
		# #止损点
		lowerLimit = min10 * (1 - 0.08)
		# #计算交易量
		volume = max_lose / (highestPrice - lowerLimit)
		return volume
	
	def __stopProfile(self):
		#10十日内收盘价
		historyTrade10 = self.__getTrades([stock],days=10).sort_index(by='closePrice')['closePrice'].values  
		#10日内最高价
		max10 = historyTrade10[-1:]
		#10日内最低价
		min10 = historyTrade10[0]

		#获取今日最高价
		highestPrice = self.__getTrades(
			[stock],
			field = ['highestPrice'],
			beginDate = self.__date, endDate = self.__date)['closePrice'].values[0]
		#止盈
		upperLimit = max10 * (1+0.08) 
		#止损
		lowerLimit = min10 * (1-0.08) 
		#买卖信号
		dealSingal = (highestPrice > upperLimit) or (highestPrice < lowerLimit)
		return dealSingal
		
	def __IdxMA(self, days = 100, ticker = u'000001'):
		beginDate = self.__calendar.advanceDate(self.__date, '-' + str(days) + 'B').strftime('%Y%m%d')
		endDate = self.__calendar.advanceDate(self.__date, '-1B').strftime('%Y%m%d')
		idxs = DataAPI.MktIdxFactorDateRangeGet(ticker = ticker, beginDate = beginDate, endDate = endDate, field = ['tradeDate', 'Close'], pandas = '1').loc[:]['Close'].values
		ma = np.mean([float(idx) for idx in idxs])
		return ma
	
	def __MA(self, secID, days):
		ma = {}
		
		if isinstance(secID, basestring):
			secID = [secID]
		
		for stock in secID:
			trade = self.__getTrades([stock], days = days)
			prices = trade.loc[:]['closePrice'].values
			ma[stock] = np.mean([float(price) for price in prices])
		
		return ma if len(secID) > 1 else ma[ma.keys()[0]]
	
	def __getTrades(self, secID, field = [], **kw):
		if isinstance(secID, basestring):
			secID = [secID]

		if 'days' in kw.keys():
			beginDate = self.__calendar.advanceDate(self.__date, '-' + str(kw['days']) + 'B')
			endDate = self.__calendar.advanceDate(self.__date, '-1B')
		elif 'beginDate' in kw.keys() and 'endDate' in kw.keys():
			beginDate = kw['beginDate']
			endDate = kw['endDate']
		else:
			raise ValueError('__getTrades takes days or (beginDate and endDate) as parameters')

		sql = '(secID == "' + '" | secID == "'.join(secID) + '") & tradeDate >= "' + beginDate.strftime('%Y-%m-%d') + '" & tradeDate <= "' + endDate.strftime('%Y-%m-%d') + '"'

		# if 'secID' not in self.__tradeCache.columns or self.__tradeCache.query(sql).empty:
		#     fields = list(set(field + ['secID', 'tradeDate', 'closePrice']))
		#     print fields
		#     self.__tradeCache = DataAPI.MktEqudGet(
		#         secID = self.__stockPool,
		#         beginDate = beginDate,
		#         endDate = self.__calendar.advanceDate(endDate, '+100B'),
		#         field = list(set(field + ['secID', 'tradeDate', 'closePrice'])),
		#         pandas = '1'
		#     )
		#     log.info(sql + ' missing')
		#fields = list(set(field + ['secID', 'tradeDate', 'closePrice']))
		self.__tradeCache = DataAPI.MktEqudGet(
			secID = self.__stockPool,
			beginDate = beginDate,
			endDate = self.__calendar.advanceDate(endDate, '+100B'),
			field = list(set(field + ['secID', 'tradeDate', 'closePrice'])),
			pandas = '1'
		)
		return self.__tradeCache.query(sql)
