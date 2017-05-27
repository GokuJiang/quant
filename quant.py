import datetime
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

start_date ='20161031'
end_date = '20170105'
factors = ['factor_ma', 'factor_bbi']

#初始化账户       
def initialize(account):      
	account.sample = '000300.SH'
	account.max_stocks = 5 # 最大持有股数
	cal = CAL()
	samples = get_index_stocks(account.sample, end_date)

	pass
	
#设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data): 
	currentDay = get_datetime()
	samples = get_index_stocks(account.sample, end_date)

	log.info(currentDay)
	quant = Quant(account,data,currentDay)
	samples = get_index_stocks(account.sample, end_date)
	endDate = get_datetime().strftime('%Y%m%d')
	cal = CAL()
	beginDate = cal.getDateByAdvance(endDate, -5)
	test = quant.timing(samples)

class Quant(object):
	def __init__(self, account,data,currentDay):
		self.account = account
		self.stockPool = []
		self.bought = []
		self.data = data
		self.currentDay = currentDay
		self.cal = CAL()
		
	def pickStocks(self):
		pass
	
	def timing(self, stocks = []):
		if not stocks:
			stocks = self.stockPool
		
		buylist = train_model(self.data,stocks, self.currentDay)
		log.info(buylist[:5].values)
		market = '000001.SH'
		marketMA1 = self.MA(market, 1)
		marketMA100 = self.MA(market, 100)
		
		for stock in buylist[:5]:
			self.account.security = stock

			cal = CAL()
			MACD = self.MACD(stock, get_datetime())['factor_macd'].values
			
			stockMA = (self.MA(stock, 20) + self.MA(stock, 30)) / 2
			
			log.info(MACD)
			
			if MACD < 0:
				# to do sell all
				order_target(self.account.security,0)
				self.bought = []
			elif marketMA1 > marketMA100:
				if stock not in self.bought and marketMA1 > marketMA100 * 1.03:
					self.positionControl(stock)
					self.bought.append(stock)
				elif stock in self.bought and marketMA1 < marketMA100 * 0.97:
					# to do sell stock
					order_target(self.account.security,0)
					self.bought.remove(stock)
					
	def MACD(self, stock, date):
		
		q = query(
			factor.macd
		).filter(
			factor.symbol == stock,
			factor.date == date.strftime("%Y-%m-%d")
		)
		return get_factors(q)
		
					
	def MA(self, stock, period):
		closePrices = self.getPrice(stock, period)['close']
		closePrices = closePrices[closePrices.columns[0]].values
		return np.mean([float(price) for price in closePrices])
	
	def getPrice(self, stocks, period, fields = []):
		if not isinstance(stocks, list):
			stocks = [stocks,]
		
		endDate = get_datetime()
		cal = CAL()
		beginDate = cal.getDateByAdvance(endDate, -int(period - 1))
		
		return get_price(
			stocks,
			start_date = beginDate,
			end_date = endDate,
			fre_step = '1d',
			fields = list(set(fields + ['close'])),
			skip_paused = True,
			is_panel = 1,
		)
					
	def positionControl(self,stock):
		#获取每只股票的进仓量
		positionQuantity = self.stockMaxQuantity(stock)
		#产生下单信号
		order_to(stock, positionQuantity)
		log.info(self.account.position)
				
	
	def stockMaxQuantity(self,stock):
		#最大风险随时
		max_lose = self.account.cash * self.risk
		beginDate = self.cal.getDateByAdvance(self.currentDate,-10)
		log.info(beginDate)
		#获取今日最高价
		highestPrice = self.getPrice(stock,1,['high'])
		#10十日内收盘价
		historyTrade10 = self.self.getPrice(stock,19,['close'])['close'].sort_index(by='close')
		# #10日内最低价格
		min10 = historyTrade10['closePrice'].values[0]
		# #止损点
		lowerLimit = min10 * (1 - 0.08)
		# #计算交易量
		volume = max_lose / (highestPrice - lowerLimit)
		return volume
	
	def stopProfile(self):
		#10十日内收盘价
		historyTrade10 = self.__getTrades([stock],days=10).sort_index(by='closePrice')['closePrice'].values  
		#10日内最高价
		max10 = historyTrade10[-1:]
		#10日内最低价
		min10 = historyTrade10[0]

		#获取今日最高价
		highestPrice = self.getPrice(stock,1,['close'])['close'].values[0]

		#止盈
		upperLimit = max10 * (1+0.08) 
		#止损
		lowerLimit = min10 * (1-0.08) 
		#买卖信号
		dealSingal = (highestPrice > upperLimit) or (highestPrice < lowerLimit)
		return dealSingal
	
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
		return datetime.datetime.strptime(date['date'].values[0], "%Y%m%d")

	def getDateByAdvance(self,date,step):
		# log.info(date)
		queryCondition = 'date>= "' + self.getDate(date).strftime('%Y%m%d') + '"'

		index = int(self.tradeDate.query(queryCondition).index[0])+step
		
		
		if index >=0 and index < self.tradeDate.shape[0]:
			return datetime.datetime.strptime(self.tradeDate['date'].values[index], "%Y%m%d")
		return None
		
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

	return buylist
