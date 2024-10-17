#!/usr/bin/env python
# coding: utf-8

# # Project

# ## Introduction to the Project
# The S&P 500 (Standard & Poor's 500) is a stock market index that tracks the performance of 500 of the largest publicly traded companies in the United States. It is widely regarded as one of the best representations of the U.S. stock market and economy. Over the long term, the S&P 500 has shown consistent growth, making it a key focus for long-term investors. However, it can also experience significant volatility in the short term.
# 
# In this project, we will make our first attempt to build a momentum-based strategy to trade the S&P 500 index. At the end of the project, you will have built a program that you can later expand and customise to suit your needs. We will use the python packages numpy, scipy and sqlite3, among others, in this project.
# 
# Tips: Review the code snippets that we went through during the course. Reorganize them and you are half-way done! Try experimenting with different configurations of the confidence interval, the lookback window and the forecast horizon. Be brave and experiment with different ways of deciding the position size. You may be surprised by your talent!

# Re-organize your code from the exercises into a Python program that 
# 1. read prices from a database
# 2. calibrate a GBM model on each day when new prices are received.
# 3. forecast the price in e.g. 10 days and
#    1. calculate the confidence interval of the forecast
#    2. calculate the expected shortfall from the price forecast
# 4. code your trading signals using the price forecast and the expected shortfall.
# 5. store your postions into the *positions* table after each trade.
# 6. produce a 1-year backtest record from 2020-06-01 to 2021-05-31.
# 
# **Hint**
# 
# 1. Collect all the code related to the GBM model into a class

# In[1]:


import csv
import sqlite3
from contextlib import closing
from datetime import datetime

import numpy as np
from scipy.stats import norm

conn = sqlite3.connect("SP500.db")
cs = conn.cursor()


# In[4]:


class GBM:
    def __init__(self):
        # TODO: Replace the following with your code
        self.mu = np.nan
        self.sigma = np.nan
        self.rng = np.random.default_rng()
        
    def simulate(self, N, K, Dt, S0):
        sqrt_Dt = np.sqrt(Dt)
        traj = np.full((N+1, K), np.nan)
        drift = (self.mu - self.sigma**2/2) * np.linspace(1, N, N) * Dt
        for i in range(K):
            W = sqrt_Dt * np.cumsum(norm.rvs(size=N))
            traj[1:, i] = S0 * np.exp(drift + self.sigma * W)
            traj[0, i] = S0
        return traj

    def calibrate(self, trajectory, Dt):
        increments = np.diff(np.log(trajectory))
        moments = [0, 0]
        n_iter = 10
        for iter in range(n_iter):
            X = self.rng.choice(increments, size=len(increments)//2)
            moments[0] += np.mean(X)/n_iter
            moments[1] += np.mean(X**2)/n_iter
        std = np.sqrt(moments[1] - moments[0]**2)
        self.sigma = std/np.sqrt(Dt)
        self.mu = moments[0] / Dt + self.sigma**2/2
        # TODO: Your code goes here
        pass
        
    def forecast(self, latest, T, confidence):
        m = (self.mu - self.sigma**2/2) * T
        s = self.sigma * np.sqrt(T)
        Q = norm.ppf([(1-confidence)/2, (1+confidence)/2], loc=m, scale=s)
        return {
            'confidence': confidence,
            'expected': latest * np.exp(self.mu * T),
            'interval': latest * np.exp(Q)
        }
        # TODO: Your code goes here
        pass
        
    def expected_shortfall(self, T, confidence):
        m = (self.mu - self.sigma**2/2) * T
        s = self.sigma * np.sqrt(T)
        z = norm.ppf(confidence)
        ES = (self.mu - self.sigma**2/2) * T + self.sigma * norm.pdf(z) / (1 - confidence) * s
        return ES


# In[5]:


# test your code here
model = GBM()
model.mu = 0.3
model.sigma = 0.2
simulated = model.simulate(500, 1, 1/250, 100)
simulated = simulated[:, 0]

model2 = GBM()
model2.calibrate(simulated, 1/250)

print(F"Calibrated: mu = {model2.mu}, sigma = {model2.sigma}")



# 2. Write a function that prepares the database for trading, i.e.
#    1. load the historical prices into the *prices* table
#    2. create the *positions* table
#    3. initialize the *positions* table with the your initial cash reserve. The initial *time_of_trade* can be any date before the earliest possible trading date.
# 
#     Call this function *prepare*.

# In[6]:


def prepare():
    # TODO: Your code goes here
    cs.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        theday TEXT PRIMARY KEY,
        price REAL
    );
    """)
    with closing(open('SP500.csv')) as datafile:
        reader = csv.DictReader(datafile, fieldnames=["date", "price"], delimiter='\t')
        for row in reader:
            cs.execute("""
            INSERT OR IGNORE INTO prices (theday, price) VALUES (?, ?);
            """, (row['date'], float(row['price'])))
    cs.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        time_of_trade TEXT,
        instrument TEXT,
        quantity REAL,
        cash REAL,
        PRIMARY KEY (time_of_trade, instrument)
    );
    """)
    cs.execute("""
    INSERT OR IGNORE INTO positions (time_of_trade, instrument, quantity, cash)
    VALUES ('1666-01-01', 'SP500', 0, 1000000);
    """)
    conn.commit()


# In[7]:


# check whether you have loaded the prices correctly
prepare()
latest_prices = cs.execute("select * from prices order by theday desc limit 10")
for item in latest_prices:
    print(item)


# 3. Write a function that determines the trade size, i.e. how many units of the instrument you would like to own when the date is *which_day* and the price forecast of the instrument is *forecast* and the expected shortfall from the same forecast is *ES*.

# In[8]:


def position_size(which_day, forecast, ES):
    # TODO: Your code goes here
    # Fetch the latest position and cash before which_day
    cs.execute("""
        SELECT quantity, cash FROM positions
        WHERE instrument = 'SP500'
        AND time_of_trade < ?
        ORDER BY time_of_trade DESC
        LIMIT 1;
    """, (which_day,))
    result = cs.fetchone()
    
    if result:
        current_qty, current_cash = result
    else:
        # If no prior position exists, initialize with 0 quantity and default cash
        current_qty, current_cash = 0, 1000000  # Adjust initial cash as needed

    # Fetch the latest price up to which_day
    cs.execute("""
        SELECT price FROM prices
        WHERE theday <= ?
        ORDER BY theday DESC
        LIMIT 1;
    """, (which_day,))
    price_result = cs.fetchone()
    
    if price_result:
        current_price = price_result[0]
    else:
        # Handle cases where price data is unavailable
        print(f"No price data available for {which_day}.")
        return current_qty  # Hold current position

    # Calculate expected return
    expected_return = (forecast['expected'] / current_price) - 1

    # Define a risk-adjusted factor based on ES
    # Lower ES (less risk) allows for larger positions
    # Higher ES (more risk) reduces the position size
    risk_tolerance = 1  # You can adjust this parameter based on your risk appetite
    if ES != 0:
        position_factor = (expected_return * risk_tolerance) / ES
    else:
        position_factor = 0

    # Normalize position_factor to be within [-1, 1] to prevent overleveraging
    position_factor = max(min(position_factor, 1), -1)

    # Determine desired position size
    desired_qty = int(position_factor * (current_cash / current_price))

    # Optional: Set boundaries for position size to prevent extreme positions
    max_position = 10000  # Maximum number of shares you want to hold
    min_position = -10000  # Maximum number of shares you want to short
    desired_qty = max(min(desired_qty, max_position), min_position)

    return desired_qty


# 4. Write a function that, for a given date, calibrates a GBM model to the data prior to that date and that forecasts the price in 10 days. Call this function *analyse*.

# In[9]:


def analyse(which_day):
    # TODO: Your code goes here   
    # Fetch the last 120 days of price data up to which_day
    cs.execute("""
        SELECT price FROM prices
        WHERE theday <= ?
        ORDER BY theday DESC
        LIMIT 120;
    """, (which_day,))
    price_data = cs.fetchall()
    
    if len(price_data) < 2:
        print(f"Not enough data to analyze for {which_day}.")
        return 0  # Hold position if insufficient data

    # Convert fetched data to a NumPy array and reverse it to chronological order
    P = np.flipud(np.array([price[0] for price in price_data]))

    # Initialize and calibrate the GBM model
    model = GBM()
    Dt = 1.0 / 252  # Daily time step
    model.calibrate(P, Dt)

    # Define forecast parameters
    confidence = 0.95  # 95% confidence interval
    forecast_horizon = 10  # Forecast 10 days into the future
    T = forecast_horizon * Dt

    # Generate forecast using the calibrated model
    forecast = model.forecast(P[-1], T, confidence)

    # Calculate Expected Shortfall (ES)
    ES = model.expected_shortfall(T, confidence)

    # Determine the desired position size based on forecast and ES
    desired_qty = position_size(which_day, forecast, ES)

    return desired_qty

# Prepare the database
prepare()


# In[10]:


# Test the analyse function
test_dates = ['2021-05-09', '2021-05-14']
positions = [np.nan, np.nan]
for i in range(2):
    positions[i] = analyse(test_dates[i])
    print(F"{positions[i]} shares advised on {test_dates[i]}.")


# 5. The main loop of the program: Loop over the dates in the backtest period and use the *analyse* function to decide what to do on each day. Call this function *main*.

# In[11]:


def main(begin_on):
    cs.execute(F"select theday from prices where theday >= '{begin_on}';")
    days = [d[0] for d in cs.fetchall()]
    asset = {
        'old': np.nan,
        'new': np.nan
    };
    cash = {
        'old': np.nan,
        'new': np.nan
    };
    cs.execute("delete from positions where time_of_trade > '2020-01-01';");
    for d in days:
        asset['new'] = analyse(d)
        cs.execute(F"""
        select quantity, cash from positions
        where time_of_trade < '{d}'
        order by time_of_trade desc
        limit 1;
        """);
        asset['old'], cash['old'] = cs.fetchall()[0];
        cs.execute(F"""
        select price from prices
        where theday <= '{d}'
        order by theday desc
        limit 1;
        """);
        latest = cs.fetchall()[0][0]
        trade_size = round(asset['new']) - round(asset['old']);
        if trade_size != 0:
            cash['new'] = cash['old'] - trade_size * latest;
            cs.execute(F"""
            insert into positions values
            ('{d}', 'SP500', {round(asset['new'])}, {cash['new']});
            """);
        conn.commit();



# 6. Connect to the database and create a *cursor* object associated with the connection. Share the connection and the cursor object across the program so that you don't have to connect to and disconnect from the database in every function of the program.

# In[12]:


if __name__ == "__main__":
    with closing(sqlite3.connect("SP500.db")) as conn:
        with closing(conn.cursor()) as cs:
            prepare()
            main('2020-06-01')
    


# In[13]:


# plot your track record
conn = sqlite3.connect("SP500.db")
cs = conn.cursor()

day1 = '2020-06-01'
day1_dt = datetime.strptime(day1, '%Y-%m-%d')

cs.execute(f"""
    select theday, quantity * price + cash as wealth
    from positions as PO
    join prices as PR
    on PO.time_of_trade = (
        select time_of_trade from positions
        where time_of_trade <= PR.theday
        order by time_of_trade desc limit 1
    )
    where theday >= '{day1}';
""")

records = cs.fetchall()

def calculate_T(record, day1_dt):
    theday, wealth = record
    theday_dt = datetime.strptime(theday, '%Y-%m-%d')
    T = (theday_dt - day1_dt).days
    return (T, wealth)

records = [calculate_T(record, day1_dt) for record in records]
W = np.asarray(records)


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


fig = plt.plot(W[:,0], W[:, 1])
plt.grid()
plt.xlabel("Number of days of trading")
plt.ylabel('Total Wealth');


# In[ ]:




