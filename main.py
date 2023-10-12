# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import logging
import copy
log = logging.getLogger(__name__)
pd.set_option('display.max_columns', None)

class Lease(object):

    def __init__(self, kwargs):
        self.save = True
        self.label = kwargs['label']
        # Platform
        self.operation_commision = 0.1
        self.credit_fee = 50
        self.sales_fee = 1000
        self.phone_cost = 8500
        self.insurance_fee = self.phone_cost * 0.01
        self.other_cost = self.credit_fee + self.insurance_fee + self.sales_fee
        self.total_cost = self.phone_cost + self.other_cost

        # Lease Data
        self.daily_lease_cap = 23
        self.official_px = 9899
        self.lease_ir = 0.28
        self.monthly_rent = 651
        self.deposit = 1200
        self.lease_days = 11 * 30
        self.total_sale_px = self.official_px * (1 + self.lease_ir)
        self.buyback = self.total_sale_px - self.monthly_rent * 12 - self.deposit
        self.reinvest_days = kwargs['reinvest_days']

        # Finance Data
        self.initial_cost = 30000
        self.initial_equity = kwargs['initial_equity']
        self.initial_libility = kwargs['initial_libility']
        self.initial_asset = self.initial_equity + self.initial_libility
        self.apr = 0.1
        self.finance_days = self.reinvest_days
        self.monthly_ir = self.initial_libility * self.apr / 12

        self.key_col = ['Phone Lease Num', 'Rent Income', 'Deposit', 'BuyBack', 'Phone Cost', 'Other Cost',
                        'Operation Cost', 'Interest']

        self.output_cols = ['天数','出租手机数量', '租金收入', '押金', '买断金', '手机成本', '其他成本',
                            '运营费用', '利息','现金']
        self.month_end_days = [30 * i for i in range(1, int(self.reinvest_days * 2 / 30) + 1)]
        self.operation_protection_days = kwargs['o_day']
        self.interest_protection_days = kwargs['i_day']

    def build_init_tbl(self):
        cols = self.key_col + ['Cash']
        df = pd.DataFrame(0.0, index=np.arange(self.reinvest_days * 2 + 1), columns=cols)
        return df

    def combine_orders(self, final, o):
        for i in o[o.sum(axis=1) != 0].index:
            for c in self.key_col:
                final.at[i, c] = final.loc[i, c] + o.loc[i, c]
        return final

    def create_11month_order(self, day, phone):
        """
        Create 11 month product order
        :param day:
        :param phone:
        :return:
        """
        df = self.build_init_tbl()
        df.at[day, 'Phone Lease Num'] = phone
        df.at[day, 'Deposit'] = phone * self.deposit
        df.at[day, 'Phone Cost'] = phone * self.phone_cost * -1
        df.at[day, 'Other Cost'] = phone * self.other_cost * -1
        df.at[day + self.lease_days, 'BuyBack'] = phone * self.buyback

        for i, d in enumerate(range(day, day + self.lease_days + 30, 30)):
            rent = phone * self.monthly_rent
            df.at[d, 'Rent Income'] = df.loc[d, 'Rent Income'] + rent
            if i > 0:
                next_month_end = [s for s in self.month_end_days if s > d][0]
                df.at[next_month_end, 'Operation Cost'] = (df.loc[d, 'Rent Income'] + df.loc[
                    d, 'BuyBack']) * self.operation_commision * -1

        df = df[self.key_col]
        return df

    def calc_phone_purhcase_amount(self, cash, df, day):

        if day < self.lease_days:
            reserve = df.loc[day + 1:day + self.operation_protection_days, 'Operation Cost'].sum() * -1 + df.loc[
                                                                                                          day + 1:day + self.interest_protection_days,
                                                                                                          'Interest'].sum() * -1
        else:
            reserve = df.loc[day + 1:day + self.operation_protection_days - 1, 'Operation Cost'].sum() * -1 + df.loc[
                                                                                                              day + 1:day + self.interest_protection_days - 1,
                                                                                                              'Interest'].sum() * -1

        phone = max(min(self.daily_lease_cap, int((cash - reserve) / self.total_cost)), 0)

        return phone

    def run(self):
        final_df = self.build_init_tbl()

        # Generate Interest Payment Schedule
        for i in range(30, self.finance_days + 30, 30):
            final_df.at[i, 'Interest'] = self.monthly_ir * -1

        cash = self.initial_asset - self.initial_cost
        phone = self.daily_lease_cap

        for i in range(0, self.reinvest_days * 2 + 1):
            # Fist Create Order
            if self.reinvest_days > i > 0:
                df_order = self.create_11month_order(i, phone)
                final_df = self.combine_orders(final_df, df_order)

            if i > 0:
                cashflow = final_df.loc[i, 'Rent Income'] + final_df.loc[i, 'Deposit'] + final_df.loc[i, 'BuyBack'] + \
                           final_df.loc[i, 'Phone Cost'] + final_df.loc[i, 'Other Cost'] + final_df.loc[
                               i, 'Operation Cost'] + final_df.loc[i, 'Interest']
                cash = cash + cashflow
                phone = self.calc_phone_purhcase_amount(cash, final_df, i)

            final_df.at[i, 'Cash'] = cash


        if self.save:
            final_df = final_df.reset_index()
            #final_df.columns = self.output_cols
            final_df.to_csv(r'D:\Lease\lease.csv', index=False, encoding='utf_8_sig')
        return final_df

    def show_inputs(self):
        res = []
        res.append(['手机成本', format(self.phone_cost, ',')])
        res.append(['其他成本', format(self.other_cost, ',')])
        res.append(['押金', format(self.deposit, ',')])
        res.append(['复投天数', format(self.reinvest_days, ',')])
        res.append(['手机官网售价', format(self.official_px, ',')])
        res.append(['手机租赁天数', format(self.lease_days, ',')])
        res.append(['平台手续费', str(self.operation_commision*100)+'%'])
        res.append(['每天手机租赁上限', format(self.daily_lease_cap, ',')])
        df = pd.DataFrame(res, columns=['指标', '数值'])
        print(df)

    def show_summary(self, df):
        res = []
        last_day = df[df['Operation Cost']<0].index[-1]
        a = df[df['Cash']==df['Cash'].max()].index[0]
        b = int(df['Phone Lease Num'].sum())
        c = int(df['Rent Income'].sum())
        d = int(df['Deposit'].sum())
        e = int(df['BuyBack'].sum())
        f = int(df['Phone Cost'].sum())
        g = int(df['Other Cost'].sum())
        h = int(df['Operation Cost'].sum())
        i = int(df['Interest'].sum())
        j = int(df.loc[last_day, 'Cash'])
        k = int(df['Cash'].min())

        total_return = int(j - self.initial_equity - self.initial_libility)
        total_return_rate = round(total_return/self.initial_equity*100, 2)
        annualized_return = round(((total_return_rate/100+1)**(360/a) -1)*100,2)

        res.append(['复投天数', format(self.reinvest_days, ',')])
        res.append(['投资天数', format(a, ',')])
        res.append(['总手机数', format(b, ',')])
        res.append(['总租金收入', format(c, ',')])
        res.append(['总押金收入', format(d, ',')])
        res.append(['总买断收入', format(e, ',')])
        res.append(['总手机采购成本', format(f, ',')])
        res.append(['总其他成本', format(g, ',')])
        res.append(['总平台运营成本', format(h, ',')])
        res.append(['总利息', format(i, ',')])
        res.append(['起始投资', format(int(self.initial_equity), ',')])
        res.append(['起始负债', format(self.initial_libility, ',')])
        res.append(['退出时现金', format(j, ',')])
        res.append(['最低现金', format(k, ',')])
        res.append(['总回报', format(total_return, ',')])
        res.append(['总回报率', str(total_return_rate)+'%'])
        res.append(['年化回报率', str(annualized_return)+'%'])

        df = pd.DataFrame(res, columns=['指标', self.label])
        df = df.set_index('指标')
        return df


if __name__ == '__main__':

    base = {'initial_equity':4e6, 'initial_libility':0, 'reinvest_days':360, 'o_day': 10, 'i_day':6, 'label':'Base'}
    lease = Lease(base)
    df = lease.run()
    summary = lease.show_summary(df)
    # base = {'initial_equity': 4e6, 'initial_libility': 0, 'reinvest_days': 360, 'o_day': 12, 'i_day': 6,
    #         'label': 'Base'}
    print(summary)
    inputs = [base]
    res = []
    print("============BASE==============")
    lease = Lease(base)
    lease.show_inputs()
    print("=================================")
    for i, e in enumerate(range(390, 750, 30)):
        t = copy.deepcopy(base)
        t['reinvest_days'] = e
        t['label'] = f'Scenario {i}'
        #t['o_day'] = 22- e/1e6
        inputs.append(t)

    for params in inputs:
        print(f"running {params['label']} ...")
        lease = Lease(params)
        df = lease.run()
        summary = lease.show_summary(df)
        res.append(summary)

    for i, t in enumerate(res):
        final = final.join(t) if i>0 else t

    print(final)
    final.to_csv(r'D:\Lease\performance2.csv', encoding='utf_8_sig')


