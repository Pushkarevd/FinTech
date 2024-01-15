from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np


class Simulation:

    def __init__(self):
        # Константы
        self.All_Lost = 0
        self.BasicOptOfferPrice = 35
        self.BasicOptOfferVol = 50
        self.BasicStore = 80
        self.InitAccount = 10000
        self.Max_Demand = 40
        self.MeanDPrice = 100
        self.OptOfferAcceptDecision = 1
        self.OptOfferBaseVolume = 40
        self.RentRate = 200
        self.Ret_Price = 70
        self.ShopStore = 30
        self.STOP_SELL = 0
        self.TransferDecision = 1
        self.TransferRate = 150
        self.TransferVol = 100
        self.Weekends = 2
        self.Qualification = 20
        self.AccumIncome = 0
        self.CurrDate = datetime.now()
        self.Income = 0
        self.dt = 1
        self.t = 0
        self.change_ind = True
        self.Account = self.InitAccount

    def step(self, init_step=False):
        self.BasicPriceRnd = self.BasicOptOfferVol * (np.random.uniform(.7, 1.3))
        self.AddPriceByTime = self.BasicOptOfferPrice * 0.03 * self.t + self.BasicOptOfferPrice * 0.01 * self.t * np.random.uniform(
            0, 1)
        self.OfferOnePrice = self.AddPriceByTime + self.BasicPriceRnd
        self.RndOfferVolume = round(self.OptOfferBaseVolume * np.random.uniform(0.75, 1.25))
        self.OfferFullPrice = self.OfferOnePrice * self.RndOfferVolume
        self.OfferAcceptPossibility = 1 if self.Account >= self.OfferFullPrice else 0
        self.SmallOptIncome = self.OfferAcceptPossibility * self.OptOfferAcceptDecision * self.RndOfferVolume

        self.TransferActualVolume = min(self.BasicStore,
                                        self.TransferVol * self.TransferDecision) if self.Account >= self.TransferRate else 0
        self.GoodsTransfer = np.floor(self.TransferActualVolume)
        self.Lost = self.ShopStore + self.GoodsTransfer - 100 if self.ShopStore + self.GoodsTransfer > 100 else 0

        self.Demand = round(self.Max_Demand * (1 - 1 / (1 + np.exp(-0.05 * (self.Ret_Price - self.MeanDPrice)))))
        self.RND_Demand = round(
            self.Demand * np.random.uniform(0.9, 1.2) + self.Demand * np.random.uniform(0.05, 0.3) * (
                    self.Qualification / 100 + (self.Weekends / 7)))
        self.SoldRet = (1 - self.STOP_SELL) * min(self.RND_Demand, self.ShopStore)
        self.Selling = self.SoldRet
        self.Income = self.Ret_Price * self.SoldRet

        self.AccumIncome += self.Income if self.CurrDate.day != 2 else -self.AccumIncome
        self.Tax = self.AccumIncome if self.CurrDate.day == 1 else 0

        self.DailySpending = min(self.RentRate + self.Tax, self.Account)
        self.TransSpend = self.TransferRate if self.TransferActualVolume > 0 else 0
        self.VAT = 0.13 * self.Income

        if not init_step:
            self.CurrDate += timedelta(days=1)

    def simulate(self, change_ind=False):
        # Инициализирующий шаг
        self.step(True)

        # Начало симуляции
        lost_arr = [self.All_Lost]
        shop_store_data = [self.ShopStore]
        basic_store_data = [self.BasicStore]
        demand_arr = [self.Demand]
        income_arr = [self.Income]

        accum_income = [0]
        tax = [0]

        for t in range(1, 200):
            if change_ind:
                change_ind_iter = True if input("Необходимо ли менять параметры в симуляции").lower() == 'y' else False
                if change_ind_iter:
                    self.TransferVol = float(input(f"TransferVolume"))
                    self.Ret_Price = float(input(f"Ret_Price"))

            self.Account += self.dt * (self.Income - self.DailySpending - self.TransSpend - self.VAT)
            self.BasicStore += self.dt * (self.SmallOptIncome - self.GoodsTransfer)
            self.ShopStore += self.dt * (self.GoodsTransfer - self.Selling - self.Lost)
            self.All_Lost += self.dt * self.Lost
            lost_arr.append(self.All_Lost)

            self.step()

            accum_income.append(self.AccumIncome)
            tax.append(self.Tax)

            demand_arr.append(self.RND_Demand)
            income_arr.append(self.Income)
            shop_store_data.append(self.ShopStore)
            basic_store_data.append(self.BasicStore)

        start_date = datetime.now()
        end_date = self.CurrDate

        delta = end_date - start_date

        days_list = [start_date]

        for i in range(delta.days + 1):
            days_list.append(start_date + timedelta(days=i))

        fig, axs = plt.subplots(3, 2, figsize=(10, 5), dpi=100)

        axs[0, 0].plot(days_list, demand_arr)
        axs[0, 0].set_title('Demand')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Values')

        axs[0, 1].plot(days_list, lost_arr)
        axs[0, 1].set_title('Lost')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Values')

        axs[1, 0].plot(days_list, basic_store_data, label='Basic store')
        axs[1, 0].plot(days_list, shop_store_data, label='Shop store')

        axs[1, 0].set_title('Stores')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Values')
        axs[1, 0].legend()

        axs[1, 1].plot(days_list, income_arr)
        axs[1, 1].set_title('Income')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Values')

        axs[2, 0].plot(days_list, accum_income)
        axs[2, 0].set_title('AccumIncome')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Values')

        axs[2, 1].plot(days_list, tax)
        axs[2, 1].set_title('Tax')
        axs[2, 1].set_xlabel('Time')
        axs[2, 1].set_ylabel('Values')

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.tight_layout(pad=0.5)
        plt.show()


if __name__ == '__main__':
    instance = Simulation()
    instance.simulate()
