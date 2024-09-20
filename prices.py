import numpy as np
import matplotlib.pyplot as plt

class Prices:
    ELECTRICITY_PRICE_BASE = 20
    PERCENTILE_MIN = 0.1
    PERCENTILE_MAX = 99.9
    PREDICTION_WINDOW = 24

    def __init__(self, external_prices=None):
        self.external_prices = external_prices
        self.price_index = 0

        if self.external_prices is not None:
            self.MIN_PRICE = np.percentile(self.external_prices, self.PERCENTILE_MIN)
            self.MAX_PRICE = np.percentile(self.external_prices, self.PERCENTILE_MAX)
            self.predicted_prices = np.array(self.external_prices[:self.PREDICTION_WINDOW])
            self.price_index = self.PREDICTION_WINDOW
        else:
            self.MAX_PRICE = 24
            self.MIN_PRICE = 16
            self.predicted_prices = self.get_initial_prices(self.PREDICTION_WINDOW)

    def get_initial_prices(self, num_hours):
        if self.external_prices is not None:
            return np.array(self.external_prices[:num_hours])
        else:
            initial_prices = np.zeros(num_hours, dtype=np.float32)
            initial_prices[0] = self.ELECTRICITY_PRICE_BASE
            for i in range(1, num_hours):
                initial_prices[i] = self.ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin(i / 24 * 2 * np.pi))
                initial_prices[i] = max(1.0, initial_prices[i])
            return initial_prices

    def get_next_price(self):
        if self.external_prices is not None:
            new_price = self.external_prices[self.price_index % len(self.external_prices)]
            self.price_index += 1
        else:
            new_price = self.ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin((self.price_index % 24) / 24 * 2 * np.pi))
            self.price_index += 1

        return new_price

    def get_predicted_prices(self):
        # Update the prediction window by shifting and adding a new price
        new_price = self.get_next_price()
        self.predicted_prices = np.roll(self.predicted_prices, -1)
        self.predicted_prices[-1] = new_price
        return self.predicted_prices.copy()

    def plot_price_histogram(self, num_bins=50, save_path=None):
        if self.external_prices is not None:
            prices = self.external_prices
        else:
            prices = [self.get_next_price() for _ in range(24 * 7 * 52)]  # Generate a year's worth of prices

        plt.figure(figsize=(10, 6))
        plt.hist(prices, bins=num_bins, edgecolor='black')
        plt.title('Distribution of Electricity Prices')
        plt.xlabel('Price ($/MWh)')
        plt.ylabel('Frequency')
        plt.axvline(self.MIN_PRICE, color='r', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MIN}th Percentile')
        plt.axvline(self.MAX_PRICE, color='g', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MAX}th Percentile')
        plt.axvline(np.mean(prices), color='b', linestyle='dashed', linewidth=2, label='Mean Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def get_price_stats(self):
        if self.external_prices is not None:
            prices = self.external_prices
        else:
            prices = [self.get_next_price() for _ in range(24 * 7 * 52)]  # Generate a year's worth of prices

        return {
            'min': np.min(prices),
            'max': np.max(prices),
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            f'{self.PERCENTILE_MIN}th_percentile': self.MIN_PRICE,
            f'{self.PERCENTILE_MAX}th_percentile': self.MAX_PRICE
        }