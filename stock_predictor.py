import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import ta
import optuna
from sklearn.model_selection import train_test_split

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    """
    Hisse senedi fiyat tahmini için LSTM tabanlı makine öğrenimi modeli.
    
    Bu sınıf, geçmiş hisse senedi verilerini kullanarak gelecekteki fiyatları
    tahmin etmek için LSTM kullanır.
    """
    
    def __init__(self):
        """
        StockPredictor sınıfını başlatır.
        
        Attributes:
            model: LSTM modeli
            scaler: Veri normalizasyonu için MinMaxScaler
            currency_symbol: Para birimi sembolü (varsayılan: $)
            sequence_length: LSTM modeli için kullanılacak gün sayısı
            feature_columns: Özellik sütunlarının listesi
        """
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.currency_symbol = '$'
        self.sequence_length = 60
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_stock_data(self, symbol, period='1y'):
        """
        Yahoo Finance'den hisse senedi verilerini çeker.
        
        Args:
            symbol (str): Hisse senedi sembolü (örn. 'AAPL', 'TSLA')
            period (str, optional): Veri periyodu. Varsayılan '1y' (1 yıl)
            
        Returns:
            pandas.DataFrame: Hisse senedi geçmiş verileri
            
        Raises:
            ValueError: Sembol için veri bulunamadığında
            Exception: Veri çekme sırasında oluşan diğer hatalar
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"{symbol} için veri bulunamadı.")
                
            #para birimi kontrol
            info = stock.info
            if 'currency' in info:
                currency = info['currency']
                if currency == 'USD':
                    self.currency_symbol = '$'
                elif currency == 'EUR':
                    self.currency_symbol = '€'
                elif currency == 'GBP':
                    self.currency_symbol = '£'
                elif currency == 'TRY':
                    self.currency_symbol = '₺'
                else:
                    self.currency_symbol = currency
                    
            return df
            
        except Exception as e:
            if 'symbol may be delisted' in str(e).lower():
                raise ValueError(f"{symbol} sembolü artık listelenmiyor veya geçersiz.")
            elif 'not found' in str(e).lower():
                raise ValueError(f"{symbol} sembolü bulunamadı.")
            else:
                raise Exception(f"Veri çekme hatası: {str(e)}")

    def add_technical_indicators(self, df):
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD indikatörü
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger bantları
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        
        # Volume indikatörleri
        df['Volume_MA'] = ta.volume.volume_weighted_average_price(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        
        # Momentum indikatörleri
        df['Stoch'] = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).stoch()
        
        return df

    def prepare_data(self, df):
        """
        Verileri LSTM modeli için hazırlar.
        
        Args:
            df (pandas.DataFrame): Ham hisse senedi verileri
            
        Returns:
            tuple: (X, y) şeklinde eğitim verileri
                - X: Şekil (n_samples, sequence_length, feature_columns) olan normalize edilmiş girdi verileri
                - y: Şekil (n_samples,) olan normalize edilmiş hedef değerler
                
        Raises:
            ValueError: Yeterli veri yoksa
        """
        # Teknik göstergeleri ekle
        df = self.add_technical_indicators(df)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        # Özellik sütunlarını belirle
        self.feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 
                              'BB_High', 'BB_Low', 'Volume_MA', 'Stoch']
        
        # Veriyi normalize et
        scaled_data = self.scaler.fit_transform(df[self.feature_columns])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Close price
            
        return np.array(X), np.array(y)

    def create_model(self, input_size):
        model = LSTMModel(
            input_size=input_size,
            hidden_size=50,
            num_layers=2
        ).to(self.device)
        return model

    def train_model(self, X, y):
        """
        LSTM modelini eğitir.
        
        Args:
            X (numpy.ndarray): Eğitim verileri
            y (numpy.ndarray): Hedef değerler
        """
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # PyTorch tensörlerine dönüştür
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Model oluştur
        self.model = self.create_model(input_size=X.shape[2])
        
        # Optimizer ve loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Eğitim
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs.squeeze(), y_test)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # En iyi modeli kaydet
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # En iyi modeli yükle
        self.model.load_state_dict(torch.load('best_model.pth'))

    def predict_future(self, df, days_to_predict):
        """
        Gelecekteki fiyatları tahmin eder.
        
        Args:
            df (pandas.DataFrame): Geçmiş veriler
            days_to_predict (int): Tahmin edilecek gün sayısı
            
        Returns:
            numpy.ndarray: Tahmin edilen fiyatlar (days_to_predict, 1) şeklinde
        """
        # Son sequence_length günlük veriyi al
        last_sequence = df[self.feature_columns].values[-self.sequence_length:]
        last_sequence = self.scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(days_to_predict):
                X = torch.FloatTensor(current_sequence[-self.sequence_length:]).unsqueeze(0).to(self.device)
                predicted_price = self.model(X)
                
                # Tahmin edilen fiyatı diğer özelliklerle birleştir
                new_row = current_sequence[-1].copy()
                new_row[0] = predicted_price.item()
                current_sequence = np.vstack([current_sequence, new_row])
                
                future_predictions.append(predicted_price.item())
        
        # Tahminleri orijinal ölçeğe geri dönüştür
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        dummy_array = np.zeros((len(future_predictions), len(self.feature_columns)))
        dummy_array[:, 0] = future_predictions.flatten()
        future_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        return future_predictions
    
    def test_accuracy(self, df, days_to_predict=10):
        """
        Modelin doğruluğunu test eder.
        
        Args:
            df (pandas.DataFrame): Geçmiş veriler
            days_to_predict (int, optional): Tahmin edilecek gün sayısı. Varsayılan 10
            
        Returns:
            tuple: (mae, rmse, mape, r2) şeklinde metrikler
        """
        if len(df) < days_to_predict:
            raise ValueError(f"Test için en az {days_to_predict} günlük veri gerekiyor.")
            
        actual_prices = df['Close'].values[-days_to_predict:]
        predicted_prices = self.predict_future(df, days_to_predict)
        
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predicted_prices)
        
        print(f"\nModel Performans Metrikleri:")
        print(f"Ortalama Mutlak Hata (MAE): {mae:.2f} {self.currency_symbol}")
        print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f} {self.currency_symbol}")
        print(f"Ortalama Mutlak Yüzde Hata (MAPE): {mape:.2f}%")
        print(f"R² Skoru: {r2:.2f}")
        
        return mae, rmse, mape, r2

    def plot_predictions(self, df, future_predictions, symbol):
        """
        Geçmiş verileri ve tahminleri görselleştirir.
        
        Args:
            df (pandas.DataFrame): Geçmiş veriler
            future_predictions (numpy.ndarray): Tahmin edilen değerler
            symbol (str): Hisse senedi sembolü
        """
        plt.figure(figsize=(15, 7))
        plt.plot(df.index[-100:], df['Close'].values[-100:], label='Geçmiş Fiyatlar')
        
        future_dates = pd.date_range(start=df.index[-1], periods=len(future_predictions)+1)[1:]
        plt.plot(future_dates, future_predictions, label='Tahminler', color='red')
        
        plt.title(f'{symbol} Hisse Senedi Fiyat Tahmini')
        plt.xlabel('Tarih')
        plt.ylabel(f'Fiyat ({self.currency_symbol})')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """
    Ana program fonksiyonu.
    
    Kullanıcıdan hisse senedi sembolü ve tahmin süresi alır,
    modeli eğitir ve tahminleri görselleştirir.
    
    Hata durumlarını yönetir ve kullanıcıya uygun geri bildirimler sağlar.
    """
    try:
        symbol = input("Hisse senedi sembolünü girin (örn. AAPL): ").upper()
        days_to_predict = int(input("Kaç günlük tahmin yapmak istiyorsunuz? "))
        
        if days_to_predict <= 0:
            raise ValueError("Tahmin gün sayısı pozitif olmalıdır.")
        
        predictor = StockPredictor()
        
        print(f"{symbol} için veriler çekiliyor...")
        df = predictor.get_stock_data(symbol)
        
        print("Veriler hazırlanıyor ve teknik göstergeler ekleniyor...")
        X, y = predictor.prepare_data(df)
        
        print("LSTM modeli eğitiliyor...")
        predictor.train_model(X, y)
        
        print("Gelecek tahminleri yapılıyor...")
        future_predictions = predictor.predict_future(df, days_to_predict)
        
        predictor.plot_predictions(df, future_predictions, symbol)
        predictor.test_accuracy(df, days_to_predict)
        
        print("\nTahmin Sonuçları:")
        future_dates = pd.date_range(start=df.index[-1], periods=len(future_predictions)+1)[1:]
        for date, pred in zip(future_dates, future_predictions):
            print(f"{date.strftime('%Y-%m-%d')}: {predictor.currency_symbol}{pred:.2f}")
            
    except ValueError as ve:
        print(f"Hata: {str(ve)}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 