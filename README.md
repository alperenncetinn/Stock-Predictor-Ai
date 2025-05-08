# Hisse Senedi Tahmin Uygulaması

Bu uygulama, LSTM (Long Short-Term Memory) derin öğrenme modeli kullanarak hisse senedi fiyatlarını tahmin eder.

## Özellikler

- Yahoo Finance API üzerinden gerçek zamanlı hisse senedi verileri
- LSTM tabanlı derin öğrenme modeli
- Geçmiş veriler ve tahminlerin görselleştirilmesi
- Kullanıcı dostu arayüz

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
python stock_predictor.py
```

## Kullanım

1. Programı çalıştırdığınızda, bir hisse senedi sembolü girmeniz istenecektir (örn. AAPL, TSLA, GOOGL)
2. Ardından, kaç günlük tahmin yapmak istediğinizi belirtin
3. Program otomatik olarak:
   - Geçmiş verileri çekecek
   - Modeli eğitecek
   - Gelecek tahminlerini yapacak
   - Sonuçları grafik olarak gösterecek

## Notlar
- Docstringlere bakarak hangi modelin ne yaptığını Türkçe olarak anlamlandırabilirsiniz.
- Sadece akademik amaçlıdır ticari amaçlı kullanmayın lütfen. 
- Sonuçlarda beklenmedik bir durum çıkarsa yada hatalı bir kısım görürseniz bilgilerinizi alperencetin.space adresindeki iletişim kısmından bana iletirseniz çok memmun olurum.
(grafik çizme kısmında bazı hatalar olabiliyor windows cihazlarda diğer sürümlerde gidermeye çalışacağım) 