import sys

try:
    from data_loader import load_movielens_data
    from recommender import UserBasedRecommender
except ImportError as e:
    print("HATA: Gerekli kütüphaneler bulunamadı.")
    print(f"Detay: {e}")
    print("Lütfen sanal ortamı aktif ettiğinizden ve gereksinimleri yüklediğinizden emin olun:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def get_new_user_ratings(data):
    """
    Asks the user to rate popular movies.
    Returns a dictionary: {movie_title: rating}
    """
    print("\n👋 Hosgeldiniz! Sizi tanimak icin birkaç film sormamiz gerek.")
    print("Lütfen asagidaki filmlere 1-5 arasi puan verin (İzlemediyseniz 'gec' yazin).")
    
    # Select popular movies (most rated) to ask
    most_rated = data.groupby("movie").size().sort_values(ascending=False).head(10).index.tolist()
    
    new_ratings = {}
    for movie in most_rated:
        while True:
            response = input(f"🎥 {movie} (1-5 veya 'gec'): ").strip().lower()
            if response == 'gec' or response == '':
                break
            try:
                rating = float(response)
                if 1 <= rating <= 5:
                    new_ratings[movie] = rating
                    break
                else:
                    print("⚠️ Lütfen 1 ile 5 arasinda bir sayi girin.")
            except ValueError:
                print("⚠️ Geçersiz giris.")
                
    return new_ratings

def main():
    # 1. Veriyi Yükle
    print("Veri seti yükleniyor, lütfen bekleyin...")
    try:
        ratings = load_movielens_data()
    except Exception as e:
        print(f"Veri yüklenirken hata: {e}")
        return

    print("Veri basariyla yuklendi.")
    print(f"Toplam Satır: {len(ratings)}")
    print("-" * 30)

    # 2. Oneri Sistemini Baslat
    print("Sistem hazirlaniyor (Benzerlik matrisi hesaplaniyor)...")
    try:
        # Tam veriyi kullanalim
        recommender = UserBasedRecommender(ratings)
        print("Sistem hazir!")
        print("-" * 30)
    except Exception as e:
        print(f"Sistem baslatilirken hata olustu: {e}")
        return

    # 3. Kullanici Secimi
    print("\n🔍 Nasıl devam etmek istersiniz?")
    print("1. Mevcut bir kullanici icin öneri al (Test Modu)")
    print("2. Yeni kullaniciyim, bana öneri yap (Anket Modu)")
    
    choice = input("Seciminiz (1/2): ").strip()
    
    target_user = None
    
    if choice == "2":
        # Yeni Kullanici Senaryosu
        new_ratings = get_new_user_ratings(ratings)
        if not new_ratings:
            print("Hiçbir filme puan vermediniz. Size 'En Popüler' filmleri öneriyoruz (henüz implemente edilmedi).")
            return
            
        print("\nTesekkurler! Zevkinize uygun filmleri buluyoruz...")
        # Geçici bir ID uyduralım
        target_user = 999999
        recommender.add_user_ratings(target_user, new_ratings)
        
    else:
        # Mevcut Kullanici (Varsayilan: 1)
        target_user = 1
        print(f"\nVarsayilan olarak Kullanici {target_user} seçildi.")

    # 4. Onerileri Uret
    top_n = 5
    print(f"Kullanici ID: {target_user} icin öneriler hesaplaniyor...")
    
    try:
        # Gerçek veri setinde daha fazla komşu gerekebilir
        recommendations = recommender.recommend(target_user, k_neighbors=50, top_n=top_n)
    except Exception as e:
        print(f"Oneri uretilirken hata: {e}")
        return

    # 5. Sonuclari Goster
    print(f"\n🌟 {target_user} icin Ozel Öneriler:")
    if recommendations.empty:
        print("Üzgünüz, yeterli veri bulunamadı.")
    else:
        for i, row in recommendations.iterrows():
            print(f"{i+1}. {row['movie']:<30}  tahmini puan: {row['pred_rating']:.2f}")

if __name__ == "__main__":
    main()
