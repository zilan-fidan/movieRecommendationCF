import sys

try:
    from data_loader import load_sample_data
    from recommender import UserBasedRecommender
except ImportError as e:
    print("HATA: Gerekli kütüphaneler bulunamadı.")
    print(f"Detay: {e}")
    print("Lütfen sanal ortamı aktif ettiğinizden ve gereksinimleri yüklediğinizden emin olun:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def main():
    # 1. Veriyi Yükle
    ratings = load_sample_data()
    print("Veri basariyla yuklendi.")
    print(ratings.head())
    print("-" * 30)

    # 2. Oneri Sistemini Baslat
    try:
        recommender = UserBasedRecommender(ratings)
        print("Oneri sistemi baslatildi.")
        print("-" * 30)
    except Exception as e:
        print(f"Sistem baslatilirken hata olustu: {e}")
        return

    # 3. Onerileri Uret
    target_user = "U1"
    top_n = 5
    
    print(f"{target_user} icin oneriler uretiliyor...")
    try:
        recommendations = recommender.recommend(target_user, k_neighbors=3, top_n=top_n)
    except Exception as e:
        print(f"Oneri uretilirken hata: {e}")
        return

    # 4. Sonuclari Goster
    print(f"\n{target_user} icin Top-{top_n} oneriler:")
    for i, row in recommendations.iterrows():
        print(f"{i+1}. {row['movie']:<15}  tahmini puan: {row['pred_rating']:.2f}")

if __name__ == "__main__":
    main()
