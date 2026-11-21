from pathlib import Path

import numpy as np
import pandas as pd


class DataAnalyzer:
    """H&M 거래에 고객·상품 정보를 결합하고 구매 패턴을 분석한다."""

    def __init__(self, data_dir):
        """원본 CSV와 images 폴더가 있는 경로를 저장한다."""
        self.data_dir = Path(data_dir)

    def load_data(self, sample_size=10000, seed=42, chunksize=500000):
        """고객을 표본 추출하고 해당 고객의 거래에 고객·상품 정보를 붙인다."""
        articles = pd.read_csv(
            self.data_dir / "articles.csv",
            usecols=[
                "article_id", "prod_name", "product_type_name",
                "product_group_name", "colour_group_name", "index_group_name",
            ],
            dtype={"article_id": "string"},
        )
        articles["article_id"] = articles["article_id"].str.zfill(10)
        customers = pd.read_csv(
            self.data_dir / "customers.csv",
            usecols=["customer_id", "age", "club_member_status", "fashion_news_frequency"],
            dtype={"customer_id": "string"},
        )
        for table, key in [(articles, "article_id"), (customers, "customer_id")]:
            if table[key].isna().any() or table[key].duplicated().any():
                raise ValueError(f"{key} must be present and unique in its lookup table.")

        # 고객을 먼저 뽑고 거래 파일을 나눠 읽어 표본 고객의 구매 이력을 보존한다.
        sampled = customers.sample(n=min(sample_size, len(customers)), random_state=seed)
        transactions = pd.read_csv(
            self.data_dir / "transactions_train.csv",
            usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            dtype={"customer_id": "string", "article_id": "string"},
            parse_dates=["t_dat"],
            chunksize=chunksize,
        )
        selected = []
        for chunk in transactions:
            selected.append(chunk.loc[chunk["customer_id"].isin(sampled["customer_id"])])

        purchases = pd.concat(selected, ignore_index=True)
        purchases["article_id"] = purchases["article_id"].str.zfill(10)
        if purchases[["t_dat", "customer_id", "article_id", "price"]].isna().any().any():
            raise ValueError("Sampled transactions contain missing dates, IDs or prices.")

        # 여러 거래가 고객·상품 한 행과 연결되는지 검증해 병합 중 행이 불어나는 것을 막는다.
        merged = purchases.merge(
            sampled, on="customer_id", how="left", validate="many_to_one"
        ).merge(
            articles, on="article_id", how="left", validate="many_to_one", indicator=True
        )
        if len(merged) != len(purchases) or not merged["_merge"].eq("both").all():
            raise ValueError("The merge did not retain all sampled transactions with matching articles.")

        merged = merged.drop(columns="_merge")
        for column in [
            "sales_channel_id", "club_member_status", "fashion_news_frequency",
            "product_type_name", "product_group_name", "colour_group_name", "index_group_name",
        ]:
            merged[column] = merged[column].astype("category")
        merged["prod_name"] = merged["prod_name"].astype("string")
        return merged

    def handle_missing_values(self, data, column, group, strategy="median"):
        """결측값을 그룹 통계로 채우고, 그룹 통계가 없으면 전체 통계를 사용한다."""
        values = data[column]
        # transform은 그룹별 통계값을 원본의 각 행에 맞춰 돌려준다.
        group_values = data.groupby(group, observed=True, dropna=False)[column]
        if strategy == "median":
            estimates = group_values.transform("median")
            fallback = values.median()
        elif strategy == "mean":
            estimates = group_values.transform("mean")
            fallback = values.mean()
        else:
            raise ValueError("Choose 'median' or 'mean' for imputation.")

        return values.fillna(estimates).fillna(fallback)

    def detect_outliers(self, data, column, threshold=1.5):
        """IQR 경계와 이상치 여부를 반환한다. 원본 값은 그대로 둔다."""
        values = data[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        # 기본 경계는 Q1에서 1.5×IQR을 뺀 값과 Q3에 1.5×IQR을 더한 값이다.
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = values.lt(lower) | values.gt(upper)
        return q1, q3, lower, upper, mask

    def add_text_features(self, data):
        """상품명의 글자 수와 공백으로 나눈 단어 수를 새 컬럼으로 추가한다."""
        names = data["prod_name"]
        return data.assign(
            product_name_length=names.str.len(),
            product_name_word_count=names.str.split().str.len(),
        )

    def add_image_features(self, data, sample_size=500, seed=42):
        """상품 이미지의 픽셀 평균과 표준편차를 구해 거래 데이터에 붙인다."""
        from matplotlib.image import imread

        image_dir = self.data_dir / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Place the product JPG files under {image_dir}.")

        ids = data["article_id"].drop_duplicates().sort_values(ignore_index=True)
        ids = ids.sample(frac=1, random_state=seed)
        images = []
        image_ids = []
        for article_id in ids:
            path = image_dir / article_id[:3] / f"{article_id}.jpg"
            if not path.is_file():
                continue
            pixels = imread(path)
            # 가로·세로에서 픽셀 위치를 골라 32×32로 맞추고, 회색조는 RGB로 확장한다.
            rows = np.linspace(0, pixels.shape[0] - 1, 32, dtype=int)
            cols = np.linspace(0, pixels.shape[1] - 1, 32, dtype=int)
            pixels = pixels[rows[:, None], cols]
            if pixels.ndim == 2:
                pixels = np.repeat(pixels[..., None], 3, axis=2)
            images.append(pixels[..., :3])
            image_ids.append(article_id)
            if len(image_ids) == sample_size:
                break

        if not images:
            raise ValueError("No product JPG files matched the sampled transactions.")

        # (이미지 수, 높이, 너비, 채널) 배열을 0~1로 바꾸고 이미지마다 픽셀 통계를 한 번에 계산한다.
        pixels = np.stack(images).astype(float) / 255
        features = pd.DataFrame({
            "article_id": image_ids,
            "image_mean": pixels.mean(axis=(1, 2, 3)),
            "image_std": pixels.std(axis=(1, 2, 3)),
        })
        return data.merge(features, on="article_id", how="left", validate="many_to_one")

    def summarize(self, data, columns):
        """선택한 수치형 컬럼의 기술통계와 피어슨 상관행렬을 반환한다."""
        values = data[columns]
        return values.describe().T, values.corr()

    def calculate_rfm(
        self, data, customer="customer_id", date="t_dat", amount="price", as_of=None,
    ):
        """기준일까지의 고객별 RFM 지표와 점수를 계산해 고객군을 분류한다."""
        purchases = data[[customer, date, amount]].copy()
        purchases[date] = pd.to_datetime(purchases[date]).dt.normalize()
        reference = purchases[date].max() if as_of is None else pd.Timestamp(as_of).normalize()
        purchases = purchases.loc[purchases[date].le(reference)]
        if purchases.empty:
            raise ValueError("No purchases on or before the RFM reference date.")

        # 주문 ID가 없어 빈도는 서로 다른 구매일 수로, 구매금액은 원본 가격의 합으로 계산한다.
        rfm = purchases.groupby(customer).agg(
            first_purchase=(date, "min"),
            last_purchase=(date, "max"),
            frequency=(date, "nunique"),
            monetary=(amount, "sum"),
        )
        rfm["recency"] = (reference - rfm["last_purchase"]).dt.days
        # 동점에는 평균 순위를 주고 백분위 순위를 1~4점으로 바꾼다. R은 경과일이 짧을수록 고득점이다.
        for metric, ascending in [("recency", False), ("frequency", True), ("monetary", True)]:
            ranks = rfm[metric].rank(method="average", pct=True, ascending=ascending)
            rfm[f"{metric[0]}_score"] = np.ceil(ranks * 4).astype(int)

        recent = rfm["r_score"].ge(3)
        frequent = rfm["f_score"].ge(3) & rfm["frequency"].gt(1)
        high_value = rfm["m_score"].ge(3)
        # New는 관측 기간에 구매일이 하루인 최근 고객을 뜻한다.
        first_purchase = rfm["frequency"].eq(1) & rfm["recency"].le(30)
        # 조건이 겹치면 앞의 그룹을 적용해 VIP가 Loyal로 중복 분류되지 않게 한다.
        rfm["segment"] = np.select(
            [recent & frequent & high_value, recent & frequent,
             recent & first_purchase, ~recent],
            ["VIP", "Loyal", "New", "Churned"],
            default="Active",
        )
        rfm.attrs["reference_date"] = reference
        return rfm

    def summarize_segments(self, rfm):
        """고객군별 고객 수, 평균 RFM, 구매금액 합계와 비중을 집계한다."""
        summary = rfm.groupby("segment").agg(
            customers=("segment", "size"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        summary["customer_share"] = summary["customers"] / len(rfm)
        summary["monetary_share"] = summary["total_monetary"] / rfm["monetary"].sum()
        return summary.sort_values("customers", ascending=False)
