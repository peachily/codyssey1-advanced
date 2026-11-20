from pathlib import Path

import numpy as np
import pandas as pd


class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_data(self, sample_size=10000, seed=42, chunksize=500000):
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
        values = data[column]
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
        values = data[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = values.lt(lower) | values.gt(upper)
        return q1, q3, lower, upper, mask

    def add_text_features(self, data):
        names = data["prod_name"]
        return data.assign(
            product_name_length=names.str.len(),
            product_name_word_count=names.str.split().str.len(),
        )

    def add_image_features(self, data, sample_size=500, seed=42):
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

        pixels = np.stack(images).astype(float) / 255
        features = pd.DataFrame({
            "article_id": image_ids,
            "image_mean": pixels.mean(axis=(1, 2, 3)),
            "image_std": pixels.std(axis=(1, 2, 3)),
        })
        return data.merge(features, on="article_id", how="left", validate="many_to_one")

    def summarize(self, data, columns):
        values = data[columns]
        return values.describe().T, values.corr()
