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
