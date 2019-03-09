import pandas as pd

df = pd.DataFrame([["London", "car", 20],
                   ["Cambridge", "car", 10],
                   ["Liverpool", "bus", 30]],
                  columns=["city", "transport", "duration"])
print(df.shape)
cat_columns = ["city", "transport"]